# 4_reranking_demo.py - Re-ranking Demo Script
import os
import sys
from typing import List, Dict
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
from cohere import ClientV2 as CohereClient

# Load environment variables
load_dotenv()

DO_GENAI_KEY = os.getenv("DO_GENAI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not DO_GENAI_KEY or not MONGODB_URI:
    print("❌ Please ensure DO_GENAI_KEY and MONGODB_URI are set in the .env file")
    exit(1)

if not COHERE_API_KEY:
    print("❌ Please ensure COHERE_API_KEY is set in the .env file")
    exit(1)

# Initialize clients
client_ai = OpenAI(
    api_key=DO_GENAI_KEY,
    base_url="https://inference.do-ai.run/v1"
)

# Initialize Cohere client
cohere_client = CohereClient(api_key=COHERE_API_KEY)

try:
    client_db = MongoClient(MONGODB_URI)
    client_db.admin.command('ping')
    col = client_db.ai_demo.tickets
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

class ReRanker:
    """Re-ranking functionality class"""
    
    def __init__(self):
        self.client_ai = client_ai
        self.cohere_client = cohere_client
    
    def embed(self, text: str) -> List[float]:
        """Generate text embedding"""
        # Try OpenAI first if available
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            try:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                resp = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return resp.data[0].embedding
            except Exception as e:
                print(f"⚠️  OpenAI embedding failed: {e}")
        
        # Fallback to hash-based embedding for demo purposes
        print("🔧 Using hash-based pseudo-embedding (DigitalOcean Gradient AI doesn't support embedding API)")
        return self.embed_with_simple_hash(text)
    
    def embed_with_simple_hash(self, text: str) -> List[float]:
        """Generate simple hash-based pseudo-embedding for demo purposes"""
        import hashlib
        import struct
        
        # Create hash of the text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert to 1536-dimensional vector (same as OpenAI text-embedding-3-small)
        embedding = []
        for i in range(0, len(hash_bytes), 2):
            if i + 1 < len(hash_bytes):
                # Combine two bytes and normalize to [-1, 1]
                val = struct.unpack('H', hash_bytes[i:i+2])[0]
                normalized_val = (val / 32767.5) - 1.0
                embedding.append(normalized_val)
        
        # Pad or truncate to 1536 dimensions
        while len(embedding) < 1536:
            embedding.extend(embedding[:min(len(embedding), 1536 - len(embedding))])
        
        return embedding[:1536]
    
    def cohere_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Cohere re-ranking using rerank-v3.5"""
        print("🔄 Executing Cohere re-ranking...")
        
        try:
            # Prepare documents for Cohere rerank
            doc_texts = [doc['searchable_text'] for doc in documents]
            
            # Call Cohere rerank API
            rerank_response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=doc_texts,
                top_n=min(len(documents), top_k)
            )
            
            # Reorder results based on Cohere ranking
            cohere_results = []
            for result in rerank_response.results:
                original_doc = documents[result.index].copy()
                original_doc['cohere_score'] = result.relevance_score
                cohere_results.append(original_doc)
            
            print(f"🎯 Cohere re-ranking results:")
            for i, doc in enumerate(cohere_results, 1):
                score = doc.get('cohere_score', 0)
                print(f"  {i}. (Cohere score: {score:.4f}) {doc['searchable_text'][:100]}...")
            
            return cohere_results
            
        except Exception as e:
            print(f"❌ Cohere re-ranking failed: {e}")
            print("  Falling back to original order...")
            return documents[:top_k]
    


def initial_search(query: str, k: int = 6) -> List[Dict]:
    """Perform initial vector search"""
    print(f"🔍 Performing initial vector search for: '{query}'")
    
    # Generate query embedding
    reranker = ReRanker()
    query_embedding = reranker.embed(query)
    
    if query_embedding is None:
        print("❌ Failed to generate query embedding")
        return []
    
    # Vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding-index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "searchable_text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(col.aggregate(pipeline))
        
        if not results:
            print("❌ No documents found")
            return []
        
        print(f"📋 Found {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. (score: {doc['score']:.4f}) {doc['searchable_text']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return []

def compare_rankings(query: str):
    """Compare original vector search vs Cohere re-ranking"""
    print(f"\n{'='*60}")
    print(f"🎯 Re-ranking Comparison for Query: '{query}'")
    print(f"{'='*60}")
    
    # Initial search
    documents = initial_search(query, k=6)
    if not documents:
        return
    
    reranker = ReRanker()
    
    # 1. Original vector search results
    print(f"\n📊 Original Vector Search Results:")
    for i, doc in enumerate(documents[:3], 1):
        print(f"  {i}. (score: {doc['score']:.4f}) {doc['searchable_text'][:100]}...")
    
    # 2. Cohere re-ranking
    print(f"\n🎯 Enhanced with Cohere Re-ranking:")
    cohere_results = reranker.cohere_rerank(query, documents.copy(), top_k=3)
    
    # Comparison summary
    print(f"\n📊 Re-ranking Comparison Summary:")
    print(f"   🔹 Original Vector Search: Top result has score {documents[0]['score']:.4f}")
    if cohere_results and cohere_results[0].get('cohere_score'):
        print(f"   🔹 Cohere Re-ranking: Top result has score {cohere_results[0]['cohere_score']:.4f}")
        if cohere_results[0]['searchable_text'] != documents[0]['searchable_text']:
            print(f"   ✨ Re-ranking changed the top result!")
        else:
            print(f"   ✅ Re-ranking confirmed the top result")
    
    print(f"\n{'='*60}")

def interactive_demo():
    """Interactive re-ranking comparison demo"""
    print("\n🎯 Interactive Re-ranking Comparison Demo")
    print("📊 Compare Original Vector Search vs Cohere Re-ranking")
    print("💡 You can ask questions about support tickets, for example:")
    print("   - What login issues do we have?")
    print("   - What payment problems exist?")
    print("   - What are the high priority issues?")
    print("\nType 'quit' to exit")
    
    while True:
        query = input("\n💬 Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        compare_rankings(query)

def main():
    """Main function"""
    print("🚀 Re-ranking Comparison Demo")
    print("📊 Vector Search vs Cohere Re-ranking")
    print("=" * 50)
    
    # Check data
    doc_count = col.count_documents({})
    if doc_count == 0:
        print("❌ No documents found in database")
        print("Please run data ingestion first")
        return
    
    print(f"📊 Found {doc_count} documents in database")
    
    choice = input("\nChoose mode:\n1. Interactive demo (enter 1)\n2. Quick test (enter 2)\nPlease select: ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        # Quick test with sample queries
        sample_queries = [
            "login problems",
            "payment issues", 
            "mobile app crashes"
        ]
        
        for query in sample_queries:
            compare_rankings(query)
            input("\nPress Enter to continue to next query...")
    else:
        print("❌ Invalid selection")

if __name__ == "__main__":
    main() 