# 4_reranking_demo.py - Re-ranking Demo
import os
import sys
import math
from typing import List, Dict, Tuple
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DO_GENAI_KEY = os.getenv("DO_GENAI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not DO_GENAI_KEY or not MONGODB_URI:
    print("❌ Please ensure DO_GENAI_KEY and MONGODB_URI are set in the .env file")
    exit(1)

# Initialize clients
client_ai = OpenAI(
    api_key=DO_GENAI_KEY,
    base_url="https://inference.do-ai.run/v1"
)

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
    
    def embed(self, text: str) -> List[float]:
        """Generate text embedding"""
        try:
            resp = self.client_ai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def semantic_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Semantic similarity-based re-ranking"""
        print("🔄 Executing semantic re-ranking...")
        
        query_embedding = self.embed(query)
        if query_embedding is None:
            return documents[:top_k]
        
        # Calculate similarity between each document and query
        for doc in documents:
            doc_embedding = self.embed(doc['text'])
            if doc_embedding:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                doc['semantic_score'] = similarity
            else:
                doc['semantic_score'] = 0
        
        # Sort by semantic similarity
        reranked = sorted(documents, key=lambda x: x.get('semantic_score', 0), reverse=True)
        
        print(f"📊 Semantic re-ranking results:")
        for i, doc in enumerate(reranked[:top_k], 1):
            score = doc.get('semantic_score', 0)
            print(f"  {i}. (semantic score: {score:.4f}) {doc['text']}")
        
        return reranked[:top_k]
    
    def llm_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """LLM-based re-ranking"""
        print("🔄 Executing LLM re-ranking...")
        
        # Prepare document list
        doc_list = []
        for i, doc in enumerate(documents):
            doc_list.append(f"{i+1}. {doc['text']}")
        
        docs_text = "\n".join(doc_list)
        
        prompt = f"""
Please rank the following documents by relevance to the query. Return only the ranked numbers separated by commas.

Query: {query}

Documents:
{docs_text}

Return the most relevant {min(top_k, len(documents))} document numbers (e.g., 3,1,5):
"""
        
        try:
            response = self.client_ai.chat.completions.create(
                model="llama3-8b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ranks documents by relevance."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=50,
                temperature=0.1
            )
            
            ranking_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM ranking result: {ranking_text}")
            
            # Parse ranking results
            try:
                rankings = [int(x.strip()) - 1 for x in ranking_text.split(',')]  # Convert to 0-based index
                reranked = []
                
                for rank in rankings:
                    if 0 <= rank < len(documents):
                        doc = documents[rank].copy()
                        doc['llm_rank'] = len(reranked) + 1
                        reranked.append(doc)
                
                # Add remaining documents
                used_indices = set(rankings)
                for i, doc in enumerate(documents):
                    if i not in used_indices and len(reranked) < top_k:
                        doc_copy = doc.copy()
                        doc_copy['llm_rank'] = len(reranked) + 1
                        reranked.append(doc_copy)
                
                print(f"📊 LLM re-ranking results:")
                for i, doc in enumerate(reranked[:top_k], 1):
                    print(f"  {i}. {doc['text']}")
                
                return reranked[:top_k]
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  LLM ranking parsing failed, using original order: {e}")
                return documents[:top_k]
                
        except Exception as e:
            print(f"❌ LLM re-ranking failed: {e}")
            return documents[:top_k]
    
    def hybrid_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Hybrid re-ranking: combining vector search scores, semantic similarity and LLM ranking"""
        print("🔄 Executing hybrid re-ranking...")
        
        # 1. Semantic similarity
        query_embedding = self.embed(query)
        if query_embedding:
            for doc in documents:
                doc_embedding = self.embed(doc['text'])
                if doc_embedding:
                    semantic_score = self.cosine_similarity(query_embedding, doc_embedding)
                    doc['semantic_score'] = semantic_score
                else:
                    doc['semantic_score'] = 0
        
        # 2. Normalize scores
        vector_scores = [doc.get('score', 0) for doc in documents]
        semantic_scores = [doc.get('semantic_score', 0) for doc in documents]
        
        # Min-Max normalization
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)
            min_score, max_score = min(scores), max(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        norm_vector = normalize_scores(vector_scores)
        norm_semantic = normalize_scores(semantic_scores)
        
        # 3. Calculate hybrid score
        for i, doc in enumerate(documents):
            # Weights: vector search 40%, semantic similarity 60%
            hybrid_score = 0.4 * norm_vector[i] + 0.6 * norm_semantic[i]
            doc['hybrid_score'] = hybrid_score
        
        # 4. Sort
        reranked = sorted(documents, key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        print(f"📊 Hybrid re-ranking results:")
        for i, doc in enumerate(reranked[:top_k], 1):
            vector_score = doc.get('score', 0)
            semantic_score = doc.get('semantic_score', 0)
            hybrid_score = doc.get('hybrid_score', 0)
            print(f"  {i}. (hybrid: {hybrid_score:.4f}, vector: {vector_score:.4f}, semantic: {semantic_score:.4f}) {doc['text']}")
        
        return reranked[:top_k]

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
                "text": 1,
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
            print(f"  {i}. (score: {doc['score']:.4f}) {doc['text']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return []

def compare_rankings(query: str):
    """Compare different re-ranking methods"""
    print(f"\n{'='*60}")
    print(f"🎯 Re-ranking Comparison for Query: '{query}'")
    print(f"{'='*60}")
    
    # Initial search
    documents = initial_search(query, k=6)
    if not documents:
        return
    
    reranker = ReRanker()
    
    # 1. Original ranking
    print(f"\n1️⃣ Original Vector Search Results:")
    for i, doc in enumerate(documents[:3], 1):
        print(f"  {i}. (score: {doc['score']:.4f}) {doc['text']}")
    
    # 2. Semantic re-ranking
    print(f"\n2️⃣ Semantic Re-ranking:")
    semantic_results = reranker.semantic_rerank(query, documents.copy(), top_k=3)
    
    # 3. LLM re-ranking
    print(f"\n3️⃣ LLM Re-ranking:")
    llm_results = reranker.llm_rerank(query, documents.copy(), top_k=3)
    
    # 4. Hybrid re-ranking
    print(f"\n4️⃣ Hybrid Re-ranking:")
    hybrid_results = reranker.hybrid_rerank(query, documents.copy(), top_k=3)
    
    print(f"\n{'='*60}")

def interactive_demo():
    """Interactive re-ranking demo"""
    print("\n🎯 Interactive Re-ranking Demo")
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
    print("🚀 Re-ranking Demo System")
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