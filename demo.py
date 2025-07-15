# demo.py - Interactive RAG Demo
import os
import json
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    print("✅ All services connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

class RAGDemo:
    def __init__(self):
        self.client_ai = client_ai
        self.cohere_client = cohere_client
        self.col = col
    
    def embed(self, text: str) -> List[float]:
        """Generate text embedding"""
        # Try OpenAI first if available
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
    
    def test_endpoints(self):
        """Test API endpoint connections"""
        print("\n🔧 Testing API endpoint connections...")
        
        # Test DigitalOcean Gradient AI
        try:
            response = self.client_ai.chat.completions.create(
                model="llama3-8b-instruct",
                messages=[{"role": "user", "content": "Please respond with 'pong' to confirm connection"}],
                max_completion_tokens=256,
                temperature=0.1
            )
            print("✅ DigitalOcean Gradient AI connection successful")
            print(f"   Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ DigitalOcean Gradient AI connection failed: {e}")
        
        # Test MongoDB
        try:
            doc_count = self.col.count_documents({})
            print(f"✅ MongoDB connection successful, found {doc_count} documents")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
        
        # Test OpenAI (if configured)
        if OPENAI_API_KEY:
            try:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test"
                )
                print("✅ OpenAI connection successful")
            except Exception as e:
                print(f"❌ OpenAI connection failed: {e}")
        else:
            print("⚠️  OpenAI API key not configured")
        
        # Test Cohere
        try:
            test_response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query="test query",
                documents=["test document 1", "test document 2"],
                top_n=1
            )
            print("✅ Cohere connection successful")
        except Exception as e:
            print(f"❌ Cohere connection failed: {e}")
    
    def ingest_data(self):
        """Read JSON files from SAMPLE_DATA directory and ingest into database"""
        print("\n📥 Starting data ingestion from SAMPLE_DATA directory...")
        
        sample_data_dir = "SAMPLE_DATA"
        
        # Check if SAMPLE_DATA directory exists
        if not os.path.exists(sample_data_dir):
            print(f"❌ {sample_data_dir} directory not found")
            return
        
        # Find all JSON files
        json_files = [f for f in os.listdir(sample_data_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"❌ No JSON files found in {sample_data_dir} directory")
            return
        
        print(f"📁 Found {len(json_files)} JSON files: {json_files}")
        
        all_tickets = []
        
        # Load data from all JSON files
        for json_file in json_files:
            file_path = os.path.join(sample_data_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_tickets.extend(data)
                    else:
                        all_tickets.append(data)
                print(f"✅ Successfully loaded {json_file}")
            except Exception as e:
                print(f"❌ Failed to load {json_file}: {e}")
        
        if not all_tickets:
            print("❌ No ticket data loaded")
            return
        
        print(f"📊 Total loaded tickets: {len(all_tickets)}")
        
        # Clear existing data
        self.col.delete_many({})
        print("🗑️  Cleared existing data")
        
        # Process and insert each ticket
        success_count = 0
        fail_count = 0
        
        for ticket in all_tickets:
            try:
                # Create searchable text
                searchable_text = f"Ticket ID: {ticket.get('ticket_id', 'N/A')}\n"
                searchable_text += f"Title: {ticket.get('title', '')}\n"
                searchable_text += f"Description: {ticket.get('description', '')}\n"
                searchable_text += f"Category: {ticket.get('category', '')}\n"
                searchable_text += f"Priority: {ticket.get('priority', '')}\n"
                searchable_text += f"Status: {ticket.get('status', '')}"
                
                # Generate embedding
                embedding = self.embed(searchable_text)
                if embedding is None:
                    print(f"❌ Failed to generate embedding for ticket {ticket.get('ticket_id', 'Unknown')}")
                    fail_count += 1
                    continue
                
                # Prepare document
                doc = {
                    "ticket_id": ticket.get("ticket_id"),
                    "title": ticket.get("title"),
                    "description": ticket.get("description"),
                    "category": ticket.get("category"),
                    "priority": ticket.get("priority"),
                    "status": ticket.get("status"),
                    "searchable_text": searchable_text,
                    "embedding": embedding
                }
                
                # Insert into database
                self.col.insert_one(doc)
                print(f"✅ Successfully inserted ticket: {ticket.get('title', 'Untitled')}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Failed to process ticket {ticket.get('ticket_id', 'Unknown')}: {e}")
                fail_count += 1
        
        print(f"\n📊 Data ingestion completed:")
        print(f"   ✅ Successfully inserted: {success_count} tickets")
        print(f"   ❌ Failed: {fail_count} tickets")
        
        # Show data distribution
        try:
            # Category distribution
            category_pipeline = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            categories = list(self.col.aggregate(category_pipeline))
            
            print(f"\n📈 Category distribution:")
            for cat in categories:
                print(f"   {cat['_id']}: {cat['count']} tickets")
            
            # Priority distribution
            priority_pipeline = [
                {"$group": {"_id": "$priority", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            priorities = list(self.col.aggregate(priority_pipeline))
            
            print(f"\n🎯 Priority distribution:")
            for pri in priorities:
                print(f"   {pri['_id']}: {pri['count']} tickets")
                
        except Exception as e:
            print(f"❌ Failed to generate statistics: {e}")
    
    def show_sample_questions(self):
        """Display sample questions for easy copy-paste"""
        print("\n💡 Sample Questions (copy and paste):")
        print("   1. What login issues do we have?")
        print("   2. What payment problems exist?")
        print("   3. What are the high priority issues?")
        print("   4. What mobile app crashes are reported?")
        print("   5. What security vulnerabilities need attention?")
        print("   6. What feature requests are pending?")
        print("   7. What network connectivity problems exist?")
        print("   8. What database errors are occurring?")
        print("   9. What UI/UX issues need fixing?")
        print("   10. What performance problems are reported?")
    
    def rag_query(self, query: str, k: int = 3) -> str:
        """Execute RAG query"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embed(query)
        if query_embedding is None:
            return "❌ Failed to generate query embedding"
        
        # Vector search
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
                    "ticket_id": 1,
                    "title": 1,
                    "description": 1,
                    "category": 1,
                    "priority": 1,
                    "status": 1,
                    "searchable_text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.col.aggregate(pipeline))
            
            if not results:
                return "❌ No relevant documents found"
            
            print(f"📋 Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                print(f"  {i}. (score: {doc['score']:.4f}) {doc['title']}")
            
            # Prepare context
            context = "\n\n".join([
                f"Ticket {doc['ticket_id']}: {doc['title']}\n{doc['description']}\nCategory: {doc['category']}, Priority: {doc['priority']}, Status: {doc['status']}"
                for doc in results
            ])
            
            # Generate response
            prompt = f"""Based on the following support ticket information, please answer the user's question.

Support ticket information:
{context}

User question: {query}

Please provide a helpful and accurate answer based on the ticket information:"""
            
            response = self.client_ai.chat.completions.create(
                model="llama3-8b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant. Answer questions based on the provided support ticket information."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Query execution failed: {e}"
    
    def reranking_demo(self):
        """Re-ranking demo"""
        print("\n🎯 Re-ranking Demo")
        
        # Check if data exists
        doc_count = self.col.count_documents({})
        if doc_count == 0:
            print("❌ No data found in database")
            print("Please run data ingestion first (option 2)")
            return
        
        print(f"✅ Data ingestion completed, vector search index created")
        print(f"📊 Found {doc_count} documents in database")
        
        self.show_sample_questions()
        
        while True:
            query = input("\n💬 Please enter your question (or type 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Perform initial search
            query_embedding = self.embed(query)
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                continue
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embedding-index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": 50,
                        "limit": 6
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
                results = list(self.col.aggregate(pipeline))
                
                if not results:
                    print("❌ No relevant documents found")
                    continue
                
                print(f"\n📋 Original vector search results:")
                for i, doc in enumerate(results[:3], 1):
                    print(f"  {i}. (score: {doc['score']:.4f}) {doc['searchable_text'][:100]}...")
                
                # Cohere re-ranking
                print(f"\n🔄 Cohere re-ranking...")
                
                try:
                    # Prepare documents for Cohere rerank
                    documents = [doc['searchable_text'] for doc in results]
                    
                    # Call Cohere rerank API
                    rerank_response = self.cohere_client.rerank(
                        model="rerank-v3.5",
                        query=query,
                        documents=documents,
                        top_n=min(len(documents), 3)
                    )
                    
                    # Reorder results based on Cohere ranking
                    cohere_results = []
                    for result in rerank_response.results:
                        original_doc = results[result.index]
                        original_doc['cohere_score'] = result.relevance_score
                        cohere_results.append(original_doc)
                    
                    print(f"\n🎯 Cohere re-ranking results:")
                    for i, doc in enumerate(cohere_results, 1):
                        score = doc.get('cohere_score', 0)
                        print(f"  {i}. (Cohere score: {score:.4f}) {doc['searchable_text'][:100]}...")
                        
                except Exception as e:
                    print(f"❌ Cohere re-ranking failed: {e}")
                    print("  Falling back to original order...")
                    cohere_results = results[:3]
                    for i, doc in enumerate(cohere_results, 1):
                        print(f"  {i}. {doc['searchable_text'][:100]}...")
                
                # Comparison Summary
                print(f"\n📊 Re-ranking Comparison Summary:")
                print(f"   🔹 Original Vector Search: Top result has score {results[0]['score']:.4f}")
                if cohere_results and cohere_results[0].get('cohere_score'):
                    print(f"   🔹 Cohere Re-ranking: Top result has score {cohere_results[0]['cohere_score']:.4f}")
                    if cohere_results[0]['searchable_text'] != results[0]['searchable_text']:
                        print(f"   ✨ Re-ranking changed the top result!")
                    else:
                        print(f"   ✅ Re-ranking confirmed the top result")
                
                # Generate final answer based on Cohere results
                context = "\n\n".join([doc['searchable_text'] for doc in cohere_results])
                
                final_prompt = f"""Based on the following information, please answer the user's question.

Information:
{context}

User question: {query}

Please provide a helpful and accurate answer:"""
                
                try:
                    final_response = self.client_ai.chat.completions.create(
                        model="llama3-8b-instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided information."},
                            {"role": "user", "content": final_prompt}
                        ],
                        max_completion_tokens=512,
                        temperature=0.7
                    )
                    
                    print(f"\n🤖 AI Generated Answer:")
                    print(f"{final_response.choices[0].message.content}")
                    
                except Exception as e:
                    print(f"❌ Failed to generate final answer: {e}")
                
            except Exception as e:
                print(f"❌ Search failed: {e}")
            
            # Show sample questions again after each query
            self.show_sample_questions()
    
    def show_vector_index_config(self):
        """Display vector index configuration"""
        print("\n📋 Vector Search Index Configuration")
        print("=" * 50)
        
        config = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,
                    "similarity": "cosine"
                }
            ]
        }
        
        print("Index name: embedding-index")
        print("Configuration:")
        print(json.dumps(config, indent=2))
        
        print("\nTo create this index in MongoDB Atlas:")
        print("1. Go to your MongoDB Atlas cluster")
        print("2. Navigate to Search -> Create Search Index")
        print("3. Select 'Vector Search'")
        print("4. Choose database: ai_demo, collection: tickets")
        print("5. Set index name: embedding-index")
        print("6. Use the above JSON configuration")

def main():
    """Main function"""
    demo = RAGDemo()
    
    while True:
        print("\n" + "="*50)
        print("🚀 RAG Demo System")
        print("="*50)
        print("1. Test endpoint connections")
        print("2. Data ingestion")
        print("3. RAG query demo")
        print("4. Re-ranking demo")
        print("5. Show vector index configuration")
        print("6. Exit")
        
        choice = input("\nPlease select an option (1-6): ").strip()
        
        if choice == "1":
            demo.test_endpoints()
        elif choice == "2":
            demo.ingest_data()
        elif choice == "3":
            demo.show_sample_questions()
            while True:
                query = input("\n💬 Please enter your question (or type 'quit' to return to main menu): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    result = demo.rag_query(query)
                    print(f"\n🤖 Answer:\n{result}")
                    demo.show_sample_questions()  # Show sample questions again after each query
        elif choice == "4":
            demo.reranking_demo()
        elif choice == "5":
            demo.show_vector_index_config()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option, please select 1-6")

if __name__ == "__main__":
    main() 