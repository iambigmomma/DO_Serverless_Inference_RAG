# demo.py - Integrated Demo Script
import os
import sys
import time
import math
import json
from typing import List, Dict
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DO_GENAI_KEY = os.getenv("DO_GENAI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGDemo:
    def __init__(self):
        self.do_client = None
        self.openai_client = None
        self.client_db = None
        self.col = None
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize client connections"""
        if not DO_GENAI_KEY:
            print("❌ Error: DO_GENAI_KEY not found in environment variables")
            return False
        
        if not MONGODB_URI:
            print("❌ Error: MONGODB_URI not found in environment variables")
            return False
            
        if not OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Please add your OpenAI API Key to the .env file")
            return False
        
        try:
            # DigitalOcean AI client (for chat)
            self.do_client = OpenAI(
                api_key=DO_GENAI_KEY,
                base_url="https://inference.do-ai.run/v1"
            )
            
            # OpenAI client (for embeddings)
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            
            # MongoDB client
            self.client_db = MongoClient(MONGODB_URI)
            self.client_db.admin.command('ping')
            self.col = self.client_db.ai_demo.tickets
            
            print("✅ All services connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def test_endpoint(self):
        """Test DigitalOcean endpoint"""
        print("\n🧪 Testing DigitalOcean Serverless Inference...")
        try:
            resp = self.do_client.chat.completions.create(
                model="llama3-8b-instruct",
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5
            )
            print(f"✅ Test successful! Response: {resp.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def embed(self, text: str) -> list[float]:
        """Generate text embeddings (using OpenAI API)"""
        try:
            resp = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None
    
    def ingest_data(self):
        """Ingest sample support tickets with embeddings from JSON files"""
        
        print("🔄 Starting data ingestion from SAMPLE_DATA directory...")
        
        # Load data from JSON files in SAMPLE_DATA directory
        sample_data_dir = "SAMPLE_DATA"
        all_tickets = []
        
        if not os.path.exists(sample_data_dir):
            print(f"❌ {sample_data_dir} directory not found!")
            print("Please create the SAMPLE_DATA directory and add JSON files with ticket data.")
            return
        
        # Find all JSON files in SAMPLE_DATA directory
        json_files = [f for f in os.listdir(sample_data_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"❌ No JSON files found in {sample_data_dir} directory!")
            print("Please add JSON files with ticket data to the SAMPLE_DATA directory.")
            return
        
        print(f"📁 Found {len(json_files)} JSON file(s): {', '.join(json_files)}")
        
        # Load data from each JSON file
        for json_file in json_files:
            file_path = os.path.join(sample_data_dir, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tickets = json.load(f)
                    if isinstance(tickets, list):
                        all_tickets.extend(tickets)
                        print(f"✅ Loaded {len(tickets)} tickets from {json_file}")
                    else:
                        print(f"⚠️  Warning: {json_file} does not contain a JSON array")
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
                continue
        
        if not all_tickets:
            print("❌ No valid ticket data found in JSON files!")
            return
        
        print(f"📊 Total tickets to process: {len(all_tickets)}")
        
        # Clear existing data
        print("🗑️  Clearing existing data...")
        self.col.delete_many({})
        
        # Process each ticket
        successful_inserts = 0
        failed_inserts = 0
        
        for ticket in all_tickets:
            try:
                # Create searchable text
                searchable_text = f"{ticket.get('title', '')} {ticket.get('description', '')} {ticket.get('category', '')}"
                
                # Generate embedding
                print(f"📝 Processing: {ticket.get('title', 'Unknown Title')}")
                embedding = self.embed(searchable_text)
                
                if embedding:
                    # Add embedding and searchable text to ticket
                    ticket['embedding'] = embedding
                    ticket['searchable_text'] = searchable_text
                    
                    # Insert into MongoDB
                    self.col.insert_one(ticket)
                    print(f"✅ Inserted: {ticket.get('ticket_id', ticket.get('_id', 'Unknown ID'))}")
                    successful_inserts += 1
                else:
                    print(f"❌ Skipping {ticket.get('ticket_id', 'Unknown')} due to embedding failure")
                    failed_inserts += 1
                    
            except Exception as e:
                print(f"❌ Failed to process ticket {ticket.get('ticket_id', 'Unknown')}: {e}")
                failed_inserts += 1
        
        print(f"\n✅ Data ingestion completed!")
        print(f"📊 Successfully inserted: {successful_inserts} tickets")
        if failed_inserts > 0:
            print(f"⚠️  Failed to insert: {failed_inserts} tickets")
        print(f"📊 Total documents in collection: {self.col.count_documents({})}")
        
        # Show category distribution
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        categories = list(self.col.aggregate(pipeline))
        
        if categories:
            print(f"\n📈 Category distribution:")
            for cat in categories:
                print(f"  - {cat['_id']}: {cat['count']} tickets")
        
        # Show priority distribution
        pipeline = [
            {"$group": {"_id": "$priority", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        priorities = list(self.col.aggregate(pipeline))
        
        if priorities:
            print(f"\n🎯 Priority distribution:")
            for pri in priorities:
                print(f"  - {pri['_id']}: {pri['count']} tickets")
    
    def search(self, query_emb: list[float], k: int = 3) -> list[Dict]:
        """Vector search"""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embedding-index",
                        "path": "embedding",
                        "queryVector": query_emb,
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
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.col.aggregate(pipeline))
            if not results:
                print("⚠️  No relevant documents found")
                return []
            
            print(f"🔍 Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                score = doc.get('score', 0)
                print(f"  {i}. (similarity: {score:.4f}) {doc['title']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []

    def search_with_rerank(self, query_emb: list[float], query: str, k: int = 6, rerank_method: str = "hybrid") -> List[Dict]:
        """Search with re-ranking"""
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "embedding-index",
                        "path": "embedding",
                        "queryVector": query_emb,
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
            
            results = list(self.col.aggregate(pipeline))
            if not results:
                print("⚠️  No relevant documents found")
                return []
            
            print(f"🔍 Initial search found {len(results)} documents")
            
            # Execute re-ranking
            if rerank_method == "semantic":
                reranked = self.semantic_rerank(query, results, top_k=3)
            elif rerank_method == "llm":
                reranked = self.llm_rerank(query, results, top_k=3)
            elif rerank_method == "hybrid":
                reranked = self.hybrid_rerank(query, results, top_k=3)
            else:
                reranked = results[:3]
            
            return reranked
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def semantic_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Semantic re-ranking"""
        print("🔄 Executing semantic re-ranking...")
        
        query_embedding = self.embed(query)
        if query_embedding is None:
            return documents[:top_k]
        
        for doc in documents:
            doc_embedding = self.embed(doc['searchable_text'])
            if doc_embedding:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                doc['semantic_score'] = similarity
            else:
                doc['semantic_score'] = 0
        
        reranked = sorted(documents, key=lambda x: x.get('semantic_score', 0), reverse=True)
        
        print(f"📊 Semantic re-ranking results:")
        for i, doc in enumerate(reranked[:top_k], 1):
            score = doc.get('semantic_score', 0)
            print(f"  {i}. (semantic score: {score:.4f}) {doc['title']}")
        
        return reranked[:top_k]
    
    def llm_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """LLM re-ranking"""
        print("🔄 Executing LLM re-ranking...")
        
        doc_list = []
        for i, doc in enumerate(documents):
            doc_list.append(f"{i+1}. {doc['title']} - {doc['description'][:50]}...")
        
        docs_text = "\n".join(doc_list)
        
        prompt = f"""Rank the documents by relevance to the query. Return only numbers separated by commas.

Query: {query}

Documents:
{docs_text}

Return the most relevant {min(top_k, len(documents))} document numbers (format: 3,1,5):"""
        
        try:
            response = self.do_client.chat.completions.create(
                model="llama3-8b-instruct",
                messages=[
                    {"role": "system", "content": "You are a document ranking assistant. Return only the document numbers in order of relevance, separated by commas."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=20,
                temperature=0.1
            )
            
            ranking_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM ranking: {ranking_text}")
            
            try:
                # Extract numbers
                import re
                numbers = re.findall(r'\d+', ranking_text)
                rankings = [int(x) - 1 for x in numbers[:top_k]]
                
                reranked = []
                for rank in rankings:
                    if 0 <= rank < len(documents):
                        reranked.append(documents[rank])
                
                print(f"📊 LLM re-ranking results:")
                for i, doc in enumerate(reranked[:top_k], 1):
                    print(f"  {i}. {doc['title']}")
                
                return reranked[:top_k]
                
            except (ValueError, IndexError):
                print("⚠️  LLM ranking parsing failed, using original order")
                return documents[:top_k]
                
        except Exception as e:
            print(f"❌ LLM re-ranking failed: {e}")
            return documents[:top_k]
    
    def hybrid_rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Hybrid re-ranking"""
        print("🔄 Executing hybrid re-ranking...")
        
        query_embedding = self.embed(query)
        if query_embedding:
            for doc in documents:
                doc_embedding = self.embed(doc['searchable_text'])
                if doc_embedding:
                    semantic_score = self.cosine_similarity(query_embedding, doc_embedding)
                    doc['semantic_score'] = semantic_score
                else:
                    doc['semantic_score'] = 0
        
        # Normalize scores
        vector_scores = [doc.get('score', 0) for doc in documents]
        semantic_scores = [doc.get('semantic_score', 0) for doc in documents]
        
        def normalize_scores(scores):
            if not scores or max(scores) == min(scores):
                return [0.5] * len(scores)
            min_score, max_score = min(scores), max(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]
        
        norm_vector = normalize_scores(vector_scores)
        norm_semantic = normalize_scores(semantic_scores)
        
        for i, doc in enumerate(documents):
            hybrid_score = 0.4 * norm_vector[i] + 0.6 * norm_semantic[i]
            doc['hybrid_score'] = hybrid_score
        
        reranked = sorted(documents, key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        print(f"📊 Hybrid re-ranking results:")
        for i, doc in enumerate(reranked[:top_k], 1):
            hybrid_score = doc.get('hybrid_score', 0)
            print(f"  {i}. (hybrid score: {hybrid_score:.4f}) {doc['title']}")
        
        return reranked[:top_k]
    
    def ask_llm(self, contexts: List[Dict], question: str):
        """LLM Q&A"""
        if not contexts:
            print("❌ No context information available")
            return
        
        # Format context information
        context_text = ""
        for i, doc in enumerate(contexts, 1):
            context_text += f"{i}. Ticket {doc['ticket_id']}: {doc['title']}\n"
            context_text += f"   Description: {doc['description']}\n"
            context_text += f"   Category: {doc['category']} | Priority: {doc['priority']} | Status: {doc['status']}\n\n"
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional customer support assistant. Answer user questions based on the provided support ticket information. Provide accurate and helpful responses."
            },
            {
                "role": "user", 
                "content": f"Support ticket information:\n{context_text}\nQuestion: {question}"
            }
        ]
        
        try:
            print("\n💬 AI Response:")
            print("-" * 50)
            
            stream = self.do_client.chat.completions.create(
                model="llama3-8b-instruct",
                messages=messages,
                stream=True,
                max_completion_tokens=256,
                temperature=0.7
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            
            print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"❌ LLM response failed: {e}")
    
    def query_demo(self):
        """Query demonstration"""
        print("\n🤔 RAG Query Demo")
        print("💡 You can ask questions about support tickets, for example:")
        print("   - What login issues do we have?")
        print("   - What payment-related problems exist?")
        print("   - What are the high priority issues?")
        
        while True:
            question = input("\nEnter your question (type 'quit' to exit): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print(f"\n❓ Question: {question}")
            
            # Generate embedding
            query_embedding = self.embed(question)
            if query_embedding is None:
                continue
            
            # Search relevant documents
            contexts = self.search(query_embedding)
            if not contexts:
                continue
            
            # Generate answer
            self.ask_llm(contexts, question)
    
    def reranking_demo(self):
        """Re-ranking demonstration"""
        print("\n🎯 Re-ranking Demo")
        print("💡 Re-ranking improves retrieval quality, especially for complex queries")
        print("💡 You can ask questions about support tickets, for example:")
        print("   - What technical issues need urgent attention?")
        print("   - What user experience problems do we have?")
        
        while True:
            question = input("\nEnter your question (type 'quit' to exit): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print(f"\n❓ Question: {question}")
            
            # Generate embedding
            query_embedding = self.embed(question)
            if query_embedding is None:
                continue
            
            # Compare different re-ranking methods
            print("\n" + "=" * 50)
            print("📊 Re-ranking Method Comparison")
            print("=" * 50)
            
            # 1. Original search
            print("\n1️⃣ Original Vector Search:")
            original_contexts = self.search(query_embedding, k=3)
            
            # 2. Semantic re-ranking
            print("\n2️⃣ Semantic Re-ranking:")
            semantic_contexts = self.search_with_rerank(query_embedding, question, k=6, rerank_method="semantic")
            
            # 3. LLM re-ranking
            print("\n3️⃣ LLM Re-ranking:")
            llm_contexts = self.search_with_rerank(query_embedding, question, k=6, rerank_method="llm")
            
            # 4. Hybrid re-ranking
            print("\n4️⃣ Hybrid Re-ranking:")
            hybrid_contexts = self.search_with_rerank(query_embedding, question, k=6, rerank_method="hybrid")
            
            # Use hybrid re-ranking results for answer generation
            print("\n💬 Generating answer using hybrid re-ranking results:")
            if hybrid_contexts:
                self.ask_llm(hybrid_contexts, question)
    
    def show_vector_index_config(self):
        """Show Vector Search Index configuration"""
        print("\n📋 Vector Search Index Configuration:")
        print("=" * 50)
        config = """
Create the following Vector Search Index in MongoDB Atlas:

Index Name: embedding-index
Collection: ai_demo.tickets

Configuration JSON:
{
  "name": "embedding-index",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "embedding": {
          "type": "knnVector",
          "dimensions": 1536,
          "similarity": "cosine"
        }
      }
    }
  }
}

Creation Steps:
1. Log into MongoDB Atlas
2. Go to your cluster
3. Click the "Search" tab
4. Click "Create Search Index"
5. Select "JSON Editor"
6. Paste the above configuration
7. Click "Next" and "Create Search Index"

Important Notes:
- Field type must be "knnVector" (not "vector")
- Use "dimensions" not "dims"
- Ensure data exists in collection before creating index
        """
        print(config)
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "=" * 60)
        print("🚀 Atlas Vector Search + DigitalOcean RAG Demo")
        print("=" * 60)
        print("1. Test Endpoint Connection")
        print("2. Data Ingestion (Vectorize Documents)")
        print("3. RAG Query Demo")
        print("4. Re-ranking Demo 🆕")
        print("5. Show Vector Index Configuration")
        print("6. Exit")
        print("-" * 60)
    
    def run(self):
        """Run demonstration"""
        if not self.do_client or not self.client_db:
            print("❌ Initialization failed, please check configuration")
            return
        
        while True:
            self.show_menu()
            choice = input("Select operation (1-6): ").strip()
            
            if choice == "1":
                self.test_endpoint()
            elif choice == "2":
                success = self.ingest_data()
                if success:
                    print("\n💡 Tip: Please ensure you have created a Vector Search Index in MongoDB Atlas")
                    print("   To view configuration, select option 5")
            elif choice == "3":
                print("\n💡 Before starting queries, please ensure:")
                print("   1. Data ingestion completed (option 2)")
                print("   2. Vector Search Index created")
                confirm = input("Confirm to continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.query_demo()
            elif choice == "4":
                print("\n💡 Before re-ranking demo, please ensure:")
                print("   1. Data ingestion completed (option 2)")
                print("   2. Vector Search Index created")
                confirm = input("Confirm to continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.reranking_demo()
            elif choice == "5":
                self.show_vector_index_config()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid selection, please try again")

if __name__ == "__main__":
    demo = RAGDemo()
    demo.run() 