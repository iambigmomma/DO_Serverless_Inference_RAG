# 2_query.py - RAG Query Testing Script
import os
import sys
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DO_GENAI_KEY = os.getenv("DO_GENAI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DO_GENAI_KEY:
    print("❌ Error: DO_GENAI_KEY not found in environment variables")
    sys.exit(1)

if not MONGODB_URI:
    print("❌ Error: MONGODB_URI not found in environment variables")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)

# Initialize clients
try:
    # DigitalOcean client for chat
    do_client = OpenAI(
        api_key=DO_GENAI_KEY,
        base_url="https://inference.do-ai.run/v1"
    )
    
    # OpenAI client for embeddings
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    # MongoDB client
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client.ai_demo
    collection = db.tickets
    
    print("✅ All services connected successfully!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

def generate_embedding(text):
    """Generate embedding using OpenAI API"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None

def vector_search(query_text, limit=3):
    """Perform vector search using MongoDB Atlas Vector Search"""
    
    # Generate embedding for query
    query_embedding = generate_embedding(query_text)
    if not query_embedding:
        return []
    
    # Vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embedding-index",
                "path": "embedding", 
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": limit
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
    
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return []

def generate_rag_response(query, context_docs):
    """Generate RAG response using retrieved context"""
    
    # Build context from retrieved documents
    context = "\n\n".join([
        f"Ticket {doc['ticket_id']}: {doc['title']}\n"
        f"Description: {doc['description']}\n"
        f"Category: {doc['category']}, Priority: {doc['priority']}, Status: {doc['status']}"
        for doc in context_docs
    ])
    
    # Create prompt
    prompt = f"""You are a helpful customer support assistant. Based on the following support ticket information, provide a helpful response to the user's query.

Context (Support Tickets):
{context}

User Query: {query}

Please provide a helpful response based on the relevant tickets above. If the query is about a specific issue, reference similar tickets and suggest solutions."""

    try:
        response = do_client.chat.completions.create(
            model="llama3-8b-instruct",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=500,
            stream=True
        )
        
        print("🤖 AI Response:")
        print("-" * 50)
        
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print("\n" + "-" * 50)
        return full_response
        
    except Exception as e:
        print(f"❌ RAG response generation failed: {e}")
        return None

def show_sample_questions():
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

def interactive_query():
    """Interactive query interface"""
    print("\n🔍 RAG Query System")
    print("=" * 50)
    print("Enter your queries (type 'quit' to exit)")
    
    show_sample_questions()
    
    while True:
        query = input("\n💬 Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        print(f"\n🔄 Searching for: '{query}'")
        
        # Perform vector search
        results = vector_search(query, limit=3)
        
        if not results:
            print("❌ No relevant documents found")
            continue
            
        # Display search results
        print(f"\n📋 Found {len(results)} relevant tickets:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Ticket {doc['ticket_id']}: {doc['title']}")
            print(f"   Category: {doc['category']} | Priority: {doc['priority']} | Status: {doc['status']}")
            print(f"   Similarity Score: {doc['score']:.4f}")
            print(f"   Description: {doc['description'][:100]}...")
        
        # Generate RAG response
        print(f"\n🤖 Generating response...")
        rag_response = generate_rag_response(query, results)
        
        # Show sample questions again after each query
        show_sample_questions()

if __name__ == "__main__":
    # Check if data exists
    doc_count = collection.count_documents({})
    if doc_count == 0:
        print("❌ No documents found in database")
        print("Please run 'python 1_ingest.py' first to ingest data")
        sys.exit(1)
    
    print(f"📊 Found {doc_count} documents in database")
    
    # Start interactive query
    interactive_query()
    
    # Close connections
    mongo_client.close()
    print("\n🔒 Database connection closed") 