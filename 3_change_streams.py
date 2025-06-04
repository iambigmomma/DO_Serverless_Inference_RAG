# 3_change_streams.py - Real-time Data Monitoring with Change Streams
import os
import time
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
    print("✅ MongoDB Atlas connection successful!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

def embed(text: str) -> list[float]:
    """Convert text to vector embedding"""
    try:
        resp = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None

def search_and_answer(question: str, doc_id: str):
    """Search relevant documents and generate answer"""
    print(f"🔍 Processing question: {question}")
    
    # Generate vector embedding for the question
    query_embedding = embed(question)
    if query_embedding is None:
        return "Sorry, unable to process your question."
    
    # Search relevant documents
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding-index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "k": 3
                }
            },
            {"$project": {"text": 1, "_id": 0}}
        ]
        
        results = list(col.aggregate(pipeline))
        contexts = [doc["text"] for doc in results]
        
        if not contexts:
            return "Sorry, no relevant information found."
        
        print(f"📚 Found {len(contexts)} relevant documents")
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return "Error occurred during search."
    
    # Generate answer
    context_text = "\n".join([f"- {ctx}" for ctx in contexts])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided context to answer the user's question concisely."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context_text}\n\nQuestion: {question}"
        }
    ]
    
    try:
        response = client_ai.chat.completions.create(
            model="llama3-8b-instruct",
            messages=messages,
            max_completion_tokens=200,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        print(f"💬 Generated answer: {answer[:100]}...")
        
        # Update document with answer
        col.update_one(
            {"_id": doc_id},
            {"$set": {"answer": answer, "answered_at": time.time()}}
        )
        
        return answer
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Error occurred during answer generation."

def watch_changes():
    """Monitor collection changes"""
    print("👀 Starting to monitor changes in tickets collection...")
    print("💡 Tip: Insert new documents with 'question' field to trigger automatic answers")
    print("=" * 60)
    
    # Monitor insert operations
    pipeline = [
        {"$match": {"operationType": "insert"}},
        {"$match": {"fullDocument.question": {"$exists": True}}}
    ]
    
    try:
        with col.watch(pipeline) as stream:
            for change in stream:
                doc = change['fullDocument']
                doc_id = doc['_id']
                question = doc.get('question', '')
                
                print(f"\n🆕 New question detected (ID: {doc_id})")
                print(f"❓ Question: {question}")
                print("🔄 Generating answer...")
                
                # Generate and save answer
                answer = search_and_answer(question, doc_id)
                
                print(f"✅ Answer saved to document {doc_id}")
                print("=" * 60)
                
    except KeyboardInterrupt:
        print("\n👋 Stopped monitoring")
    except Exception as e:
        print(f"❌ Error occurred during monitoring: {e}")

def insert_sample_question():
    """Insert sample questions for testing"""
    sample_questions = [
        "What login issues do we have?",
        "What payment problems exist?",
        "What are the high priority tickets?",
    ]
    
    print("🧪 Inserting test questions...")
    for i, question in enumerate(sample_questions, 1):
        doc = {
            "_id": f"test_q_{int(time.time())}_{i}",
            "question": question,
            "created_at": time.time(),
            "type": "test_question"
        }
        
        try:
            col.insert_one(doc)
            print(f"✅ Inserted question {i}: {question}")
            time.sleep(2)  # Wait for processing
        except Exception as e:
            print(f"❌ Insertion failed: {e}")

if __name__ == "__main__":
    print("🚀 Change Streams RAG System")
    print("=" * 50)
    
    choice = input("Choose operation:\n1. Start monitoring (enter 1)\n2. Insert test questions (enter 2)\nPlease select: ").strip()
    
    if choice == "1":
        watch_changes()
    elif choice == "2":
        insert_sample_question()
        print("\n💡 Now you can run monitoring mode to see automatic answer effects")
    else:
        print("❌ Invalid selection") 