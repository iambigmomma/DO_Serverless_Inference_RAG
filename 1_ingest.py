# 1_ingest.py - Data Ingestion and Vectorization Script
import os
import sys
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configuration settings
DO_GENAI_KEY = os.getenv("DO_GENAI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Added OpenAI API key

if not DO_GENAI_KEY:
    print("❌ Error: DO_GENAI_KEY environment variable not found")
    print("Please add your DigitalOcean Gradient AI Platform API key to the .env file")
    sys.exit(1)

if not MONGODB_URI:
    print("❌ Error: MONGODB_URI environment variable not found")
    print("Please add your MongoDB Atlas connection string to the .env file")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY environment variable not found")
    print("Please add your OpenAI API key to the .env file")
    sys.exit(1)

# Initialize clients
try:
    # DigitalOcean client (for chat)
    do_client = OpenAI(
        api_key=DO_GENAI_KEY,
        base_url="https://inference.do-ai.run/v1"
    )
    
    # OpenAI client (for embeddings)
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
    """Generate embedding vector using OpenAI API"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None

def load_sample_data():
    """Load sample data from JSON files in SAMPLE_DATA directory"""
    sample_data_dir = "SAMPLE_DATA"
    all_tickets = []
    
    if not os.path.exists(sample_data_dir):
        print(f"❌ {sample_data_dir} directory not found!")
        print("Please create the SAMPLE_DATA directory and add JSON files containing ticket data.")
        return []
    
    # Find all JSON files in SAMPLE_DATA directory
    json_files = [f for f in os.listdir(sample_data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ No JSON files found in {sample_data_dir} directory!")
        print("Please add JSON files containing ticket data to the SAMPLE_DATA directory.")
        return []
    
    print(f"📁 Found {len(json_files)} JSON files: {', '.join(json_files)}")
    
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
    
    return all_tickets

def ingest_sample_data():
    """Ingest sample support tickets from JSON files and generate embeddings"""
    
    print("🔄 Starting data ingestion from SAMPLE_DATA directory...")
    
    # Load data from JSON files
    sample_tickets = load_sample_data()
    
    if not sample_tickets:
        print("❌ No valid ticket data found!")
        return
    
    print(f"📊 Total tickets to process: {len(sample_tickets)}")
    
    # Clear existing data
    print("🗑️  Clearing existing data...")
    collection.delete_many({})
    
    # Process each ticket
    successful_inserts = 0
    failed_inserts = 0
    
    for ticket in sample_tickets:
        try:
            # Create searchable text
            searchable_text = f"{ticket.get('title', '')} {ticket.get('description', '')} {ticket.get('category', '')}"
            
            # Generate embedding
            print(f"📝 Processing: {ticket.get('title', 'Unknown Title')}")
            embedding = generate_embedding(searchable_text)
            
            if embedding:
                # Add embedding and searchable text to ticket
                ticket['embedding'] = embedding
                ticket['searchable_text'] = searchable_text
                
                # Insert into MongoDB
                collection.insert_one(ticket)
                print(f"✅ Inserted: {ticket.get('ticket_id', ticket.get('_id', 'Unknown ID'))}")
                successful_inserts += 1
            else:
                print(f"❌ Skipping {ticket.get('ticket_id', 'Unknown')} due to embedding generation failure")
                failed_inserts += 1
                
        except Exception as e:
            print(f"❌ Failed to process ticket {ticket.get('ticket_id', 'Unknown')}: {e}")
            failed_inserts += 1
    
    print(f"\n✅ Data ingestion completed!")
    print(f"📊 Successfully inserted: {successful_inserts} tickets")
    if failed_inserts > 0:
        print(f"⚠️  Failed to insert: {failed_inserts} tickets")
    print(f"📊 Total documents in collection: {collection.count_documents({})}")
    
    # Show category distribution
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    categories = list(collection.aggregate(pipeline))
    
    if categories:
        print(f"\n📈 Category distribution:")
        for cat in categories:
            print(f"  - {cat['_id']}: {cat['count']} tickets")
    
    # Show priority distribution
    pipeline = [
        {"$group": {"_id": "$priority", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    priorities = list(collection.aggregate(pipeline))
    
    if priorities:
        print(f"\n🎯 Priority distribution:")
        for pri in priorities:
            print(f"  - {pri['_id']}: {pri['count']} tickets")

if __name__ == "__main__":
    # Ingest data
    ingest_sample_data()
    
    # Close connections
    mongo_client.close()
    print("🔒 Database connection closed")

    print("\n📋 Next steps:")
    print("1. Create Vector Search Index in MongoDB Atlas")
    print("2. Run 'python 2_query.py' for query testing")
    print("\n💡 Vector Search Index configuration:")
    print("""
{
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
    """) 