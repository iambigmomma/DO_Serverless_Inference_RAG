# setup.py - Quick Setup Script
import os
import subprocess
import sys

def install_dependencies():
    """Install dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        return False

def create_env_file():
    """Create environment variables file"""
    env_content = """# DigitalOcean Gradient AI Platform API Key
DO_GENAI_KEY=please_enter_your_digitalocean_gradient_ai_api_key

# OpenAI API Key (for embedding generation)
OPENAI_API_KEY=please_enter_your_openai_api_key

# MongoDB Atlas Connection String
MONGODB_URI=please_enter_your_mongodb_atlas_connection_string
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ Created .env environment variables file")
    else:
        print("ℹ️  .env file already exists")

def show_next_steps():
    """Show next steps"""
    print("\n" + "=" * 60)
    print("🎉 Initial setup completed!")
    print("=" * 60)
    print("📋 Please follow these steps next:")
    print()
    print("1. 📝 Edit the .env file and fill in your API credentials:")
    print("   - DO_GENAI_KEY: DigitalOcean Gradient AI Platform API Key")
    print("   - OPENAI_API_KEY: OpenAI API Key")
    print("   - MONGODB_URI: MongoDB Atlas Connection String")
    print()
    print("2. 🗄️  Set up in MongoDB Atlas:")
    print("   - Create database: ai_demo")
    print("   - Create collection: tickets")
    print("   - Create Vector Search Index (run demo.py to see detailed configuration)")
    print()
    print("3. 🚀 Run complete demo:")
    print("   python demo.py")
    print()
    print("4. 🔧 Or run step by step:")
    print("   python 0_test_endpoint.py  # Test API connections")
    print("   python 1_ingest.py         # Data ingestion and vectorization")
    print("   python 2_query.py          # RAG query testing")
    print("   python 3_change_streams.py # Real-time data monitoring")
    print()
    print("💡 Tips: If you encounter connection issues, please check:")
    print("   - API keys are correct")
    print("   - Network connection is working")
    print("   - MongoDB Atlas whitelist settings")
    print("=" * 60)

def main():
    print("🚀 Atlas Vector Search + DigitalOcean RAG Demo System Setup")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed, please check network connection or requirements.txt file")
        return
    
    # Create environment variables file
    create_env_file()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 