# 0_test_endpoint.py - DigitalOcean Gradient AI Platform Connection Test
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_digitalocean_connection():
    """Test DigitalOcean Gradient AI Platform connection"""
    print("🧪 Testing DigitalOcean Gradient AI Platform connection...")
    
    # Check API key
    api_key = os.getenv("DO_GENAI_KEY")
    if not api_key:
        print("❌ Error: DO_GENAI_KEY environment variable not found")
        print("Please set your DigitalOcean Gradient AI Platform API key in the .env file")
        return False
    
    if api_key.startswith("Please enter") or api_key == "your_digitalocean_gradient_ai_api_key_here":
        print("❌ Error: Please enter a real API key in the .env file")
        print("The current value appears to be a placeholder, please replace with your actual API key")
        return False
    
    try:
        # Create client
        client = OpenAI(
            api_key=api_key,
            base_url="https://inference.do-ai.run/v1"
        )
        
        # Send test request (Note: DigitalOcean requires max_completion_tokens to be at least 256)
        print("📡 Sending test request...")
        resp = client.chat.completions.create(
            model="llama3-8b-instruct",
            messages=[{"role": "user", "content": "Please reply 'pong' briefly to confirm connection is working"}],
            max_completion_tokens=256,  # DigitalOcean requires minimum value of 256
            temperature=0.1
        )
        
        print("✅ DigitalOcean Gradient AI Platform connection successful!")
        print(f"🤖 AI Response: {resp.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔍 Troubleshooting suggestions:")
        print("1. Check if DO_GENAI_KEY environment variable is correctly set")
        print("2. Verify that the API key is valid and not expired")
        print("3. Check if network connection is working")
        print("4. Confirm your DigitalOcean account has Gradient AI Platform access permissions")
        
        # Provide more detailed error information
        if "401" in str(e):
            print("💡 Tip: 401 error usually indicates invalid API key")
        elif "403" in str(e):
            print("💡 Tip: 403 error usually indicates no access permissions")
        elif "timeout" in str(e).lower():
            print("💡 Tip: Connection timeout, please check network connection")
        elif "connection" in str(e).lower():
            print("💡 Tip: Network connection issue, please check firewall settings")
        elif "max_completion_tokens" in str(e):
            print("💡 Tip: DigitalOcean requires max_completion_tokens to be at least 256")
        
        return False

if __name__ == "__main__":
    print("🚀 DigitalOcean Gradient AI Platform Connection Test")
    print("=" * 50)
    
    success = test_digitalocean_connection()
    
    if success:
        print("\n✅ Test completed! Your API connection is working properly")
        print("💡 You can now run other scripts for complete testing")
    else:
        print("\n❌ Test failed! Please check the suggestions above and retry")
        sys.exit(1) 