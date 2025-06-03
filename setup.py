# setup.py - 快速設置腳本
import os
import subprocess
import sys

def install_dependencies():
    """安裝依賴包"""
    print("📦 安裝 Python 依賴包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依賴包安裝成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依賴包安裝失敗: {e}")
        return False

def create_env_file():
    """Create .env file with environment variables"""
    env_content = """# DigitalOcean GenAI Platform API Key
DO_GENAI_KEY=your_digitalocean_genai_api_key_here

# OpenAI API Key (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB Atlas Connection String
MONGODB_URI=your_mongodb_atlas_connection_string_here
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")

def show_next_steps():
    """顯示後續步驟"""
    print("\n" + "=" * 60)
    print("🎉 設置完成!")
    print("=" * 60)
    print("📋 後續步驟:")
    print("1. 編輯 .env 文件，填入您的憑證:")
    print("   - DO_GENAI_KEY: DigitalOcean GenAI Platform API Key")
    print("   - MONGODB_URI: MongoDB Atlas 連接字串")
    print()
    print("2. 在 MongoDB Atlas 中:")
    print("   - 創建資料庫: ai_demo")
    print("   - 創建集合: tickets")
    print("   - 創建 Vector Search Index (運行 demo.py 查看配置)")
    print()
    print("3. 運行演示:")
    print("   python demo.py")
    print()
    print("4. 或分步運行:")
    print("   python 0_test_endpoint.py  # 測試連接")
    print("   python 1_ingest.py         # 數據攝取")
    print("   python 2_query.py          # RAG 查詢")
    print("   python 3_change_streams.py # 實時監聽")
    print("=" * 60)

def main():
    print("🚀 Atlas Vector Search + DigitalOcean RAG Demo 設置")
    print("=" * 60)
    
    # 安裝依賴
    if not install_dependencies():
        return
    
    # 創建環境變數文件
    if not create_env_file():
        return
    
    # 顯示後續步驟
    show_next_steps()

if __name__ == "__main__":
    main() 