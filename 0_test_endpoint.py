from openai import OpenAI
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

try:
    client = OpenAI(
        api_key=os.getenv("DO_GENAI_KEY"),
        base_url="https://inference.do-ai.run/v1"
    )

    resp = client.chat.completions.create(
        model="llama3-8b-instruct",
        messages=[{"role": "user", "content": "ping"}],
        max_completion_tokens=5
    )
    print("✅ DigitalOcean Serverless Inference 連接成功!")
    print(f"回應: {resp.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ 連接失敗: {e}")
    print("請檢查:")
    print("1. DO_GENAI_KEY 環境變數是否正確設置")
    print("2. API Key 是否有效")
    print("3. 網路連接是否正常") 