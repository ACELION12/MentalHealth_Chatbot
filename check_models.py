# check_models.py
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ API Key not found in .env")
    exit()

client = genai.Client(api_key=api_key)

print("Fetching available models...")
try:
    # We will just print the name directly to avoid attribute errors
    for model in client.models.list():
        # The model object usually has a .name or .display_name attribute
        # We print the raw object representation if .name isn't found, just to be safe
        print(f"✅ Found: {getattr(model, 'name', model)}")
except Exception as e:
    print(f"❌ Error: {e}")