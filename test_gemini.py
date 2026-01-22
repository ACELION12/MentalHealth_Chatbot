from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 THIS LINE WAS MISSING

print("KEY:", os.getenv("GEMINI_API_KEY"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Say hello kindly like a therapist."
)

print(response.text)
