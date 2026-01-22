import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger("aura.llm_client")
logging.basicConfig(level=logging.INFO)

# UPDATED: Using the model confirmed in your screenshot
GEMINI_MODEL = "gemini-3-flash-preview"

class LLMClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY missing")
            self.client = None
            return

        self.client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized (%s)", GEMINI_MODEL)

    def generate_response(self, system_instruction: str, history: List[Dict[str, str]], user_message: str, temperature=0.4) -> str:
        if not self.client:
            return "I’m here with you, but my connection is a bit weak. Please try again in a moment."

        try:
            # 1. Convert internal history format to Gemini format
            gemini_history = []
            for m in history:
                role = "model" if m["role"] == "assistant" else "user"
                gemini_history.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m["content"])]
                ))

            # 2. Create chat session
            chat = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=500,
                    system_instruction=system_instruction
                ),
                history=gemini_history
            )

            # 3. Send ONLY the new user message
            response = chat.send_message(user_message)
            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return "I’m still here with you. I'm having trouble connecting to my thought process right now, but I am listening."

llm_client = LLMClient()