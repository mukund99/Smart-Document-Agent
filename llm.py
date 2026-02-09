import os
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

def init_llm():
    # It's best practice to use environment variables for API keys
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    Settings.llm = Gemini(
        model="models/gemini-2.5-flash-lite", # Adjusted to latest available stable version
        api_key=api_key,
        transport="rest"
    )
    return Settings.llm