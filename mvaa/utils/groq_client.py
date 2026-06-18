import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client
