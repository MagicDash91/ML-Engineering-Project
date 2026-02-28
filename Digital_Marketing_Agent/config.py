"""
config.py – LLM & API configuration for Digital Marketing Agent.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from the Digital_Marketing directory ──────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# ── LangSmith tracing (optional) ────────────────────────────────────────────
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = "Digital_Marketing_Agent"

# ── Gemini 2.5 Flash via LangChain ──────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
        timeout=30,       # 30s per attempt (was 120s) → retries stay fast
        max_retries=3,    # up to 4 total attempts × 30s = 120s max
        max_tokens=2048,  # cap output to prevent unbounded generation
    )
except Exception as _e:
    print(f"[config] WARNING: could not initialise Gemini LLM – {_e}")
    gemini_llm = None

# ── google-genai client (for Veo 3 video generation) ────────────────────────
try:
    from google import genai as _genai
    genai_client = _genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
except Exception as _e:
    print(f"[config] WARNING: could not initialise google-genai client – {_e}")
    genai_client = None
