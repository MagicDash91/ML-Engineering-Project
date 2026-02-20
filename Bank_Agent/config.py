"""
Banking Analytics - Central Configuration
Initializes all LLMs, API clients, and shared settings.

LLMs:
  - NVIDIA Llama 3.3 Nemotron Super 49B  -> main reasoning engine (Data Engineer, Data Scientist)
  - Gemini 2.5 Flash                      -> vision + business insights (Data Analyst)

Data Warehouse:
  - PostgreSQL (add credentials to .env)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same directory as this file.
# override=True ensures Bank_Agent/.env always wins over any previously
# loaded env vars (e.g. from D:\Langsmith-main\.env loaded by another module).
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

# ===========================
# API Keys
# ===========================
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ===========================
# LangSmith Tracing (optional)
# ===========================
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = "Bank_Agent"

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ===========================
# NVIDIA Llama Nemotron
# (via OpenAI-compatible REST API)
# ===========================
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    timeout=180,
    max_retries=10,
) if NVIDIA_API_KEY else None


def query_nvidia(messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Direct NVIDIA Llama Nemotron call (mirrors the pattern in Super_Agentic_RAG)."""
    if not nvidia_client:
        raise ValueError("NVIDIA_API_KEY not set. Check your .env file.")
    response = nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ===========================
# Gemini 2.5 Flash
# (for vision & business insights)
# ===========================
GEMINI_MODEL = "gemini-2.5-flash"

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    gemini_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
        timeout=180,
        max_retries=10,
    ) if GOOGLE_API_KEY else None
except Exception:
    gemini_llm = None

# ===========================
# PostgreSQL Data Warehouse
# Add these to .env:
#   POSTGRES_HOST, POSTGRES_PORT,
#   POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
# ===========================
POSTGRES_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "localhost"),
    "port":     os.getenv("POSTGRES_PORT",     "5432"),
    "database": os.getenv("POSTGRES_DB",       "bank_analytics"),
    "user":     os.getenv("POSTGRES_USER",     "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

POSTGRES_URI = (
    f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
)

# ===========================
# Output directories
# ===========================
OUTPUT_DIRS = [
    "outputs/charts",
    "outputs/reports",
    "outputs/models",
]

for _d in OUTPUT_DIRS:
    os.makedirs(_d, exist_ok=True)
