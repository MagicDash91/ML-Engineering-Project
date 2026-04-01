"""
config.py – Combined Banking Analytics + Digital Marketing Agent configuration.

LLMs:
  - NVIDIA LLaMA 3.3 Nemotron Super 49B  → banking analytics (Engineer, Scientist, Analyst)
  - Gemini 2.5 Flash (LangChain)         → chart vision, marketing planning, content tools
  - Gemini 2.5 Flash (LiteLLM/CrewAI)   → marketing CrewAI agents

Data:
  - PostgreSQL  → banking churn warehouse
  - Tavily      → web research (banking + marketing)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from same directory as this file (always wins over parent env)
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path), override=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# ── LangSmith Tracing ────────────────────────────────────────────────────────
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = "Bank_Analytics_Digital_Marketing"

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# LiteLLM (used by CrewAI) resolves openai/* models using OPENAI_API_KEY from
# the environment, even when api_key is explicitly passed to LLM().
# Setting it here ensures the NVIDIA endpoint auth never fails with
# "The api_key client option must be set ... OPENAI_API_KEY".
if NVIDIA_API_KEY:
    os.environ["OPENAI_API_KEY"] = NVIDIA_API_KEY

# ── NVIDIA LLaMA Nemotron (banking analytics) ────────────────────────────────
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

try:
    from openai import OpenAI as _OpenAI
    nvidia_client = _OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=120,   # 49B model needs up to 2 min on NIM under load
        max_retries=0,
    ) if NVIDIA_API_KEY else None
except Exception:
    nvidia_client = None


def query_nvidia(messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Call NVIDIA LLaMA Nemotron via OpenAI-compatible endpoint."""
    if not nvidia_client:
        raise ValueError("NVIDIA_API_KEY not set. Check .env file.")
    response = nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# ── Gemini 2.5 Flash (LangChain – used in tool helpers & graph nodes) ────────
GEMINI_MODEL = "gemini-2.5-flash"

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    gemini_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY,
        timeout=30,
        max_retries=3,
    ) if GOOGLE_API_KEY else None
except Exception:
    gemini_llm = None

# ── PostgreSQL Data Warehouse ────────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST",     "localhost"),
    "port":     os.getenv("POSTGRES_PORT",     "5432"),
    "database": os.getenv("POSTGRES_DB",       "churn"),
    "user":     os.getenv("POSTGRES_USER",     "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

POSTGRES_URI = (
    f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
)

# ── Output directories ───────────────────────────────────────────────────────
_BASE = Path(__file__).parent
for _d in ("outputs/charts", "outputs/reports", "outputs/models",
           "outputs/videos", "outputs/content", "outputs/audit"):
    (_BASE / _d).mkdir(parents=True, exist_ok=True)
