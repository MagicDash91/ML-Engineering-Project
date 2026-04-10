"""
config.py – AI Financial Risk & Compliance Monitor configuration.

LLMs:
  - NVIDIA LLaMA 3.3 Nemotron Super 49B  → risk analytics crew agents
  - Ollama qwen3.5:cloud (LangChain)      → LangGraph planning nodes + chart vision

Data sources (user-supplied at runtime — all optional individually):
  - Database URI  → any SQLAlchemy-compatible DB (PostgreSQL, MySQL, SQLite, etc.)
  - File uploads  → CSV / Excel loaded into per-run SQLite; PDF/Word/PPTX as text context
  Requirement: user question + at least one of (Database URI, uploaded file)

External:
  - Tavily → web research (risk news, regulatory updates, benchmarks)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path), override=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# ── LangSmith Tracing ─────────────────────────────────────────────────────────
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = "Financial_Risk_Compliance_Monitor"

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# LiteLLM (used by CrewAI) resolves openai/* models using OPENAI_API_KEY env var.
if NVIDIA_API_KEY:
    os.environ["OPENAI_API_KEY"] = NVIDIA_API_KEY

# ── NVIDIA LLaMA Nemotron ─────────────────────────────────────────────────────
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

try:
    from openai import OpenAI as _OpenAI
    nvidia_client = _OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=120,
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


# ── Active DB URI (set at runtime per analysis run) ───────────────────────────
# Never set here — always set via os.environ["ACTIVE_DB_URI"] in app.py
# after resolving the user-supplied database_uri or uploaded structured files.

def get_active_db_uri() -> str:
    """Return the database URI for the current analysis run."""
    uri = os.environ.get("ACTIVE_DB_URI", "")
    if not uri:
        raise RuntimeError(
            "No active database URI. Provide a Database URI or upload a structured file."
        )
    return uri


# ── Output directories ────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
for _d in ("outputs/charts", "outputs/reports", "outputs/models",
           "outputs/audit", "outputs/uploads"):
    (_BASE / _d).mkdir(parents=True, exist_ok=True)
