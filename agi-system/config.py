import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / "Universal_AI_Analytics" / ".env"
load_dotenv(dotenv_path=str(_env_path), override=True)

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
ZEP_API_KEY       = os.getenv("ZEP_API_KEY", "")
EXA_API_KEY       = os.getenv("EXA_API_KEY", "")

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = "AGI_System"

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

if NVIDIA_API_KEY:
    os.environ["OPENAI_API_KEY"] = NVIDIA_API_KEY

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

OLLAMA_MODEL    = "gpt-oss:120b-cloud"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def query_ollama(messages: list, temperature: float = 0, max_tokens: int = 1024) -> str:
    from openai import OpenAI as _OllamaClient
    client = _OllamaClient(base_url=OLLAMA_BASE_URL, api_key="ollama")
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


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

_BASE = Path(__file__).parent
for _d in ("outputs/charts", "outputs/reports", "outputs/audit"):
    (_BASE / _d).mkdir(parents=True, exist_ok=True)
