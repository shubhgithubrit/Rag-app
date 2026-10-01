import os
from dotenv import load_dotenv

load_dotenv()

# Sensitive secrets: keyring first, .env as fallback
from .secrets_manager import get_secret

# OpenAI
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

# Azure OpenAI
USE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"
AZURE_OPENAI_API_KEY               = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT              = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT            = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION           = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# RAG parameters
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K   = int(os.getenv("RETRIEVAL_K", "4"))

# V4 — Advanced Retrieval
FETCH_K           = int(os.getenv("FETCH_K", "8"))
RERANK_TOP_N      = int(os.getenv("RERANK_TOP_N", "4"))
MMR_LAMBDA        = float(os.getenv("MMR_LAMBDA", "0.5"))
CHILD_CHUNK_SIZE  = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))

# Paths — root is one level up from this file (core/)
_ROOT             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.path.join(_ROOT, "vector_db")
DATA_DIR          = os.path.join(_ROOT, "data")

# V5 — Production
DB_PATH = os.path.join(DATA_DIR, "app.db")

# Redis cache (optional — app works without Redis)
REDIS_URL       = get_secret("REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379"))
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", str(7 * 24 * 3600)))

# LangSmith tracing
_langchain_key = get_secret("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_ENDPOINT",   os.getenv("LANGCHAIN_ENDPOINT",   "https://api.smith.langchain.com"))
os.environ.setdefault("LANGCHAIN_API_KEY",    _langchain_key)
os.environ.setdefault("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT",    "simple-rag-v5"))
