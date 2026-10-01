# Doc Assistant — Enterprise RAG · V5

A production-ready Retrieval-Augmented Generation (RAG) application built with Streamlit and Azure OpenAI. Upload documents, ask questions, and get accurate answers grounded in your content — with per-user isolation, multi-session chat, semantic caching, and an admin dashboard.

---

## Features

| Category | Details |
|---|---|
| **Authentication** | Login/logout, PBKDF2 password hashing, role-based access (admin / user) |
| **Document Ingestion** | PDF, DOCX, PPTX, XLSX/XLS, URLs — per-user ChromaDB isolation |
| **RAG Pipeline** | Similarity, MMR, Hybrid BM25+Vector, Re-ranking, Parent-Child chunking |
| **Semantic Cache** | Redis L1 (exact match) + SQLite L2 (cosine similarity ≥ 0.75) |
| **Multi-session Chat** | Persistent conversations stored in SQLite, rename/delete sessions |
| **Admin Dashboard** | User management, query logs, document index, cache stats, RAGAS eval |
| **Secrets Management** | Windows Credential Manager via keyring, falls back to `.env` |
| **Themes** | Light / Dark mode toggle (default: light) |
| **Evaluation** | RAGAS-style LLM-as-judge scoring (faithfulness, relevancy, precision, recall) |
| **Tracing** | LangSmith integration for full pipeline observability |

---

## Prerequisites

- Python 3.11+
- Azure OpenAI access (deployment + embedding model)
- Redis (optional — app works without it, SQLite cache is always active)

---

## Installation

```bash
# 1. Clone / download the project
cd simple-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 4. (Optional) Migrate secrets to Windows Credential Manager
python setup_secrets.py

# 5. Run the app
streamlit run app.py
```

---

## Configuration

Edit `.env` with the following keys:

```env
# Azure OpenAI
USE_AZURE=true
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# RAG tuning (optional)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# Redis cache (optional)
REDIS_URL=redis://localhost:6379

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=simple-rag-v5
```

---

## Default Credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin |
| `demo` | `demo123` | User |

> Change passwords via the Admin Dashboard → Users tab after first login.

---

## Project Structure

```
simple-rag/
├── app.py                  # Streamlit entry point — UI, routing, chat logic
├── setup_secrets.py        # One-time: migrate .env secrets → Windows Credential Manager
├── requirements.txt
├── .env                    # Local secrets (gitignored)
├── .env.example            # Template for env vars
├── users.json              # User credentials (auto-created on first run)
│
├── core/                   # All business logic
│   ├── config.py           # Environment variables and path constants
│   ├── secrets_manager.py  # Keyring-based secret storage
│   ├── auth.py             # Authentication — PBKDF2 hashing, user CRUD
│   ├── db.py               # SQLite — conversations, messages, query logs
│   ├── cache.py            # Two-layer semantic cache (Redis + SQLite)
│   ├── ingest.py           # Document ingestion pipeline
│   ├── retriever.py        # RAG retrieval strategies
│   └── ragas_eval.py       # LLM-as-judge evaluation metrics
│
├── .streamlit/
│   └── config.toml         # Streamlit server config
│
├── data/                   # Runtime data (gitignored)
│   └── app.db              # SQLite database
│
└── vector_db/              # ChromaDB vector stores (gitignored)
    └── {username}/         # Per-user isolated collection
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.64+ |
| LLM | Azure OpenAI (gpt-5.6-luna / gpt-4) |
| Embeddings | Azure OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent, per-user) |
| Relational DB | SQLite |
| Cache L1 | Redis (optional, graceful degradation) |
| Cache L2 | SQLite cosine similarity |
| Orchestration | LangChain LCEL |
| Secrets | Windows Credential Manager (keyring) |
| Tracing | LangSmith |

---

## Supported File Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word Document | `.docx` |
| PowerPoint | `.pptx` |
| Excel | `.xlsx`, `.xls` |
| Web URL | `https://...` |

---

## Search Strategies

| Strategy | Description |
|---|---|
| Similarity | Standard cosine similarity search |
| MMR | Maximal Marginal Relevance — diverse, less redundant results |
| Hybrid | BM25 keyword + vector search merged via Reciprocal Rank Fusion |

---

## Admin Dashboard Tabs

| Tab | Purpose |
|---|---|
| Users | Create, view, delete users; see per-user query stats |
| Query Logs | Full query history with token counts and search mode |
| Documents | Browse all indexed documents across all users |
| Cache | Redis L1 + SQLite L2 stats; clear by scope or all |
| Secrets | View/update secrets stored in Windows Credential Manager |
| RAGAS Eval | Run LLM-as-judge evaluation on sample Q&A pairs |
