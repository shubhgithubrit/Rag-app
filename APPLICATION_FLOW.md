# Application Flow — Doc Assistant Enterprise RAG V5

This document explains the complete technical flow of the application from user interaction to response.

---

## 1. Startup & Bootstrap

When `streamlit run app.py` is executed:

```
app.py (top-level)
  ├── core.auth.init_default_users()     → creates users.json with admin + demo if missing
  ├── core.db.init_db()                  → creates SQLite tables (conversations, messages, query_logs)
  └── core.cache.init_cache_table()      → creates query_cache table + index in app.db
```

Streamlit then renders the login page and waits for input.

---

## 2. Authentication Flow

```
User enters username + password
  │
  ▼
core.auth.verify_login(username, password)
  ├── Loads users.json
  ├── PBKDF2-HMAC-SHA256 hash comparison (salt:hash format)
  └── Returns True / False

If True:
  ├── st.session_state["username"] = username
  ├── st.session_state["is_admin"] = core.auth.is_admin(username)
  └── st.rerun() → renders main app

If False:
  └── Shows "Invalid credentials" error
```

Secrets are loaded via `core.secrets_manager.get_secret()`:
- Checks **Windows Credential Manager** first (keyring)
- Falls back to **environment variable / .env**

---

## 3. Session Initialization

On first login, `init_user_session(username)` runs:

```
  ├── Create a new conversation in SQLite → returns conversation_id
  ├── st.session_state["current_conv"] = conversation_id
  ├── st.session_state["messages"] = []         ← chat display buffer
  ├── st.session_state["lc_history"] = []       ← LangChain message history
  ├── st.session_state["tokens_in/out"] = 0
  └── st.session_state["theme"] = "light"       ← default theme
```

---

## 4. Document Ingestion Flow

```
User uploads file(s) or pastes a URL in sidebar
  │
  ▼
Sidebar "Process Documents" button clicked
  │
  ├── File uploaded → saved to temp path
  │
  ▼
core.ingest.ingest_file(path, user_id)   OR   ingest_file_parent_child(...)
  │
  ├── Load document based on extension:
  │     .pdf  → PyPDFLoader
  │     .docx → python-docx
  │     .pptx → python-pptx
  │     .xlsx → pandas ExcelFile
  │     URL   → requests + BeautifulSoup
  │
  ├── Split into chunks:
  │     RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  │     Parent-Child mode: large parent chunks + small child chunks
  │
  ├── Generate embeddings:
  │     AzureOpenAIEmbeddings (text-embedding-3-small)
  │
  └── Store in ChromaDB:
        persist_directory = vector_db/{username}/
        collection_name   = {username}_docs
```

---

## 5. Chat & RAG Query Flow

This is the core pipeline triggered on every user message:

```
User types a question → st.chat_input fires
  │
  ▼
┌─────────────────────────────────────────────────┐
│              SEMANTIC CACHE CHECK               │
│                                                 │
│  1. Compute question embedding                  │
│     AzureOpenAIEmbeddings.embed_query(question) │
│                                                 │
│  2. Redis L1 lookup (if Redis running)          │
│     Key = sha256(normalized_question)           │
│     → HIT: return answer instantly (<1ms)       │
│                                                 │
│  3. SQLite L2 lookup (always)                   │
│     SELECT all entries for this scope           │
│     Compute cosine similarity vs each embedding │
│     → HIT if best_sim ≥ 0.75                    │
│       return cached answer + show similarity %  │
└─────────────────────────────────────────────────┘
  │ MISS
  ▼
┌─────────────────────────────────────────────────┐
│              RETRIEVAL                          │
│                                                 │
│  Strategy set by user in sidebar:               │
│                                                 │
│  Similarity → vectorstore.similarity_search()   │
│  MMR        → max_marginal_relevance_search()   │
│  Hybrid     → BM25 + vector → RRF merge         │
│                                                 │
│  Optional post-retrieval:                       │
│  Re-rank    → LLM picks top N most relevant     │
│  Parent-Child → swap child chunks for parents   │
│                                                 │
│  Admin can query:                               │
│  - Own documents                                │
│  - Specific user's documents                    │
│  - All users' documents (merged retrieval)      │
└─────────────────────────────────────────────────┘
  │ docs[]
  ▼
┌─────────────────────────────────────────────────┐
│              GENERATION (LLM)                   │
│                                                 │
│  1. Contextualize question                      │
│     If conversation history exists:             │
│     LLM rewrites question to be self-contained  │
│                                                 │
│  2. QA chain (LCEL)                             │
│     System: "Answer using only context below"   │
│     Context: retrieved chunks joined            │
│     Question: (contextualized) user question    │
│                                                 │
│  3. Stream response → st.write_stream()         │
│     Model: gpt-5.6-luna / gpt-4 (Azure)        │
│     (No temperature param for gpt-5.6-luna)     │
└─────────────────────────────────────────────────┘
  │ full_answer, sources[]
  ▼
┌─────────────────────────────────────────────────┐
│              POST-RESPONSE                      │
│                                                 │
│  1. cache.store(question, embedding,            │
│                 answer, sources, scope)         │
│     → Write to Redis L1 (if available)          │
│     → Write to SQLite L2 (always)               │
│                                                 │
│  2. db.save_message(conv_id, role, content)     │
│     → Persist to SQLite messages table          │
│                                                 │
│  3. db.log_query(user, tokens_in, tokens_out)   │
│     → Persist to query_logs table               │
│                                                 │
│  4. Update st.session_state                     │
│     → lc_history, messages, token counts        │
│                                                 │
│  5. st.rerun() → re-render chat with new msg    │
└─────────────────────────────────────────────────┘
```

---

## 6. Cache Scope

Cache entries are scoped to the **retrieval context** so answers from different document sets are never mixed:

| User Type | Query Context | Cache Scope |
|---|---|---|
| Regular user | Own documents | `username` |
| Admin | My Documents | `"admin"` |
| Admin | User: alice | `"alice"` |
| Admin | All Users | `"all"` |

---

## 7. Multi-Session Conversation Flow

```
Sidebar "Conversations" section
  │
  ├── "+ New Chat" → db.create_conversation(user_id)
  │                  → new conversation_id, empty messages
  │
  ├── Click existing session → load messages from SQLite
  │     db.get_messages(conv_id)
  │     Rebuild lc_history from stored messages
  │
  └── "×" Delete → db.delete_conversation(conv_id)
                    → removes conversation + all messages

Auto-naming: first message in session → LLM or truncated text used as name
```

---

## 8. Admin Dashboard Flow

```
Admin clicks "⚙️ Admin" in sidebar → show_admin()
  │
  ├── Tab: 👥 Users
  │     List all users + stats (queries, tokens, conversations)
  │     Create new user / Delete user
  │
  ├── Tab: 📋 Query Logs
  │     Recent 100 queries with token counts, search mode, timestamps
  │     Charts: queries over time, tokens by user
  │
  ├── Tab: 📁 Documents
  │     All indexed files across all users
  │     Delete individual documents from vector store
  │
  ├── Tab: ⚡ Cache
  │     Redis L1: status, entry count, hit count, URL
  │     SQLite L2: total entries, hits, breakdown by scope
  │     Clear cache (all scopes or specific user)
  │
  ├── Tab: 🔑 Secrets
  │     Status of each managed key (keyring / env / missing)
  │     Update secret → stored in Windows Credential Manager
  │     Delete secret from keyring
  │
  └── Tab: 📊 RAGAS Eval
        Input: questions + ground truths (one per line)
        Build samples: retrieve context + generate answers
        Score: 4 metrics via direct LLM prompts
        Output: DataFrame with faithfulness, relevancy, precision, recall
```

---

## 9. Data Storage Architecture

```
data/app.db  (SQLite)
  ├── conversations   (id, user_id, name, created_at, updated_at)
  ├── messages        (id, conversation_id, role, content, sources, created_at)
  ├── query_logs      (id, user_id, conversation_id, question, tokens_in, tokens_out, search_mode, created_at)
  └── query_cache     (id, scope, question, embedding, answer, sources, hit_count, created_at, last_hit_at)

users.json  (project root)
  └── {username: {password_hash, role, email}}

vector_db/  (ChromaDB)
  └── {username}/
        ├── chroma.sqlite3     ← ChromaDB internal index
        └── parents.pkl        ← parent chunks for parent-child retrieval

Redis  (optional, external)
  └── rag:cache:{scope}:{sha256(question)}  → {answer, sources, question}
  └── rag:cache:{scope}:{sha256(question)}:hits  → hit counter
```

---

## 10. Secret Resolution Order

```
get_secret("AZURE_OPENAI_API_KEY")
  │
  ├── 1. Windows Credential Manager (keyring)
  │      Service: "simple-rag-v5"
  │      Username: "AZURE_OPENAI_API_KEY"
  │
  ├── 2. Environment variable
  │      os.getenv("AZURE_OPENAI_API_KEY")
  │
  └── 3. Fallback default (if provided)
```

Run `python setup_secrets.py` once to migrate `.env` values into the Credential Manager.

---

## 11. Theme System

```
st.session_state["theme"]  →  "light" (default) | "dark"

inject_theme_css()
  └── Reads theme from session state
  └── Injects CSS variables via st.markdown(unsafe_allow_html=True)
      Covers: app background, sidebar, inputs, chat messages,
              buttons (primary/secondary), chat input box,
              tabs, expanders, metrics, file uploader
```

Toggle button rendered at sidebar bottom next to Logout.
