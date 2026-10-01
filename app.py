"""
RAG Chat — Version 5 (Production Ready)
  • User authentication with per-user document isolation
  • Persistent multi-session conversations (SQLite)
  • Admin dashboard — users, query logs, document index
Run: streamlit run app.py
"""
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core import auth, cache, db
from core.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    DATA_DIR,
    FETCH_K,
    MMR_LAMBDA,
    RERANK_TOP_N,
    RETRIEVAL_K,
    USE_AZURE,
)
from core.ingest import (
    clear_vectorstore,
    collection_count,
    delete_document,
    get_all_user_ids,
    get_indexed_files,
    get_vectorstore,
    ingest_file,
    ingest_file_parent_child,
    ingest_url,
    load_parents,
)
from core.retriever import expand_to_parents, hybrid_retrieve, mmr_retrieve, rerank, similarity_retrieve

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Doc Assistant",
    page_icon="🗂️",
    layout="wide",
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
auth.init_default_users()
db.init_db()
cache.init_cache_table()

# ── Theme CSS ─────────────────────────────────────────────────────────────────
_DARK = {
    "bg":           "#0f172a",
    "bg2":          "#1e293b",
    "sidebar_bg":   "#0f172a",
    "text":         "#e2e8f0",
    "text_muted":   "#64748b",
    "border":       "rgba(255,255,255,0.08)",
    "accent":       "#6366f1",
    "accent_soft":  "rgba(99,102,241,0.15)",
    "input_bg":     "#1e293b",
    "msg_user":     "#1e293b",
    "msg_ai":       "#0f172a",
    "card":         "rgba(255,255,255,0.04)",
    "toggle_icon":  "☀️",
    "toggle_label": "Light mode",
}
_LIGHT = {
    "bg":           "#f8fafc",
    "bg2":          "#ffffff",
    "sidebar_bg":   "#f1f5f9",
    "text":         "#1e293b",
    "text_muted":   "#64748b",
    "border":       "rgba(0,0,0,0.08)",
    "accent":       "#6366f1",
    "accent_soft":  "rgba(99,102,241,0.12)",
    "input_bg":     "#ffffff",
    "msg_user":     "#ede9fe",
    "msg_ai":       "#f8fafc",
    "card":         "rgba(0,0,0,0.03)",
    "toggle_icon":  "🌙",
    "toggle_label": "Dark mode",
}

def inject_theme_css():
    t = _DARK if st.session_state.get("theme", "light") == "dark" else _LIGHT
    st.markdown(f"""
    <style>
    /* ── App shell ── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {t["bg"]} !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stHeader"] {{
        background-color: {t["bg"]} !important;
        border-bottom: 1px solid {t["border"]};
    }}
    /* ── Sidebar ── */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: {t["sidebar_bg"]} !important;
        border-right: 1px solid {t["border"]};
    }}
    /* ── Main text ── */
    p, span, label, div, h1, h2, h3, h4, li {{
        color: {t["text"]} !important;
    }}
    /* ── Inputs ── */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {t["input_bg"]} !important;
        color: {t["text"]} !important;
        border-color: {t["border"]} !important;
    }}
    /* ── Chat messages ── */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background-color: {t["msg_user"]} !important;
        border-radius: 12px;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background-color: {t["msg_ai"]} !important;
        border-radius: 12px;
    }}
    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background-color: {t["card"]} !important;
        border: 1px solid {t["border"]};
        border-radius: 10px; padding: 12px;
    }}
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        background-color: {t["bg2"]} !important;
    }}
    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background-color: {t["card"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 8px;
    }}
    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background-color: {t["bg2"]} !important;
        border-radius: 8px; padding: 4px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {t["text_muted"]} !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background-color: {t["accent_soft"]} !important;
        color: {t["accent"]} !important;
        border-radius: 6px;
    }}
    /* ── Divider ── */
    hr {{ border-color: {t["border"]} !important; }}
    /* ── Code blocks ── */
    code {{ background-color: {t["card"]} !important; color: {t["accent"]} !important; }}
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        background-color: {t["card"]} !important;
        border: 1px dashed {t["border"]} !important;
        border-radius: 8px;
    }}
    /* ── Secondary buttons (dark themed) ── */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {{
        background-color: {t["card"]} !important;
        color: {t["text"]} !important;
        border: 1px solid {t["border"]} !important;
    }}
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {{
        background-color: {t["accent_soft"]} !important;
        color: {t["accent"]} !important;
        border-color: {t["accent"]} !important;
    }}
    /* ── Chat input box ── */
    [data-testid="stChatInput"], [data-testid="stChatInputContainer"],
    .stChatFloatingInputContainer, [data-testid="stBottom"],
    [data-testid="stBottom"] > div {{
        background-color: {t["bg"]} !important;
    }}
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] > div > textarea {{
        background-color: {t["input_bg"]} !important;
        color: {t["text"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {t["text_muted"]} !important;
    }}
    [data-testid="stChatInput"] button {{
        background-color: {t["accent"]} !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── LLM ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    if USE_AZURE:
        return AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return ChatOpenAI(model="gpt-4o-mini")


@st.cache_resource
def _get_embed_model():
    from core.ingest import get_embeddings
    return get_embeddings()


def _embed_question(text: str) -> list:
    return _get_embed_model().embed_query(text)


# ── Prompts ───────────────────────────────────────────────────────────────────
_CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a conversation history and a follow-up question, rewrite the question "
     "to be self-contained. If already self-contained, return it as-is. "
     "Return ONLY the rewritten question, no explanation."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions based on the provided context.\n"
     "Context:\n{context}\n\n"
     "Answer concisely and accurately using only the context above. "
     "If the answer is not in the context, say so clearly."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# ── Multi-user retrieval (admin: All Users mode) ──────────────────────────────
def _multi_user_retrieve(query: str, user_ids: list, k: int) -> list:
    all_docs = []
    for uid in user_ids:
        try:
            docs = similarity_retrieve(query, get_vectorstore(uid), k)
            all_docs.extend(docs)
        except Exception:
            pass
    seen, unique = set(), []
    for doc in all_docs:
        fp = doc.page_content[:80]
        if fp not in seen:
            seen.add(fp)
            unique.append(doc)
    return unique[:k]


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def build_components(vectorstore, user_id: str):
    """
    Builds retriever_fn and qa_chain.
    retriever_fn reads st.session_state.admin_query_context at call-time so
    the admin can switch context without rebuilding.
    Regular users always query their own collection (context is ignored).
    """
    llm = get_llm()
    contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

    def retriever_fn(question: str, history: list) -> list:
        q = contextualize_chain.invoke({"question": question, "history": history}) if history else question
        mode = st.session_state.get("search_mode", "similarity")
        k = RETRIEVAL_K

        # Determine which vectorstore(s) to query
        ctx = st.session_state.get("admin_query_context", user_id)
        if ctx == "all":
            docs = _multi_user_retrieve(q, get_all_user_ids(), k)
        elif ctx != user_id:
            # Admin querying a specific other user's collection
            target_vs = get_vectorstore(ctx)
            if mode == "mmr":
                docs = mmr_retrieve(q, target_vs, k, FETCH_K,
                                    st.session_state.get("mmr_lambda", MMR_LAMBDA))
            elif mode == "hybrid":
                docs = hybrid_retrieve(q, target_vs, k, FETCH_K)
            else:
                docs = similarity_retrieve(q, target_vs, k)
        else:
            if mode == "mmr":
                docs = mmr_retrieve(q, vectorstore, k, FETCH_K,
                                    st.session_state.get("mmr_lambda", MMR_LAMBDA))
            elif mode == "hybrid":
                docs = hybrid_retrieve(q, vectorstore, k, FETCH_K)
            else:
                docs = similarity_retrieve(q, vectorstore, k)

        if st.session_state.get("use_reranking"):
            docs = rerank(q, docs, llm, RERANK_TOP_N)
        # Parent-child expansion uses the context user's parents
        if st.session_state.get("use_parent_child"):
            expand_uid = ctx if ctx not in ("all", user_id) else user_id
            docs = expand_to_parents(docs, load_parents(expand_uid))
        return docs

    qa_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {
            "context": "\n\n".join(d.page_content for d in x["docs"]),
            "history": x["history"],
            "question": x["question"],
        })
        | _QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever_fn, qa_chain


# ── Token counting ────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ── Helpers ───────────────────────────────────────────────────────────────────
def file_icon(ftype: str) -> str:
    return {"pdf": "📄", "docx": "📝", "pptx": "📊", "xlsx": "📈",
            "xls": "📈", "url": "🌐"}.get(ftype, "📎")


def _load_conv_into_state(cid: str):
    db_msgs = db.get_messages(cid)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"], "sources": m["sources"]}
        for m in db_msgs
    ]
    lc = []
    for m in db_msgs:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    st.session_state.lc_history = lc


# ═════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown(
        "<style>.block-container{max-width:460px;margin:auto;padding-top:80px}</style>",
        unsafe_allow_html=True,
    )
    st.title("🗂️ Enterprise Doc Assistant")
    st.markdown("---")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if auth.verify_login(username, password):
            st.session_state.username = username
            st.session_state.user_role = auth.get_user(username)["role"]
            st.session_state.session_initialized = False
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.caption("Default credentials — **admin / admin123**  ·  **demo / demo123**")


# ═════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def show_admin():
    import pandas as pd

    st.title("⚙️ Admin Dashboard")

    all_users = auth.list_users()
    stats_by_user = {s["user_id"]: s for s in db.get_user_stats()}
    total_chunks = sum(collection_count(u["username"]) for u in all_users)
    logs = db.get_recent_logs(500)

    # ── Top metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Users", len(all_users))
    col2.metric("💬 Conversations", db.total_conversations())
    col3.metric("🔍 Total Queries", db.total_queries())
    col4.metric("📦 Total Chunks", total_chunks)

    st.markdown("---")
    tab_users, tab_logs, tab_docs, tab_cache, tab_secrets, tab_ragas = st.tabs(
        ["👥 Users", "📊 Query Logs", "📁 Document Index", "⚡ Cache", "🔑 Secrets", "🧪 RAGAS Eval"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # USERS TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_users:
        rows = []
        for u in all_users:
            s = stats_by_user.get(u["username"], {})
            rows.append({
                "Username": u["username"],
                "Role": u["role"],
                "Email": u["email"],
                "Conversations": s.get("conversations", 0),
                "Queries": s.get("queries", 0),
                "Tokens Used": s.get("total_tokens", 0),
                "Chunks Indexed": collection_count(u["username"]),
            })
        df_users = pd.DataFrame(rows)
        st.dataframe(df_users, use_container_width=True, hide_index=True)

        # ── User activity charts ───────────────────────────────────────────
        if len(df_users) > 0 and df_users["Queries"].sum() > 0:
            st.markdown("#### 📊 User Activity")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption("Queries per User")
                chart_q = df_users.set_index("Username")[["Queries"]]
                st.bar_chart(chart_q, use_container_width=True)

            with c2:
                st.caption("Tokens Used per User")
                chart_t = df_users.set_index("Username")[["Tokens Used"]]
                st.bar_chart(chart_t, use_container_width=True)

            with c3:
                st.caption("Chunks Indexed per User")
                chart_c = df_users.set_index("Username")[["Chunks Indexed"]]
                st.bar_chart(chart_c, use_container_width=True)

        st.markdown("---")
        st.markdown("#### ➕ Create User")
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        new_uname = c1.text_input("Username", key="nu")
        new_pw    = c2.text_input("Password", type="password", key="np")
        new_role  = c3.selectbox("Role", ["user", "admin"], key="nr")
        c4.markdown("<br>", unsafe_allow_html=True)
        if c4.button("Create", key="create_btn"):
            if new_uname and new_pw:
                if auth.create_user(new_uname, new_pw, new_role):
                    st.success(f"User '{new_uname}' created")
                    st.rerun()
                else:
                    st.error("Username already exists")

        st.markdown("#### 🗑️ Delete Non-Admin User")
        deletable = [u["username"] for u in all_users if u["role"] != "admin"]
        if deletable:
            del_target = st.selectbox("Select user", deletable, key="del_u")
            if st.button("Delete User", type="secondary"):
                auth.delete_user(del_target)
                st.success(f"User '{del_target}' deleted")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY LOGS TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_logs:
        if logs:
            df = pd.DataFrame(logs)[
                ["user_id", "conversation_id", "question", "tokens_in", "tokens_out", "search_mode", "created_at"]
            ]
            df.columns = ["User", "Conv", "Question", "In Tokens", "Out Tokens", "Mode", "Time"]
            df["Question"] = df["Question"].str[:70]
            df["Time"] = pd.to_datetime(df["Time"].str[:19])

            # ── Charts row ─────────────────────────────────────────────────
            st.markdown("#### 📈 Query Trends")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption("Queries per Day")
                daily = df.groupby(df["Time"].dt.date).size().reset_index(name="Queries")
                daily.columns = ["Date", "Queries"]
                st.line_chart(daily.set_index("Date"), use_container_width=True)

            with c2:
                st.caption("Queries by User")
                by_user = df["User"].value_counts().reset_index()
                by_user.columns = ["User", "Queries"]
                st.bar_chart(by_user.set_index("User"), use_container_width=True)

            with c3:
                st.caption("Search Mode Distribution")
                by_mode = df["Mode"].value_counts().reset_index()
                by_mode.columns = ["Mode", "Count"]
                st.bar_chart(by_mode.set_index("Mode"), use_container_width=True)

            # Token usage over time
            st.markdown("#### 🔢 Token Usage Over Time")
            df["Total Tokens"] = df["In Tokens"] + df["Out Tokens"]
            token_daily = df.groupby(df["Time"].dt.date)["Total Tokens"].sum().reset_index()
            token_daily.columns = ["Date", "Total Tokens"]
            st.area_chart(token_daily.set_index("Date"), use_container_width=True)

            # Raw table
            st.markdown("#### 📋 Raw Query Log")
            df_display = df.copy()
            df_display["Time"] = df_display["Time"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No query logs yet — queries appear here after users chat.")

    # ══════════════════════════════════════════════════════════════════════════
    # DOCUMENT INDEX TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_docs:
        # Chunks per user bar chart
        chunk_data = {u["username"]: collection_count(u["username"]) for u in all_users}
        if any(chunk_data.values()):
            st.markdown("#### 📦 Chunks Indexed per User")
            df_chunks = pd.DataFrame(list(chunk_data.items()), columns=["User", "Chunks"])
            st.bar_chart(df_chunks.set_index("User"), use_container_width=True)

        # File type distribution across all users
        all_files = []
        for u in all_users:
            for f in get_indexed_files(u["username"]):
                all_files.append({"User": u["username"], "Type": f["type"].upper(), "Name": f["name"]})

        if all_files:
            df_files = pd.DataFrame(all_files)
            st.markdown("#### 📂 File Type Distribution")
            type_counts = df_files["Type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            st.bar_chart(type_counts.set_index("Type"), use_container_width=True)

        st.markdown("#### 📁 Per-User Document Details")
        for u in all_users:
            files = get_indexed_files(u["username"])
            cnt = collection_count(u["username"])
            with st.expander(f"**{u['username']}** — {cnt} chunks · {len(files)} sources"):
                if files:
                    for f in files:
                        st.markdown(f"- {file_icon(f['type'])} `{f['name']}`")
                else:
                    st.caption("No documents indexed")

    # ══════════════════════════════════════════════════════════════════════════
    # CACHE TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cache:
        st.markdown(
            "Two-layer semantic cache — **L1 Redis** (exact match, sub-ms) → "
            "**L2 SQLite** (cosine similarity, catches paraphrases). "
            "Every hit avoids a retrieval + LLM call."
        )

        c_stats = cache.stats()
        r_info  = c_stats["redis"]
        s_info  = c_stats["sqlite"]

        # ── Redis status ───────────────────────────────────────────────────────
        st.markdown("#### 🔴 Redis (L1 — Exact Match)")
        if r_info["available"]:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Status", "🟢 Connected")
            rc2.metric("Cached Entries", r_info["entries"])
            rc3.metric("Cache Hits", r_info["hits"])
            st.caption(f"URL: `{r_info['url']}`")
        else:
            st.warning(
                f"Redis not available (`{r_info['url']}`). "
                "L2 SQLite cache is still active. "
                "Start Redis to enable the fast exact-match layer."
            )

        st.markdown("#### 🗄 SQLite (L2 — Semantic Match)")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Cached Entries", s_info["total_entries"])
        sc2.metric("Cache Hits",     s_info["total_hits"])
        sc3.metric("LLM Calls Saved", r_info["hits"] + s_info["total_hits"])

        if s_info["by_scope"]:
            st.markdown("**By Scope**")
            scope_df = pd.DataFrame(s_info["by_scope"])
            scope_df.columns = ["Scope (context)", "Entries", "Hits"]
            st.dataframe(scope_df, use_container_width=True, hide_index=True)

        # ── Cache management ───────────────────────────────────────────────────
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            all_scopes = list({r["scope"] for r in s_info["by_scope"]})
            clear_scope = st.selectbox(
                "Clear cache for scope",
                ["— all scopes —"] + all_scopes,
                key="cache_clear_scope",
            )
        with col_b:
            st.write("")
            st.write("")
            if st.button("🗑 Clear Cache (both layers)", type="secondary"):
                scope_val = None if clear_scope == "— all scopes —" else clear_scope
                cache.clear(scope_val)
                st.success(f"Cache cleared: {'all scopes' if not scope_val else scope_val}")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SECRETS TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_secrets:
        from core import secrets_manager as sm

        st.markdown(
            "Secrets are stored in **Windows Credential Manager** (encrypted). "
            "Keyring values take priority over `.env` — your `.env` acts as a fallback only."
        )

        # ── Status table ───────────────────────────────────────────────────────
        st.markdown("#### Current Status")
        status = sm.secret_status()
        status_icons = {"keyring": "🔒 Keyring", "env": "📄 .env file", "missing": "❌ Missing"}
        rows = [{"Secret": k, "Stored In": status_icons.get(v, v)} for k, v in status.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Update a secret ────────────────────────────────────────────────────
        st.markdown("#### Store / Update a Secret")
        st.caption("Values are written directly to Windows Credential Manager — never to any file.")

        sec_key = st.selectbox("Select secret", sm.MANAGED_KEYS, key="sec_key_select")
        sec_val = st.text_input("New value", type="password", key="sec_val_input",
                                placeholder="Paste the secret value here…")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save to Keyring", type="primary", use_container_width=True):
                if sec_val.strip():
                    sm.set_secret(sec_key, sec_val.strip())
                    st.success(f"✅ {sec_key} saved to Windows Credential Manager.")
                    st.rerun()
                else:
                    st.warning("Enter a value first.")
        with col2:
            if st.button("🗑 Delete from Keyring", use_container_width=True):
                sm.delete_secret(sec_key)
                st.success(f"Deleted {sec_key} from keyring (fallback to .env if set).")
                st.rerun()

        st.markdown("---")
        st.info(
            "**First time setup?** Run `python setup_secrets.py` in your terminal "
            "to migrate all secrets from `.env` to keyring in one step."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RAGAS EVAL TAB
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ragas:
        from core import ragas_eval as reval

        st.markdown("Measure RAG quality with **LLM-as-judge** metrics — no manual labeling needed.")

        # Metric legend
        with st.expander("ℹ️ What each metric means"):
            for m, desc in reval.METRIC_DESCRIPTIONS.items():
                st.markdown(f"- **{m.replace('_', ' ').title()}**: {desc}")

        st.markdown("---")

        # Step 1 — select which user's docs to evaluate
        users_with_docs = [u["username"] for u in all_users if collection_count(u["username"]) > 0]
        if not users_with_docs:
            st.warning("No users have indexed documents yet.")
            return

        eval_user = st.selectbox("📁 Test against documents from", users_with_docs, key="ragas_user")

        # Step 2 — upload or enter test set
        st.markdown("**📋 Test set** — CSV with columns `question` and `ground_truth`")

        sample_csv = "question,ground_truth\nWhat is the main topic of the document?,The document discusses...\nWho is the author?,The author is...\n"
        st.download_button("⬇ Download sample CSV", sample_csv, "ragas_sample.csv", "text/csv")

        csv_file = st.file_uploader("Upload test CSV", type=["csv"], key="ragas_csv")

        if csv_file:
            df_test = pd.read_csv(csv_file)
            required = {"question", "ground_truth"}
            if not required.issubset(df_test.columns):
                st.error(f"CSV must have columns: {required}")
            else:
                st.dataframe(df_test[["question", "ground_truth"]], use_container_width=True, hide_index=True)
                st.caption(f"{len(df_test)} test rows loaded")

                if st.button("▶ Run RAGAS Evaluation", type="primary", key="ragas_run"):
                    questions    = df_test["question"].tolist()
                    ground_truths = df_test["ground_truth"].tolist()

                    vs  = get_vectorstore(eval_user)
                    llm = get_llm()

                    # Build samples with progress
                    status = st.status("Building evaluation samples…", expanded=True)
                    prog   = st.progress(0)

                    def _progress(i, total, q):
                        prog.progress(i / total, f"[{i+1}/{total}] {q[:60]}…")
                        status.write(f"Q{i+1}: {q[:80]}")

                    try:
                        answers, ctx_list = reval.build_samples(
                            questions, ground_truths, vs, llm, RETRIEVAL_K, _progress
                        )
                        status.update(label="Scoring with LLM-as-judge…")
                        prog.progress(0.7)

                        def _score_progress(i, total, label):
                            prog.progress(0.7 + 0.3 * (i / total), label)
                            status.write(label)

                        result_df = reval.score_all(
                            questions, answers, ctx_list, ground_truths, llm, _score_progress
                        )
                        prog.progress(1.0)
                        status.update(label="✅ Evaluation complete!", state="complete", expanded=False)

                        # ── Score cards ────────────────────────────────────
                        metric_cols = [c for c in result_df.columns
                                       if c in {"faithfulness", "answer_relevancy",
                                                "context_precision", "context_recall"}]
                        st.markdown("### 📊 Results")
                        cols = st.columns(len(metric_cols))
                        for col, m in zip(cols, metric_cols):
                            score = round(float(result_df[m].mean()), 3)
                            col.metric(
                                f"{reval.score_color(score)} {m.replace('_', ' ').title()}",
                                f"{score:.3f}",
                                help=reval.METRIC_DESCRIPTIONS.get(m, ""),
                            )

                        # ── Bar chart ──────────────────────────────────────
                        st.markdown("#### Score Comparison")
                        scores_summary = pd.DataFrame({
                            "Metric": [m.replace("_", " ").title() for m in metric_cols],
                            "Score":  [round(float(result_df[m].mean()), 3) for m in metric_cols],
                        })
                        st.bar_chart(scores_summary.set_index("Metric"), use_container_width=True)

                        # ── Per-question table ─────────────────────────────
                        st.markdown("#### Per-Question Breakdown")
                        detail = pd.DataFrame({
                            "Question":   questions,
                            "Answer":     answers,
                            "Ground Truth": ground_truths,
                        })
                        for m in metric_cols:
                            detail[m.replace("_", " ").title()] = result_df[m].round(3).values
                        st.dataframe(detail, use_container_width=True, hide_index=True)

                        # Download results
                        st.download_button(
                            "⬇ Download Results CSV",
                            detail.to_csv(index=False),
                            "ragas_results.csv",
                            "text/csv",
                        )

                    except Exception as e:
                        status.update(label="❌ Evaluation failed", state="error")
                        st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SESSION INIT  (runs before sidebar so processed_files is ready to display)
# ═════════════════════════════════════════════════════════════════════════════
def init_user_session(username: str, is_admin: bool = False):
    if st.session_state.get("session_initialized"):
        return
    st.session_state.session_initialized = True
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("lc_history", [])
    st.session_state.setdefault("tokens_in", 0)
    st.session_state.setdefault("tokens_out", 0)
    st.session_state.setdefault("search_mode", "similarity")
    st.session_state.setdefault("use_reranking", False)
    st.session_state.setdefault("use_parent_child", False)
    st.session_state.setdefault("mmr_lambda", MMR_LAMBDA)

    own_count = collection_count(username)
    any_docs = own_count > 0 or (is_admin and len(get_all_user_ids()) > 0)

    if any_docs:
        # Build components using own VS as primary; context switching is dynamic at query time
        vs = get_vectorstore(username)
        fn, chain = build_components(vs, username)
        st.session_state.retriever_fn = fn
        st.session_state.qa_chain = chain

    st.session_state.processed_files = get_indexed_files(username)
    st.session_state.total_chunks = own_count

    if not st.session_state.get("current_conv"):
        convs = db.get_conversations(username)
        if convs:
            cid = convs[0]["id"]
            st.session_state.current_conv = cid
            _load_conv_into_state(cid)
        else:
            cid = db.create_conversation(username)
            st.session_state.current_conv = cid


# ═════════════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ═════════════════════════════════════════════════════════════════════════════
def show_chat(username: str, is_admin: bool = False):
    # Strategy badge
    mode = st.session_state.get("search_mode", "similarity")
    mode_labels = {"similarity": "Similarity", "mmr": "MMR", "hybrid": "Hybrid BM25+Vector"}
    badges = [f"🔍 **{mode_labels[mode]}**"]
    if st.session_state.get("use_reranking"):
        badges.append("🏆 Re-ranking")
    if st.session_state.get("use_parent_child"):
        badges.append("🌳 Parent-Child")
    st.caption("  ·  ".join(badges))

    # Determine if there are queryable docs based on the active context
    ctx = st.session_state.get("admin_query_context", username) if is_admin else username
    if is_admin and ctx == "all":
        has_docs = len(get_all_user_ids()) > 0
    elif is_admin and ctx != username:
        has_docs = collection_count(ctx) > 0
    else:
        has_docs = collection_count(username) > 0

    # Ensure pipeline is built if context now has docs but pipeline wasn't created yet
    if has_docs and not st.session_state.get("retriever_fn"):
        vs = get_vectorstore(username)
        fn, chain = build_components(vs, username)
        st.session_state.retriever_fn = fn
        st.session_state.qa_chain = chain

    if not has_docs:
        st.info("📂 No documents uploaded yet — you can still chat, but answers will be based on general knowledge only.")

    # Render history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        label = f"{file_icon(src.get('type', ''))} {src.get('name', src.get('source', ''))}"
                        if src.get("page") is not None:
                            label += f"  · page {src['page'] + 1}"
                        st.markdown(f"- {label}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        cid = st.session_state.current_conv

        # Auto-name on first message
        if not st.session_state.get("messages"):
            db.rename_conversation(cid, prompt[:35] + ("…" if len(prompt) > 35 else ""))

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        db.save_message(cid, "user", prompt)
        st.session_state.lc_history.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            # ── Semantic cache lookup ──────────────────────────────────────────
            _cache_scope = st.session_state.get("admin_query_context", username) if is_admin else username
            docs = []  # ensure docs is always defined
            try:
                _q_emb = _embed_question(prompt)
                _cached = cache.lookup_with_question(prompt, _q_emb, _cache_scope)
            except Exception as _ce:
                _q_emb = []
                _cached = None
                st.caption(f"⚠️ Cache error: {_ce}")

            if _cached:
                # Cache hit — render immediately, no LLM call
                _layer_icon = "🔴 Redis" if _cached.get("layer") == "redis" else "🗄 SQLite"
                st.caption(f"⚡ {_layer_icon} cache hit · similarity {_cached['similarity']:.2%} · hit #{_cached['hit_count']}")
                full_answer = _cached["answer"]
                sources = _cached["sources"]
                st.markdown(full_answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            label = f"{file_icon(src.get('type', ''))} {src['name']}"
                            if src.get("page") is not None:
                                label += f"  · page {src['page'] + 1}"
                            st.markdown(f"- {label}")
            else:
                # Cache miss — run full RAG pipeline (or direct LLM if no docs)
                with st.spinner("Thinking…"):
                    if has_docs and st.session_state.get("retriever_fn"):
                        try:
                            from langsmith import traceable
                            _retrieve = traceable(
                                name="retrieve",
                                run_type="retriever",
                                tags=[username, st.session_state.get("search_mode", "similarity")],
                                metadata={"user": username, "conversation": cid},
                            )(st.session_state.retriever_fn)
                            docs = _retrieve(prompt, st.session_state.lc_history[:-1])
                        except ImportError:
                            docs = st.session_state.retriever_fn(prompt, st.session_state.lc_history[:-1])
                    else:
                        docs = []

                sources = []
                for d in docs:
                    m = d.metadata
                    p = m.get("source", "")
                    is_url = p.startswith("http")
                    sources.append({
                        "source": p,
                        "name": p if is_url else Path(p).name,
                        "type": "url" if is_url else Path(p).suffix.lstrip(".").lower(),
                        "page": m.get("page"),
                    })

                if docs:
                    full_answer = st.write_stream(
                        st.session_state.qa_chain.stream({
                            "docs": docs,
                            "history": st.session_state.lc_history[:-1],
                            "question": prompt,
                        })
                    )
                else:
                    # No docs — answer directly from LLM
                    from langchain_core.messages import SystemMessage
                    _direct_msgs = [SystemMessage(content="You are a helpful assistant.")] + \
                                   st.session_state.lc_history
                    full_answer = st.write_stream(get_llm().stream(_direct_msgs))

                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            label = f"{file_icon(src.get('type', ''))} {src['name']}"
                            if src.get("page") is not None:
                                label += f"  · page {src['page'] + 1}"
                            st.markdown(f"- {label}")

                # Store in cache for future similar questions
                if full_answer and _q_emb:
                    cache.store(prompt, _q_emb, str(full_answer), sources, _cache_scope)

        tok_in  = count_tokens(prompt + "\n".join(d.page_content for d in (docs if docs else [])))
        tok_out = count_tokens(full_answer)
        st.session_state.tokens_in  += tok_in
        st.session_state.tokens_out += tok_out

        db.save_message(cid, "assistant", full_answer, sources)
        db.log_query(username, cid, prompt, tok_in, tok_out, mode)
        st.session_state.lc_history.append(AIMessage(content=full_answer))
        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def render_sidebar(username: str, is_admin_user: bool):
    with st.sidebar:

        # ── Shared CSS ─────────────────────────────────────────────────────────
        st.markdown("""
        <style>
        /* ── App brand header ── */
        .sb-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 4px 14px 4px;
        }
        .sb-brand-icon {
            font-size: 26px; line-height: 1;
        }
        .sb-brand-text {
            font-size: 13px; font-weight: 700; letter-spacing: .3px;
            color: #e2e8f0; line-height: 1.3;
        }
        .sb-brand-sub {
            font-size: 10px; color: #64748b; letter-spacing: .5px;
            text-transform: uppercase;
        }

        /* ── User card ── */
        .sb-user-card {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px; margin-bottom: 2px;
        }
        .sb-avatar {
            width: 34px; height: 34px; border-radius: 50%;
            background: linear-gradient(135deg,#6366f1,#8b5cf6);
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px; color: #fff; flex-shrink: 0;
        }
        .sb-user-name {
            font-size: 13px; font-weight: 600; color: #e2e8f0;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .sb-badge {
            display: inline-block; font-size: 9px; font-weight: 700;
            letter-spacing: .6px; text-transform: uppercase;
            padding: 2px 7px; border-radius: 20px; margin-top: 2px;
        }
        .sb-badge-admin { background:#fbbf24; color:#1a1a1a; }
        .sb-badge-user  { background:#3b82f6; color:#fff; }

        /* ── Section label ── */
        .sb-section {
            font-size: 9.5px; font-weight: 700; letter-spacing: 1.3px;
            text-transform: uppercase; color: #475569;
            padding: 14px 2px 4px 2px;
        }

        /* ── Conversation item ── */
        .sb-conv-active {
            background: rgba(99,102,241,0.18);
            border-left: 3px solid #6366f1;
            border-radius: 6px; padding: 5px 8px;
            font-size: 12.5px; font-weight: 600; color: #c7d2fe;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .sb-conv-idle {
            border-left: 3px solid transparent;
            border-radius: 6px; padding: 5px 8px;
            font-size: 12.5px; color: #94a3b8;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        /* ── File item ── */
        .sb-file-row {
            display: flex; align-items: center; gap: 6px;
            padding: 4px 6px; border-radius: 6px;
            background: rgba(255,255,255,0.03);
            margin-bottom: 3px;
            font-size: 12px; color: #cbd5e1;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        /* ── Token pill ── */
        .sb-token-pill {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px; padding: 7px 10px;
            font-size: 10.5px; color: #64748b; text-align: center;
            margin-top: 4px;
        }

        /* Hide default radio dot styling */
        div[data-testid="stRadio"] > label { display: none !important; }

        /* ── Secondary buttons in sidebar — dark styled ── */
        [data-testid="stSidebar"] button[kind="secondary"],
        [data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
            background-color: rgba(255,255,255,0.06) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
            background-color: rgba(99,102,241,0.2) !important;
            color: #c7d2fe !important;
            border-color: rgba(99,102,241,0.4) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Brand header ───────────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-brand">
            <div class="sb-brand-icon">🗂️</div>
            <div>
                <div class="sb-brand-text">Doc Assistant</div>
                <div class="sb-brand-sub">Enterprise RAG · v5</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── User card ──────────────────────────────────────────────────────────
        initial    = username[0].upper()
        badge_cls  = "sb-badge-admin" if is_admin_user else "sb-badge-user"
        badge_text = "Admin" if is_admin_user else "User"
        st.markdown(f"""
        <div class="sb-user-card">
            <div class="sb-avatar">{initial}</div>
            <div style="min-width:0;">
                <div class="sb-user-name">{username}</div>
                <span class="sb-badge {badge_cls}">{badge_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        current_view = st.session_state.get("_view", "chat")
        if is_admin_user:
            nav_c1, nav_c2 = st.columns(2)
            if nav_c1.button(
                "💬 Chat", use_container_width=True,
                type="primary" if current_view == "chat" else "secondary",
            ):
                st.session_state["_view"] = "chat"; st.rerun()
            if nav_c2.button(
                "⚙️ Admin", use_container_width=True,
                type="primary" if current_view == "admin" else "secondary",
            ):
                st.session_state["_view"] = "admin"; st.rerun()
        else:
            st.session_state["_view"] = "chat"

        if st.session_state.get("_view") == "admin":
            st.divider()
            _theme = st.session_state.get("theme", "light")
            _t_meta = _DARK if _theme == "dark" else _LIGHT
            ac1, ac2 = st.columns(2)
            with ac1:
                if ac1.button(
                    f"{_t_meta['toggle_icon']} {_t_meta['toggle_label']}",
                    use_container_width=True, key="theme_toggle_admin"
                ):
                    st.session_state["theme"] = "light" if _theme == "dark" else "dark"
                    st.rerun()
            with ac2:
                if ac2.button("↩ Logout", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
            return

        # ── Conversations ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Conversations</div>', unsafe_allow_html=True)
        if st.button("＋ New Chat", use_container_width=True, type="secondary"):
            cid = db.create_conversation(username)
            st.session_state.current_conv = cid
            st.session_state.messages = []
            st.session_state.lc_history = []
            st.rerun()

        convs = db.get_conversations(username)
        for conv in convs:
            is_active = conv["id"] == st.session_state.get("current_conv")
            c1, c2 = st.columns([5, 1])
            label = conv["name"][:28]
            with c1:
                if is_active:
                    st.markdown(f'<div class="sb-conv-active">▶ {label}</div>', unsafe_allow_html=True)
                    # invisible button to keep click target (active conv doesn't need a click handler)
                else:
                    if st.button(label, key=f"cv_{conv['id']}", use_container_width=True):
                        st.session_state.current_conv = conv["id"]
                        _load_conv_into_state(conv["id"])
                        st.rerun()
            if c2.button("✕", key=f"dc_{conv['id']}"):
                db.delete_conversation(conv["id"])
                remaining = [c for c in convs if c["id"] != conv["id"]]
                if st.session_state.get("current_conv") == conv["id"]:
                    if remaining:
                        st.session_state.current_conv = remaining[0]["id"]
                        _load_conv_into_state(remaining[0]["id"])
                    else:
                        new_cid = db.create_conversation(username)
                        st.session_state.current_conv = new_cid
                        st.session_state.messages = []
                        st.session_state.lc_history = []
                st.rerun()

        # ── Admin: Query Context selector ──────────────────────────────────────
        if is_admin_user:
            st.markdown('<div class="sb-section">Query Context</div>', unsafe_allow_html=True)
            all_uids = get_all_user_ids()
            ctx_opts = (
                ["My Documents"]
                + [f"User: {u}" for u in all_uids if u != username]
                + (["All Users"] if len(all_uids) > 1 else [])
            )
            prev_ctx  = st.session_state.get("_ctx_label", "My Documents")
            ctx_label = st.selectbox(
                "", ctx_opts,
                index=ctx_opts.index(prev_ctx) if prev_ctx in ctx_opts else 0,
                label_visibility="collapsed", key="_ctx_label",
            )
            if ctx_label == "My Documents":
                st.session_state.admin_query_context = username
            elif ctx_label == "All Users":
                st.session_state.admin_query_context = "all"
            else:
                st.session_state.admin_query_context = ctx_label.replace("User: ", "")

        # ── Upload Documents ───────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Upload Documents</div>', unsafe_allow_html=True)
        user_data_dir = os.path.join(DATA_DIR, username)
        os.makedirs(user_data_dir, exist_ok=True)

        uploads = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "pptx", "xlsx", "xls"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        url_input = st.text_input("", placeholder="Or paste a URL — https://…",
                                   label_visibility="collapsed")
        use_pc = st.checkbox(
            "Parent-Child chunking",
            value=st.session_state.get("use_parent_child", False),
            key="use_parent_child",
        )

        if st.button("⬆ Process Documents", type="primary", use_container_width=True):
            already    = {f["path"] for f in st.session_state.get("processed_files", [])}
            new_count  = 0

            for upload in (uploads or []):
                dest = os.path.join(user_data_dir, upload.name)
                if dest in already:
                    continue
                with open(dest, "wb") as fh:
                    fh.write(upload.read())
                try:
                    with st.spinner(f"Indexing {upload.name}…"):
                        chunks = (
                            ingest_file_parent_child(dest, user_id=username)
                            if use_pc else
                            ingest_file(dest, user_id=username)
                        )
                    st.toast(f"✅ {upload.name} → {chunks} chunks")
                    new_count += chunks
                except Exception as e:
                    st.error(f"{upload.name}: {e}")

            if url_input.strip():
                url = url_input.strip()
                if url not in already:
                    try:
                        with st.spinner("Fetching URL…"):
                            chunks = ingest_url(url, user_id=username)
                        st.toast(f"✅ URL → {chunks} chunks")
                        new_count += chunks
                    except Exception as e:
                        st.error(f"URL error: {e}")

            if new_count > 0:
                vs = get_vectorstore(username)
                fn, chain = build_components(vs, username)
                st.session_state.retriever_fn = fn
                st.session_state.qa_chain     = chain
                st.session_state.processed_files = get_indexed_files(username)
                st.session_state.total_chunks    = collection_count(username)
                st.rerun()

        # ── Indexed Documents ──────────────────────────────────────────────────
        ctx = st.session_state.get("admin_query_context", username) if is_admin_user else username

        if is_admin_user and ctx == "all":
            all_uids    = get_all_user_ids()
            total_files = sum(len(get_indexed_files(u)) for u in all_uids)
            total_ch    = sum(collection_count(u) for u in all_uids)
            st.markdown(
                f'<div class="sb-section">Indexed Docs'
                f' <span style="color:#6366f1;font-weight:800;">'
                f'{total_files} files · {total_ch} chunks</span></div>',
                unsafe_allow_html=True,
            )
            for uid in all_uids:
                uid_files = get_indexed_files(uid)
                if uid_files:
                    st.caption(f"👤 {uid}")
                    for f in uid_files:
                        st.markdown(
                            f'<div class="sb-file-row">{file_icon(f["type"])} {f["name"][:24]}</div>',
                            unsafe_allow_html=True,
                        )
        elif is_admin_user and ctx != username:
            ctx_files  = get_indexed_files(ctx)
            ctx_chunks = collection_count(ctx)
            st.markdown(
                f'<div class="sb-section">Indexed Docs — 👤 {ctx}'
                f' <span style="color:#6366f1;font-weight:800;">'
                f'{len(ctx_files)} files · {ctx_chunks} chunks</span></div>',
                unsafe_allow_html=True,
            )
            if ctx_files:
                for f in ctx_files:
                    st.markdown(
                        f'<div class="sb-file-row">{file_icon(f["type"])} {f["name"][:24]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"No documents indexed for '{ctx}'.")
        else:
            processed = st.session_state.get("processed_files", [])
            total_ch  = st.session_state.get("total_chunks", 0)
            st.markdown(
                f'<div class="sb-section">Indexed Docs'
                f' <span style="color:#6366f1;font-weight:800;">'
                f'{len(processed)} files · {total_ch} chunks</span></div>',
                unsafe_allow_html=True,
            )
            if processed:
                for f in processed:
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(
                        f'<div class="sb-file-row">{file_icon(f["type"])} {f["name"][:22]}</div>',
                        unsafe_allow_html=True,
                    )
                    if c2.button("✕", key=f"rm_{f['path']}"):
                        delete_document(f["path"], user_id=username)
                        st.session_state.processed_files = get_indexed_files(username)
                        st.session_state.total_chunks    = collection_count(username)
                        if not st.session_state.processed_files:
                            st.session_state.retriever_fn = None
                            st.session_state.qa_chain     = None
                        st.rerun()
                if st.button("🗑 Clear All", use_container_width=True):
                    clear_vectorstore(username)
                    st.session_state.processed_files = []
                    st.session_state.total_chunks    = 0
                    st.session_state.retriever_fn    = None
                    st.session_state.qa_chain        = None
                    st.rerun()
            else:
                st.caption("No documents yet — upload files above.")

        # ── Search Settings ────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Settings</div>', unsafe_allow_html=True)
        with st.expander("🔧 Search Strategy"):
            st.radio(
                "Strategy",
                ["similarity", "mmr", "hybrid"],
                format_func=lambda x: {
                    "similarity": "Similarity",
                    "mmr": "MMR (diverse)",
                    "hybrid": "Hybrid BM25+Vector",
                }[x],
                key="search_mode",
            )
            if st.session_state.get("search_mode") == "mmr":
                st.slider(
                    "MMR λ  (0 = diverse · 1 = relevant)",
                    0.0, 1.0,
                    float(st.session_state.get("mmr_lambda", MMR_LAMBDA)),
                    0.1, key="mmr_lambda",
                )
            st.checkbox(
                "Re-ranking (LLM)",
                value=st.session_state.get("use_reranking", False),
                key="use_reranking",
            )

        # ── Token counter + Logout ─────────────────────────────────────────────
        ti = st.session_state.get("tokens_in",  0)
        to = st.session_state.get("tokens_out", 0)
        st.markdown(
            f'<div class="sb-token-pill">'
            f'Tokens&nbsp;&nbsp;In <b>{ti:,}</b> · Out <b>{to:,}</b> · Total <b>{ti+to:,}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # ── Theme toggle + Logout ──────────────────────────────────────────────
        _theme = st.session_state.get("theme", "light")
        _t_meta = _DARK if _theme == "dark" else _LIGHT
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button(
                f"{_t_meta['toggle_icon']} {_t_meta['toggle_label']}",
                use_container_width=True, key="theme_toggle"
            ):
                st.session_state["theme"] = "light" if _theme == "dark" else "dark"
                st.rerun()
        with tc2:
            if st.button("↩ Logout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if "username" not in st.session_state:
    show_login()
    st.stop()

_username = st.session_state.username
_is_admin = st.session_state.get("user_role") == "admin"

inject_theme_css()

# Init session BEFORE sidebar so processed_files is populated when sidebar renders
init_user_session(_username, is_admin=_is_admin)

render_sidebar(_username, _is_admin)

if st.session_state.get("_view") == "admin" and _is_admin:
    show_admin()
else:
    show_chat(_username, is_admin=_is_admin)
