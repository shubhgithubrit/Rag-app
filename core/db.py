"""
SQLite database — conversations, messages, query logs.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime

from .config import DB_PATH


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                name       TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources         TEXT DEFAULT '[]',
                created_at      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                conversation_id TEXT,
                question        TEXT,
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                search_mode     TEXT,
                created_at      TEXT NOT NULL
            );
        """)


def _now() -> str:
    return datetime.now().isoformat()


# ── Conversations ─────────────────────────────────────────────────────────────

def create_conversation(user_id: str, name: str = "New Chat") -> str:
    cid = str(uuid.uuid4())[:8]
    now = _now()
    with _conn() as c:
        c.execute("INSERT INTO conversations VALUES (?,?,?,?,?)", (cid, user_id, name, now, now))
    return cid


def get_conversations(user_id: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(cid: str, name: str):
    with _conn() as c:
        c.execute("UPDATE conversations SET name=?, updated_at=? WHERE id=?", (name, _now(), cid))


def delete_conversation(cid: str):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(cid: str, role: str, content: str, sources: list = None):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
            (cid, role, content, json.dumps(sources or []), _now()),
        )
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), cid))


def get_messages(cid: str) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (cid,)
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources") or "[]")
        result.append(d)
    return result


# ── Query logs ────────────────────────────────────────────────────────────────

def log_query(user_id: str, cid: str, question: str,
              tokens_in: int, tokens_out: int, search_mode: str):
    with _conn() as c:
        c.execute(
            "INSERT INTO query_logs "
            "(user_id,conversation_id,question,tokens_in,tokens_out,search_mode,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, cid, question, tokens_in, tokens_out, search_mode, _now()),
        )


# ── Admin queries ─────────────────────────────────────────────────────────────

def get_recent_logs(limit: int = 100) -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_stats() -> list:
    with _conn() as c:
        rows = c.execute("""
            SELECT user_id,
                   COUNT(DISTINCT conversation_id) AS conversations,
                   COUNT(*)                        AS queries,
                   COALESCE(SUM(tokens_in+tokens_out),0) AS total_tokens
            FROM query_logs GROUP BY user_id
        """).fetchall()
    return [dict(r) for r in rows]


def total_queries() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]


def total_conversations() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
