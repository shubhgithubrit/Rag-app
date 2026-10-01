"""
Two-layer semantic query cache.

L1 — Redis   : exact/normalized question match, sub-millisecond, no embedding needed.
               Optional — app degrades gracefully if Redis is unavailable.
L2 — SQLite  : cosine similarity on embeddings, catches paraphrased questions.

Lookup order : Redis → SQLite → (caller runs full RAG pipeline)
Store order  : write to both layers on every cache miss.

Cache is scoped to the retrieval context ("alice", "all", …) so answers from
different document sets are never mixed.
Entries expire after CACHE_TTL_DAYS (default 7).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta

import numpy as np

from .config import DB_PATH, REDIS_CACHE_TTL, REDIS_URL

CACHE_TTL_DAYS    = 7
DEFAULT_THRESHOLD = 0.75

# Redis key prefix
_PREFIX = "rag:cache"


# ── Redis (L1) ────────────────────────────────────────────────────────────────

def _get_redis():
    """Return a Redis client, or None if Redis is unavailable."""
    try:
        import redis
        r = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        r.ping()
        return r
    except Exception:
        return None


def _redis_key(scope: str, question: str) -> str:
    normalized = question.strip().lower()
    h = hashlib.sha256(normalized.encode()).hexdigest()[:20]
    return f"{_PREFIX}:{scope}:{h}"


def _redis_lookup(r, scope: str, question: str) -> dict | None:
    try:
        key = _redis_key(scope, question)
        raw = r.get(key)
        if raw:
            data = json.loads(raw)
            # bump hit counter (best-effort)
            hits_key = key + ":hits"
            r.incr(hits_key)
            r.expire(hits_key, REDIS_CACHE_TTL)
            hit_count = int(r.get(hits_key) or 1)
            return {**data, "similarity": 1.0, "hit_count": hit_count, "layer": "redis"}
    except Exception:
        pass
    return None


def _redis_store(r, scope: str, question: str, answer: str, sources: list):
    try:
        key = _redis_key(scope, question)
        payload = json.dumps({"answer": answer, "sources": sources, "question": question})
        r.set(key, payload, ex=REDIS_CACHE_TTL)
    except Exception:
        pass


def _redis_clear(r, scope: str | None = None):
    try:
        pattern = f"{_PREFIX}:{scope}:*" if scope else f"{_PREFIX}:*"
        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)
    except Exception:
        pass


def _redis_stats(r) -> dict:
    try:
        keys = r.keys(f"{_PREFIX}:*")
        # exclude :hits keys
        entry_keys = [k for k in keys if not k.decode().endswith(":hits")]
        total_hits = 0
        for k in entry_keys:
            hits_key = k.decode() + ":hits"
            v = r.get(hits_key)
            total_hits += int(v) if v else 0
        return {
            "available": True,
            "entries":   len(entry_keys),
            "hits":      total_hits,
            "url":       REDIS_URL,
        }
    except Exception:
        return {"available": False, "entries": 0, "hits": 0, "url": REDIS_URL}


# ── SQLite (L2) ───────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now().isoformat()


def init_cache_table():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id          TEXT PRIMARY KEY,
                scope       TEXT NOT NULL,
                question    TEXT NOT NULL,
                embedding   TEXT NOT NULL,
                answer      TEXT NOT NULL,
                sources     TEXT DEFAULT '[]',
                hit_count   INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL,
                last_hit_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cache_scope ON query_cache(scope);
        """)


def _cosine(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _sqlite_lookup(embedding: list, scope: str, threshold: float) -> dict | None:
    if not embedding:
        return None
    cutoff = (datetime.now() - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT id, scope, question, embedding, answer, sources, hit_count "
            "FROM query_cache WHERE scope=? AND created_at > ?",
            (scope, cutoff),
        ).fetchall()

        if not rows:
            return None

        best_sim, best_row = 0.0, None
        for row in rows:
            try:
                stored_emb = json.loads(row["embedding"])
                sim = _cosine(embedding, stored_emb)
                if sim > best_sim:
                    best_sim, best_row = sim, row
            except Exception:
                continue

        if best_sim >= threshold and best_row:
            new_hits = best_row["hit_count"] + 1
            conn.execute(
                "UPDATE query_cache SET hit_count=?, last_hit_at=? WHERE id=?",
                (new_hits, _now(), best_row["id"]),
            )
            conn.commit()
            return {
                "answer":     best_row["answer"],
                "sources":    json.loads(best_row["sources"] or "[]"),
                "question":   best_row["question"],
                "similarity": round(best_sim, 4),
                "hit_count":  new_hits,
                "layer":      "sqlite",
            }
    except Exception:
        pass
    finally:
        conn.close()
    return None


def _sqlite_store(question: str, embedding: list, answer: str, sources: list, scope: str):
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO query_cache "
            "(id, scope, question, embedding, answer, sources, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4())[:8], scope, question,
                json.dumps(embedding), str(answer), json.dumps(sources or []), _now(),
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def _sqlite_clear(scope: str | None = None):
    with _conn() as c:
        if scope:
            c.execute("DELETE FROM query_cache WHERE scope=?", (scope,))
        else:
            c.execute("DELETE FROM query_cache")


def _sqlite_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        hits  = c.execute("SELECT COALESCE(SUM(hit_count),0) FROM query_cache").fetchone()[0]
        by_scope = c.execute(
            "SELECT scope, COUNT(*) AS entries, COALESCE(SUM(hit_count),0) AS hits "
            "FROM query_cache GROUP BY scope ORDER BY entries DESC"
        ).fetchall()
    return {
        "total_entries": total,
        "total_hits":    int(hits),
        "by_scope":      [dict(r) for r in by_scope],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def lookup(embedding: list, scope: str, threshold: float = DEFAULT_THRESHOLD) -> dict | None:
    return _sqlite_lookup(embedding, scope, threshold)


def lookup_with_question(
    question: str,
    embedding: list,
    scope: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict | None:
    r = _get_redis()

    # L1 — Redis
    if r:
        hit = _redis_lookup(r, scope, question)
        if hit:
            return hit

    # L2 — SQLite semantic
    return _sqlite_lookup(embedding, scope, threshold)


def store(question: str, embedding: list, answer: str, sources: list, scope: str):
    """Persist to both L1 (Redis) and L2 (SQLite)."""
    r = _get_redis()
    if r:
        _redis_store(r, scope, question, answer, sources)
    _sqlite_store(question, embedding, answer, sources, scope)


def clear(scope: str | None = None):
    """Clear both layers, optionally scoped."""
    r = _get_redis()
    if r:
        _redis_clear(r, scope)
    _sqlite_clear(scope)


def stats() -> dict:
    """Combined stats from both layers."""
    r = _get_redis()
    redis_info = _redis_stats(r) if r else {"available": False, "entries": 0, "hits": 0, "url": REDIS_URL}
    sqlite_info = _sqlite_stats()
    return {
        "redis":  redis_info,
        "sqlite": sqlite_info,
    }
