"""
V4 retrieval strategies
  similarity   — standard cosine similarity
  mmr          — Maximal Marginal Relevance (diverse, less redundant)
  hybrid       — BM25 + vector merged via Reciprocal Rank Fusion
  rerank       — single LLM call to pick the most relevant chunks
  parent-child — swap small retrieval chunks for their large parent context
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ── Search strategies ─────────────────────────────────────────────────────────

def similarity_retrieve(query: str, vectorstore, k: int) -> list:
    return vectorstore.similarity_search(query, k=k)


def mmr_retrieve(query: str, vectorstore, k: int, fetch_k: int, lambda_mult: float) -> list:
    """Diverse results: penalises chunks too similar to already-selected ones."""
    return vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )


def hybrid_retrieve(query: str, vectorstore, k: int, fetch_k: int) -> list:
    """
    Reciprocal Rank Fusion of BM25 (keyword) + vector (semantic) search.
    Falls back to similarity if rank_bm25 is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return similarity_retrieve(query, vectorstore, k)

    # Pull every stored chunk for BM25 (only text + metadata needed)
    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    texts, metas = raw["documents"], raw["metadatas"]
    if not texts:
        return []

    # BM25 ranking
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    # Vector ranking
    vec_results = vectorstore.similarity_search_with_score(query, k=fetch_k)

    # RRF merge  (constant 60 dampens rank differences)
    rrf: dict = {}
    for rank, idx in enumerate(bm25_top):
        key = texts[idx]
        rrf[key] = rrf.get(key, 0) + 1 / (rank + 60)
    for rank, (doc, _) in enumerate(vec_results):
        key = doc.page_content
        rrf[key] = rrf.get(key, 0) + 1 / (rank + 60)

    content_to_meta = {t: m for t, m in zip(texts, metas)}
    top_keys = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [
        Document(page_content=c, metadata=content_to_meta.get(c) or {})
        for c in top_keys
    ]


# ── Re-ranking ────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list, llm, top_n: int) -> list:
    """
    Single LLM call: given the question and N passages, pick the top_n most
    relevant by number. Much cheaper than one call per chunk.
    """
    if len(docs) <= top_n:
        return docs

    passages = "\n\n".join(f"[{i+1}] {d.page_content[:500]}" for i, d in enumerate(docs))
    prompt = ChatPromptTemplate.from_template(
        "Select the {top_n} most relevant passages for answering the question. "
        "Reply ONLY with their numbers, comma-separated (e.g. '3,1,4').\n\n"
        "Question: {query}\n\nPassages:\n{passages}"
    )
    try:
        raw = (prompt | llm | StrOutputParser()).invoke(
            {"query": query, "passages": passages, "top_n": top_n}
        )
        indices = [
            int(x.strip()) - 1
            for x in raw.replace(" ", "").split(",")
            if x.strip().isdigit()
        ]
        indices = [i for i in indices if 0 <= i < len(docs)]
        chosen = [docs[i] for i in indices[:top_n]]
        rest = [d for i, d in enumerate(docs) if i not in set(indices[:top_n])]
        return (chosen + rest)[:top_n]
    except Exception:
        return docs[:top_n]


# ── Parent-child expansion ────────────────────────────────────────────────────

def expand_to_parents(docs: list, parents: dict) -> list:
    """
    Replace small child chunks with their larger parent chunks.
    Chunks without a parent_id are returned as-is.
    Deduplicates by parent_id so the same parent isn't repeated.
    """
    seen, result = set(), []
    for doc in docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid in parents:
            if pid not in seen:
                seen.add(pid)
                result.append(parents[pid])
        else:
            uid = doc.page_content[:80]
            if uid not in seen:
                seen.add(uid)
                result.append(doc)
    return result
