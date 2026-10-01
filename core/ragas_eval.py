"""
RAGAS-style evaluation — LLM-as-judge quality metrics.

Uses simple direct prompts for scoring so it works with any Azure LLM
regardless of whether JSON / structured-output mode is supported.

Metrics:
  Faithfulness      — answer contains only info from context (no hallucination)
  Answer Relevancy  — answer addresses the question
  Context Precision — retrieved chunks are relevant to the question
  Context Recall    — retrieved chunks contain enough info to answer
"""
from __future__ import annotations
import re
import sys
from types import ModuleType

import pandas as pd


# ── Compatibility patches ─────────────────────────────────────────────────────

def _patch_vertexai():
    stub_path = "langchain_community.chat_models.vertexai"
    if stub_path not in sys.modules:
        stub = ModuleType(stub_path)
        class ChatVertexAI: pass
        stub.ChatVertexAI = ChatVertexAI
        sys.modules[stub_path] = stub

_patch_vertexai()

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# ── Constants ─────────────────────────────────────────────────────────────────

METRIC_DESCRIPTIONS = {
    "faithfulness":       "Answer is grounded in retrieved context (no hallucination)",
    "answer_relevancy":   "Answer actually addresses the question asked",
    "context_precision":  "Retrieved chunks are relevant to the question",
    "context_recall":     "Retrieved chunks contain enough info to answer",
}

METRIC_COLS = list(METRIC_DESCRIPTIONS.keys())


# ── Score extraction ──────────────────────────────────────────────────────────

def _extract_score(text: str) -> float | None:
    """Pull the first 0.xx or 1.0 number from an LLM response."""
    text = text.strip()
    match = re.search(r'\b(1\.0+|0\.\d+|\d\.\d+)\b', text)
    if match:
        return min(1.0, max(0.0, float(match.group())))
    match = re.search(r'\b([01])\b', text)
    if match:
        return float(match.group())
    return None


# ── Build evaluation samples (retrieve + answer) ──────────────────────────────

def build_samples(
    questions: list[str],
    ground_truths: list[str],
    vectorstore,
    llm,
    retrieval_k: int,
    progress_cb=None,
) -> tuple[list[str], list[list[str]]]:
    """
    For each question: retrieve context chunks and generate an answer.
    Returns (answers, contexts_list).
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from .retriever import similarity_retrieve

    _qa = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using only the context below.\nContext:\n{context}"),
        ("human", "{question}"),
    ]) | llm | StrOutputParser()

    answers: list[str] = []
    contexts_list: list[list[str]] = []

    for i, q in enumerate(questions):
        if progress_cb:
            progress_cb(i, len(questions), q)
        docs = similarity_retrieve(q, vectorstore, retrieval_k)
        contexts = [d.page_content for d in docs]
        answer = _qa.invoke({"context": "\n\n".join(contexts), "question": q})
        answers.append(answer)
        contexts_list.append(contexts)

    return answers, contexts_list


# ── Direct LLM scoring (no JSON / structured-output required) ─────────────────

def _score_one(llm, prompt: str) -> float:
    try:
        resp = llm.invoke(prompt).content
        s = _extract_score(resp)
        return s if s is not None else 0.5
    except Exception:
        return 0.5


def score_all(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str],
    llm,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Score each question using 4 direct LLM prompts — no JSON mode needed.
    Returns a DataFrame with columns: faithfulness, answer_relevancy,
    context_precision, context_recall.
    """
    rows = []
    n = len(questions)

    for i, (q, ans, ctxs, gt) in enumerate(
        zip(questions, answers, contexts_list, ground_truths)
    ):
        if progress_cb:
            progress_cb(i, n, f"Scoring Q{i+1}")

        ctx_text = "\n".join(ctxs)[:1200]

        faith = _score_one(llm,
            f"Score from 0.0 to 1.0: Does this answer contain ONLY information from the context "
            f"with no hallucination? (1.0 = fully faithful, 0.0 = hallucinated)\n\n"
            f"Context: {ctx_text}\n\nAnswer: {ans}\n\n"
            f"Reply with a single decimal number only, e.g. 0.85"
        )

        relevancy = _score_one(llm,
            f"Score from 0.0 to 1.0: Does this answer directly and completely address the question? "
            f"(1.0 = perfectly relevant, 0.0 = completely off-topic)\n\n"
            f"Question: {q}\n\nAnswer: {ans}\n\n"
            f"Reply with a single decimal number only, e.g. 0.85"
        )

        precision = _score_one(llm,
            f"Score from 0.0 to 1.0: Is the retrieved context relevant and useful for answering the question? "
            f"(1.0 = highly relevant, 0.0 = irrelevant)\n\n"
            f"Question: {q}\n\nContext: {ctx_text}\n\n"
            f"Reply with a single decimal number only, e.g. 0.85"
        )

        recall = _score_one(llm,
            f"Score from 0.0 to 1.0: Does the context contain enough information to produce the ground truth answer? "
            f"(1.0 = all info present, 0.0 = key info missing)\n\n"
            f"Question: {q}\n\nContext: {ctx_text}\n\nGround truth: {gt}\n\n"
            f"Reply with a single decimal number only, e.g. 0.85"
        )

        rows.append({
            "faithfulness":      round(faith, 3),
            "answer_relevancy":  round(relevancy, 3),
            "context_precision": round(precision, 3),
            "context_recall":    round(recall, 3),
        })

    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score) -> str:
    try:
        s = float(score)
        if s >= 0.8: return "🟢"
        if s >= 0.5: return "🟡"
        return "🔴"
    except Exception:
        return "⚪"
