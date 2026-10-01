"""
Document ingestion pipeline — PDF, DOCX, PPTX, XLSX/XLS, URLs
All public functions accept user_id for per-user ChromaDB isolation.
"""
import os
import pickle
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    PARENT_CHUNK_SIZE,
    USE_AZURE,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}


# ── Per-user path helpers ─────────────────────────────────────────────────────

def _user_chroma_dir(user_id: str) -> str:
    d = os.path.join(CHROMA_PERSIST_DIR, user_id)
    os.makedirs(d, exist_ok=True)
    return d


def _user_collection(user_id: str) -> str:
    return f"{user_id}_docs"


def _user_parents_file(user_id: str) -> str:
    return os.path.join(_user_chroma_dir(user_id), "parents.pkl")


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings():
    if USE_AZURE:
        return AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return OpenAIEmbeddings()


# ── Splitter ──────────────────────────────────────────────────────────────────

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


# ── File-type loaders ─────────────────────────────────────────────────────────

def _load_pdf(path: str) -> list:
    return PyPDFLoader(path).load()


def _load_docx(path: str) -> list:
    from docx import Document as DocxDoc
    doc = DocxDoc(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [Document(page_content=text, metadata={"source": path, "page": 0})]


def _load_pptx(path: str) -> list:
    from pptx import Presentation
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides):
        texts = [
            shape.text.strip()
            for shape in slide.shapes
            if hasattr(shape, "text") and shape.text.strip()
        ]
        if texts:
            docs.append(Document(
                page_content="\n".join(texts),
                metadata={"source": path, "page": i},
            ))
    return docs


def _load_excel(path: str) -> list:
    import pandas as pd
    xl = pd.ExcelFile(path)
    docs = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet).fillna("")
        text = f"Sheet: {sheet}\n{df.to_string(index=False)}"
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "page": 0, "sheet": sheet},
        ))
    return docs


def _load_url(url: str) -> list:
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(page_content=text, metadata={"source": url, "page": 0})]


# ── Internal store helper ─────────────────────────────────────────────────────

def _add_to_store(raw_docs: list, user_id: str) -> int:
    chunks = _splitter().split_documents(raw_docs)
    # Drop chunks with no usable text (scanned/image-only pages produce empty strings)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No text could be extracted. The file may be a scanned/image-only PDF "
            "or contain only non-text content."
        )
    Chroma(
        collection_name=_user_collection(user_id),
        embedding_function=get_embeddings(),
        persist_directory=_user_chroma_dir(user_id),
    ).add_documents(chunks)
    return len(chunks)


# ── Public ingest functions ───────────────────────────────────────────────────

def ingest_file(path: str, user_id: str = "default") -> int:
    ext = Path(path).suffix.lower()
    loaders = {
        ".pdf":  lambda: _load_pdf(path),
        ".docx": lambda: _load_docx(path),
        ".pptx": lambda: _load_pptx(path),
        ".xlsx": lambda: _load_excel(path),
        ".xls":  lambda: _load_excel(path),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return _add_to_store(loaders[ext](), user_id)


def ingest_url(url: str, user_id: str = "default") -> int:
    return _add_to_store(_load_url(url), user_id)


def ingest_file_parent_child(path: str, user_id: str = "default") -> int:
    ext = Path(path).suffix.lower()
    loaders = {
        ".pdf":  lambda: _load_pdf(path),
        ".docx": lambda: _load_docx(path),
        ".pptx": lambda: _load_pptx(path),
        ".xlsx": lambda: _load_excel(path),
        ".xls":  lambda: _load_excel(path),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported: {ext}")

    raw_docs = loaders[ext]()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    parents = load_parents(user_id)
    child_docs = []
    for i, parent in enumerate(parent_splitter.split_documents(raw_docs)):
        pid = f"{path}::p{i}"
        parents[pid] = parent
        for child in child_splitter.split_documents([parent]):
            child.metadata["parent_id"] = pid
            child_docs.append(child)

    with open(_user_parents_file(user_id), "wb") as f:
        pickle.dump(parents, f)

    return _add_to_store(child_docs, user_id)


# ── Parent-child helpers ──────────────────────────────────────────────────────

def load_parents(user_id: str = "default") -> dict:
    f = _user_parents_file(user_id)
    if os.path.exists(f):
        with open(f, "rb") as fp:
            return pickle.load(fp)
    return {}


# ── Vector store helpers ──────────────────────────────────────────────────────

def get_vectorstore(user_id: str = "default") -> Chroma:
    return Chroma(
        collection_name=_user_collection(user_id),
        embedding_function=get_embeddings(),
        persist_directory=_user_chroma_dir(user_id),
    )


def collection_count(user_id: str = "default") -> int:
    try:
        return (
            chromadb.PersistentClient(path=_user_chroma_dir(user_id))
            .get_collection(_user_collection(user_id))
            .count()
        )
    except Exception:
        return 0


def clear_vectorstore(user_id: str = "default"):
    try:
        chromadb.PersistentClient(path=_user_chroma_dir(user_id)).delete_collection(
            _user_collection(user_id)
        )
    except Exception:
        pass
    f = _user_parents_file(user_id)
    if os.path.exists(f):
        os.remove(f)


def delete_document(source: str, user_id: str = "default"):
    try:
        col = chromadb.PersistentClient(path=_user_chroma_dir(user_id)).get_collection(
            _user_collection(user_id)
        )
        col.delete(where={"source": source})
    except Exception:
        pass


def get_indexed_files(user_id: str = "default") -> list:
    try:
        col = chromadb.PersistentClient(path=_user_chroma_dir(user_id)).get_collection(
            _user_collection(user_id)
        )
        results = col.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                p = meta["source"]
                is_url = p.startswith("http")
                seen[p] = {
                    "name": p if is_url else Path(p).name,
                    "path": p,
                    "type": "url" if is_url else Path(p).suffix.lstrip(".").lower(),
                }
        return list(seen.values())
    except Exception:
        return []


def get_all_user_ids() -> list:
    """Return sorted list of user_ids that have at least one indexed document."""
    result = []
    if os.path.exists(CHROMA_PERSIST_DIR):
        for entry in os.scandir(CHROMA_PERSIST_DIR):
            if entry.is_dir() and collection_count(entry.name) > 0:
                result.append(entry.name)
    return sorted(result)
