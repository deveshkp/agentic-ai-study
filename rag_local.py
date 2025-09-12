#!/usr/bin/env python3
"""
Local RAG Agent using Ollama (chat + embeddings) over a local folder of PDFs and DOCX files.

✅ No external network calls at query time (assumes models already pulled into Ollama)
✅ Pure-Python vector search (SQLite + NumPy) — no faiss required
✅ Simple CLI: index files, then ask questions

Models you need in Ollama (pull once):
    ollama pull gpt-oss           # or your preferred local LLM
    ollama pull nomic-embed-text  # embeddings model

Usage:
    python rag_local.py index --root ~/Users/dp/test
    python rag_local.py ask "What does the onboarding doc say about Kafka retries?" --k 6

Notes:
- PDF text extraction uses pypdf (works for text-based PDFs). Image-only PDFs need OCR (TODO below).
- DOCX extraction uses docx2txt.
- Vectors stored in SQLite as float32 blobs; similarity computed in-memory with NumPy for simplicity and portability.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests
from pypdf import PdfReader
import docx2txt

# -----------------------------
# Config
# -----------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-oss")
DB_PATH = os.environ.get("RAG_DB", ".rag_index.sqlite")
DEFAULT_ROOT = os.environ.get("TEST_ROOT", str(Path.home() / "Users" / "dp" / "test"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

# -----------------------------
# SQLite schema
# -----------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    mtime REAL NOT NULL,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (doc_id, chunk_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

# -----------------------------
# DB helperswinget install Python.Python.3python -m pip install --upgrade pip
# -----------------------------

def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection):
    for stmt in SCHEMA.strip().split(";\n\n"):
        if stmt.strip():
            conn.execute(stmt)
    conn.commit()

# -----------------------------
# Ollama health check (accept tag variants like ":latest")
# -----------------------------

def check_ollama_ready():
    """Verify Ollama is reachable and required models are available (accepting tag variants)."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        names_full = [m.get("name", "") for m in models]
        base_names = {n.split(":", 1)[0] for n in names_full if n}

        def has_model(want: str) -> bool:
            return (
                want in names_full
                or want in base_names
                or any(n.startswith(f"{want}:") for n in names_full)
            )

        missing = []
        if not has_model(CHAT_MODEL):
            missing.append(CHAT_MODEL)
        if not has_model(EMBED_MODEL):
            missing.append(EMBED_MODEL)

        if missing:
            print("⚠️ Required Ollama models not found:", ", ".join(missing))
            print("Available models:")
            for n in sorted(names_full):
                print("  -", n)
            print("Hint: try `ollama pull <model>:latest` or set CHAT_MODEL/EMBED_MODEL env vars.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Could not connect to Ollama at {OLLAMA_HOST}: {e}")
        print("Make sure the Ollama service is running.")
        sys.exit(1)

# -----------------------------
# File loading & chunking
# -----------------------------

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pdf(path: Path) -> List[Tuple[int, str]]:
    out = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            out.append((i + 1, txt))
    return out


def load_docx(path: Path) -> List[Tuple[int, str]]:
    text = docx2txt.process(str(path)) or ""
    if not text.strip():
        return []
    return [(1, text)]


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx"}:
            yield p


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_size)
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks

# -----------------------------
# Ollama API wrappers
# -----------------------------

def ollama_embed(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    vectors = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": model, "prompt": t},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["embedding"], dtype=np.float32)
        vectors.append(vec)
    return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)


def ollama_chat(messages: List[dict], model: str = CHAT_MODEL, temperature: float = 0.2) -> str:
    payload = {"model": model, "messages": messages, "options": {"temperature": temperature}, "stream": False}
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    # Prefer non-streaming JSON; fallback to NDJSON parsing if server streamed anyway
    try:
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.JSONDecodeError:
        text = resp.text.strip()
        pieces = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get("message", {}).get("content")
            if msg:
                pieces.append(msg)
        return "".join(pieces)
# -----------------------------
# Indexing pipeline
# -----------------------------

def upsert_document(conn: sqlite3.Connection, path: Path) -> str:
    mtime = path.stat().st_mtime
    h = file_hash(path)
    doc_id = hashlib.md5(str(path).encode()).hexdigest()
    cur = conn.execute("SELECT mtime, hash FROM documents WHERE doc_id=?", (doc_id,))
    row = cur.fetchone()
    if row and abs(row[0] - mtime) < 1e-6 and row[1] == h:
        return doc_id  # up-to-date
    conn.execute(
        "REPLACE INTO documents(doc_id, path, mtime, hash) VALUES(?,?,?,?)",
        (doc_id, str(path), mtime, h),
    )
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.commit()
    return doc_id


def index_path(conn: sqlite3.Connection, root: Path):
    ensure_schema(conn)
    root = root.expanduser().resolve()
    print(f"Indexing: {root}")
    to_texts: list[tuple[str, str, int]] = []  # (doc_id, text, page)

    for path in iter_files(root):
        loader = load_pdf if path.suffix.lower() == ".pdf" else load_docx
        pages = loader(path)
        if not pages:
            continue
        doc_id = upsert_document(conn, path)
        for page_num, text in pages:
            for piece in split_text(text):
                to_texts.append((doc_id, piece, page_num))

    if not to_texts:
        print("No content found to index.")
        return

    print(f"Embedding {len(to_texts)} chunks with {EMBED_MODEL}…")
    vectors = ollama_embed([t for _, t, _ in to_texts])
    print(f"Vectors shape: {vectors.shape}")

    with conn:
        # We also need file path mapping; fetch once per doc_id
        doc_paths = {doc_id: conn.execute("SELECT path FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
                     for doc_id, _, _ in to_texts}
        # Insert chunks
        counters: dict[str, int] = {}
        for (doc_id, text, page), vec in zip(to_texts, vectors):
            cidx = counters.get(doc_id, 0)
            conn.execute(
                "INSERT OR REPLACE INTO chunks(doc_id, chunk_id, page, text, vector) VALUES(?,?,?,?,?)",
                (doc_id, cidx, page, text, vec.tobytes()),
            )
            counters[doc_id] = cidx + 1
    print("Index complete.")

# -----------------------------
# Retrieval + Answer
# -----------------------------

def load_all_vectors(conn: sqlite3.Connection) -> Tuple[np.ndarray, list[tuple[str,int,int,str]]]:
    cur = conn.execute(
        "SELECT c.doc_id, c.chunk_id, c.page, c.vector, d.path FROM chunks c JOIN documents d ON c.doc_id=d.doc_id"
    )
    metas = []
    vecs = []
    for doc_id, chunk_id, page, vec_blob, path in cur:
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        vecs.append(vec)
        metas.append((doc_id, chunk_id, page or -1, path))
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    mat = np.vstack(vecs)
    return mat, metas


def top_k(conn: sqlite3.Connection, query: str, k: int = 6):
    q_vec = ollama_embed([query])[0]
    mat, metas = load_all_vectors(conn)
    if mat.size == 0:
        return []
    q = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    M = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sims = M @ q
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        sim = float(sims[i])
        doc_id, chunk_id, page, path = metas[i]
        row = conn.execute(
            "SELECT text FROM chunks WHERE doc_id=? AND chunk_id=?",
            (doc_id, chunk_id),
        ).fetchone()
        text = row[0] if row else ""
        out.append((sim, {"path": path, "page": None if page == -1 else page, "text": text}))
    return out


def build_context(snips, max_words: int = 1800) -> str:
    parts = []
    used = 0
    for sim, s in snips:
        header = f"\n[Source: {Path(s['path']).name} | page {s['page']}]\n" if s["page"] else f"\n[Source: {Path(s['path']).name}]\n"
        body_words = s["text"].split()
        take = min(len(body_words), max(0, max_words - used))
        if take <= 0:
            break
        parts.append(header + " ".join(body_words[:take]))
        used += take
    return "\n".join(parts)


def answer(conn: sqlite3.Connection, question: str, k: int = 6) -> str:
    hits = top_k(conn, question, k=k)
    if not hits:
        return "I couldn't find any indexed content yet. Try running the index command first."
    context = build_context(hits)
    system = (
        "You are a precise assistant. Answer the user's question using ONLY the given context.\n"
        "If the answer isn't in the context, say you couldn't find it. Be concise.\n"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    resp = ollama_chat(messages, model=CHAT_MODEL)
    cite_lines = [
        f"- {Path(s['path']).name}{(' p.' + str(s['page'])) if s['page'] else ''} (score={sim:.3f})"
        for sim, s in hits
    ]
    return resp.strip() + "\n\nSources:\n" + "\n".join(cite_lines)

# -----------------------------
# CLI
# -----------------------------

def cli():
    p = argparse.ArgumentParser(description="Local RAG Agent over PDFs/DOCX using Ollama")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index all PDFs and DOCX under a folder")
    p_index.add_argument("--root", default=DEFAULT_ROOT, help="Folder to scan (default: ~/Users/dp/test)")

    p_ask = sub.add_parser("ask", help="Ask a question against the local index")
    p_ask.add_argument("question", help="Your question text")
    p_ask.add_argument("--k", type=int, default=6, help="Top-k chunks to retrieve")

    args = p.parse_args()

    # Check Ollama first
    check_ollama_ready()
    print("✅ Ollama service is up and models are ready.")

    if args.cmd == "index":
        root = Path(args.root)
        if not root.exists():
            print(f"Folder not found: {root}", file=sys.stderr)
            sys.exit(2)
        conn = db_connect(DB_PATH)
        ensure_schema(conn)  # ensure tables exist
        index_path(conn, root)
        return

    if args.cmd == "ask":
        conn = db_connect(DB_PATH)
        ensure_schema(conn)  # create tables if db is new/empty
        try:
            print(answer(conn, args.question, k=args.k))
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                print("Index not found yet. Run the index command first, e.g.""  python rag_local.py index --root ~/docs/")
            else:
                raise
        return

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    cli()

# -----------------------------
# Enhancements / TODOs
# -----------------------------
# - Add OCR for image-only PDFs: pip install pytesseract pillow; run OCR when pypdf returns empty
# - Streamlit UI wrapper (file browser, Q/A box, source viewer)
# - Watchman-style reindex on file changes
# - Swap to FAISS or sqlite-vector for larger corpora
# - Add per-file ignore globs and file-type filters
# - Add JSONL export/import of the index for portability
# - Add evaluation mode (gold Q/A) and retrieval metrics
