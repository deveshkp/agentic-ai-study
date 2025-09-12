import os
import time
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import docx2txt
import faiss

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-oss")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# add configurable timeout/retry settings
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  # seconds
OLLAMA_RETRIES = int(os.environ.get("OLLAMA_RETRIES", "3"))
OLLAMA_BACKOFF = float(os.environ.get("OLLAMA_BACKOFF", "1.5"))


def batch_embed(texts):
    """Embed list of texts with retries on timeout/errors."""
    vectors = []
    for text in texts:
        if not text.strip():
            vectors.append(np.zeros(768, dtype=np.float32))
            continue
        last_exc = None
        for attempt in range(1, OLLAMA_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{OLLAMA_HOST}/api/embeddings",
                    json={"model": EMBED_MODEL, "prompt": text},
                    timeout=OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding") or (data.get("embeddings") and data["embeddings"][0])
                if emb is None:
                    raise ValueError("no embedding in response")
                vectors.append(np.array(emb, dtype=np.float32))
                last_exc = None
                break
            except requests.exceptions.ReadTimeout as e:
                last_exc = e
                print(f"Embedding timeout (attempt {attempt}/{OLLAMA_RETRIES}), backing off...")
            except Exception as e:
                last_exc = e
                print(f"Embedding error (attempt {attempt}/{OLLAMA_RETRIES}): {e}")
            time.sleep(OLLAMA_BACKOFF * attempt)
        if last_exc is not None:
            raise last_exc
    return np.vstack(vectors).astype(np.float32)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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


def load_pdf(path):
    out = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            out.append(txt)
    return out


def load_docx(path):
    text = docx2txt.process(str(path)) or ""
    return [text] if text.strip() else []


def process_file(path):
    loader = load_pdf if path.suffix.lower() == ".pdf" else load_docx
    texts = []
    for page in loader(path):
        texts.extend(split_text(page))
    return texts


def index_folder(root):
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".pdf", ".docx"}]
    all_chunks = []
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(process_file, files))
    for file_chunks in results:
        all_chunks.extend(file_chunks)
    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No text chunks found. Check your documents.")
        return None, []
    vectors = batch_embed(all_chunks)
    print(f"Vectors shape: {vectors.shape}")
    if vectors.size == 0:
        print("No vectors returned from embedding. Check embedding API.")
        return None, all_chunks
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, all_chunks


def search(index, query, all_chunks, top_k=5):
    q_vec = batch_embed([query])[0].reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    return [all_chunks[i] for i in I[0]]


def answer_with_context(question, context_chunks):
    """Ask chat endpoint with retries; fall back to /api/generate. Trim context if too long."""
    # limit total context length to avoid huge requests
    MAX_CHARS = 4000
    context = ""
    for c in context_chunks:
        if len(context) + len(c) + 2 > MAX_CHARS:
            break
        if context:
            context += "\n\n"
        context += c
    prompt = f"Use the following context to answer the question. If the answer is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Try chat endpoint with retries
    last_exc = None
    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {"temperature": 0.2},
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "").strip()
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message") or data["choices"][0].get("content")
                if isinstance(msg, dict):
                    return msg.get("content", "").strip()
                return (msg or "").strip()
            return str(data)
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            print(f"Chat timeout (attempt {attempt}/{OLLAMA_RETRIES}), retrying...")
        except requests.HTTPError as e:
            last_exc = e
            # If 404 or endpoint missing, break and try generate fallback
            if getattr(e.response, "status_code", None) == 404:
                print("Chat endpoint not found; will try /api/generate fallback.")
                last_exc = e
                break
            print(f"Chat HTTP error (attempt {attempt}/{OLLAMA_RETRIES}): {e}")
        except Exception as e:
            last_exc = e
            print(f"Chat error (attempt {attempt}/{OLLAMA_RETRIES}): {e}")
        time.sleep(OLLAMA_BACKOFF * attempt)

    # Fallback to /api/generate
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": CHAT_MODEL, "prompt": prompt, "options": {"temperature": 0.2}},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or str(data)
    except Exception as e:
        raise RuntimeError(f"Both chat and generate failed: {last_exc}; {e}")


if __name__ == "__main__":
    root = "C://Users//deves//workspace//agentic-ai-study//docs"
    index, all_chunks = index_folder(root)
    if index is None or not all_chunks:
        print("Indexing failed or no chunks found. Exiting.")
        raise SystemExit(1)

    try:
        while True:
            question = input("\nEnter your question about the PDFs (blank to quit): ").strip()
            if not question:
                break
            top_chunks = search(index, question, all_chunks, top_k=5)
            # Do not print chunks; only present the final answer.
            answer = answer_with_context(question, top_chunks)
            print("\nFinal Answer:\n", answer)
    except KeyboardInterrupt:
        print("\nExiting.")