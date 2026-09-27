"""
rag.py
------
This file contains the four building blocks of the RAG pipeline:

1. extract_text_from_pdf()   -> Document Loader
2. chunk_text()               -> Text Chunking
3. VectorStore                -> Embeddings + Vector Database (FAISS)
4. generate_answer()          -> LLM call
"""

import os
import pickle

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import anthropic


# ---------------------------------------------------------------------------
# 1. LOADER
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text.append(page_text)
    return "\n".join(full_text)


# ---------------------------------------------------------------------------
# 2. CHUNKER
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  
    return chunks


# ---------------------------------------------------------------------------
# 3. EMBEDDINGS + VECTOR DATABASE
# ---------------------------------------------------------------------------
class VectorStore:
  
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)  
        self.chunks: list[dict] = []  

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def add_documents(self, chunks_with_meta: list[dict]) -> None:
        """chunks_with_meta: [{"text": "...", "source": "file.pdf"}, ...]"""
        texts = [c["text"] for c in chunks_with_meta]
        vectors = self.embed(texts).astype("float32")
        self.index.add(vectors)
        self.chunks.extend(chunks_with_meta)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        query_vector = self.embed([query]).astype("float32")
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.chunks[idx])
            result["distance"] = float(dist) 
            results.append(result)
        return results

    def save(self, path: str) -> None:
        """Persist the FAISS index + chunk metadata to disk."""
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str) -> None:
        if os.path.exists(path + ".faiss") and os.path.exists(path + ".pkl"):
            self.index = faiss.read_index(path + ".faiss")
            with open(path + ".pkl", "rb") as f:
                self.chunks = pickle.load(f)


# ---------------------------------------------------------------------------
# 4. LLM CALL
# ---------------------------------------------------------------------------
_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise RuntimeError(
                "API_KEY environment variable is not set. "
                "Get a key from https://console.anthropic.com and set it "
                "before running the server."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "I don't have enough information in the documents to answer that."

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are an assistant that answers employee questions using ONLY the context below, which was retrieved from internal company documents.

Rules:
- Only use information found in the context.
- If the answer is not in the context, say: "I don't have enough information in the documents to answer that."
- Be concise and cite which document(s) you used.

Context:
{context_text}

Question: {question}

Answer:"""

    client = _get_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
