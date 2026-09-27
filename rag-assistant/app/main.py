"""
main.py
-------
The API layer. This file knows nothing about embeddings or FAISS
internals - it just wires HTTP endpoints to the functions in rag.py.
This separation (API vs. pipeline logic) is good practice and also makes
this project easier to explain: "main.py is the interface, rag.py is the
engine."

Run with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive Swagger UI where
you can try /documents and /ask without writing any client code.
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.rag import extract_text_from_pdf, chunk_text, VectorStore, generate_answer

app = FastAPI(
    title="Company Policy RAG Assistant",
    description="Upload PDF policy documents, then ask questions about them.",
)

# --- global state -----------------------------------------------------------
# For a real production app this would live in a database / managed vector
# service. For a learning project, one in-memory (but disk-persisted) store
# per process is simplest and easiest to reason about.
INDEX_PATH = "vector_store"
store = VectorStore()
if os.path.exists(INDEX_PATH + ".faiss"):
    store.load(INDEX_PATH)


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF. It gets: text-extracted -> chunked -> embedded -> stored
    in FAISS. Call this once per document before asking questions.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from this PDF. "
            "It may be a scanned image rather than real text.",
        )

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    chunks_with_meta = [{"text": c, "source": file.filename} for c in chunks]
    store.add_documents(chunks_with_meta)
    store.save(INDEX_PATH)

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks_in_store": len(store.chunks),
    }


@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    Embeds the question, retrieves the top_k most similar chunks from
    FAISS, and asks the LLM to answer using only those chunks.
    """
    if len(store.chunks) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. POST a PDF to /documents first.",
        )

    relevant_chunks = store.search(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, relevant_chunks)

    return {
        "question": req.question,
        "answer": answer,
        "sources": sorted(set(c["source"] for c in relevant_chunks)),
        "chunks_used": len(relevant_chunks),
        "retrieved_chunks": relevant_chunks,  # useful for debugging retrieval quality
    }


@app.get("/health")
async def health():
    return {"status": "ok", "documents_indexed": len(store.chunks)}
