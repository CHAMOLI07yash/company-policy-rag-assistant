# Company Policy RAG Assistant

A minimal, working **Retrieval-Augmented Generation (RAG)** pipeline: upload
PDF policy documents, ask questions in plain English, get answers grounded
in those documents — not the model's imagination.

This project exists to be **understood end to end**, not just run. Read the
"How it works" and "Design decisions" sections below before you put it on
your resume — an interviewer will ask *why*, not just *what*.

---

## 0. What is RAG, in one paragraph

An LLM like Claude only knows what it was trained on — it has never seen
your company's internal PDFs. RAG fixes this without retraining the model:
you **retrieve** the few most relevant pieces of your documents for a given
question, then **stuff them into the prompt** as context, and let the LLM
**generate** an answer using that context. Retrieval + Generation = RAG.

---

## 1. Architecture

```
                PDF Documents
                     |
                     v
              Document Loader        (pypdf: extracts raw text)
                     |
                     v
                Text Chunking        (split into ~500-word overlapping pieces)
                     |
                     v
                Embeddings           (sentence-transformers: text -> vector)
                     |
                     v
             Vector Database         (FAISS: stores vectors, does similarity search)
                     ^
                     |
User Question -> Embedding
                     |
                     v
             Similarity Search       (find chunks whose vectors are closest to the question's vector)
                     |
                     v
          Relevant Text Chunks
                     |
                     v
                    LLM               (Claude: answer using ONLY those chunks)
                     |
                     v
                Final Answer
```

Two REST endpoints expose this:

| Endpoint       | What it does                                                        |
|----------------|----------------------------------------------------------------------|
| `POST /documents` | PDF → extract text → chunk → embed → store in FAISS               |
| `POST /ask`       | Question → embed → search FAISS → top-k chunks → LLM → answer     |

## 2. Project structure

```
rag-assistant/
├── app/
│   ├── main.py     # FastAPI layer: HTTP endpoints only
│   └── rag.py      # the actual pipeline: loader, chunker, vector store, LLM call
├── requirements.txt
├── .env.example
└── README.md
```

Splitting `main.py` (API) from `rag.py` (pipeline) is deliberate: you can
unit-test chunking or the vector store without ever starting a web server,
and you can explain the project in an interview as two clean layers.

## 3. Setup & running it

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key (get one at console.anthropic.com)
export ANTHROPIC_API_KEY=sk-ant-your-key-here

# 4. Run the server
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** — FastAPI gives you a free interactive
UI (Swagger) where you can upload a PDF and ask questions with no extra
tooling.

### Or use curl

```bash
# Upload a document
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@company_policy.pdf"

# Ask a question
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many annual leave days can an employee take?"}'
```

Example response:
```json
{
  "question": "How many annual leave days can an employee take?",
  "answer": "According to company_policy.pdf, employees are entitled to 18 days of annual leave per calendar year, accrued monthly...",
  "sources": ["company_policy.pdf"],
  "chunks_used": 3
}
```

The first time you run it, `sentence-transformers` will download the
embedding model (~80MB) automatically — this needs internet access once,
then it's cached locally.

## 4. How it works, step by step

### Step A — Document Loader (`extract_text_from_pdf`)
Uses `pypdf` to pull raw text out of each page of the PDF and joins it into
one string. This only works for PDFs that contain actual text (exported
from Word, Google Docs, etc.) — a scanned photo of a page has no embedded
text layer, so extraction would return an empty string.

### Step B — Chunking (`chunk_text`)
Splits the text into ~500-word pieces with a 50-word overlap between
consecutive chunks.
- **Why chunk?** LLM context windows and embedding models both have
  limits, and smaller chunks make retrieval *precise* — if the whole
  document were one chunk, every question would just retrieve the entire
  document.
- **Why overlap?** Without it, a sentence sitting right on a chunk
  boundary gets cut in half and its meaning is weakened in both halves.

### Step C — Embeddings
Each chunk is passed through `all-MiniLM-L6-v2`, a small open-source
sentence-transformer model, turning it into a 384-dimensional vector — a
list of numbers representing its *meaning*. Two chunks about "annual
leave" and "vacation days" end up with mathematically similar vectors even
though they don't share many words, which is exactly what makes semantic
search more powerful than plain keyword search (Ctrl+F).

### Step D — Vector Database (FAISS)
FAISS (Facebook AI Similarity Search) stores every chunk's vector and can
find the ones closest to a new vector extremely fast, even across millions
of vectors. Here we use `IndexFlatL2`, the simplest exact-search index —
perfect for a few thousand chunks; see "Improvements" for scaling further.

### Step E — Retrieval
The user's question is embedded the same way as the chunks were, and FAISS
returns the `top_k` (default 5) chunks whose vectors are closest to it.

### Step F — Generation (Claude)
The retrieved chunks are inserted into a prompt template that explicitly
instructs Claude to **answer only using the provided context**, and to say
so plainly if the answer isn't there. This single instruction is what
turns a generic chatbot into a grounded, trustworthy document assistant.

## 5. Design decisions (and why)

| Decision | Reasoning |
|---|---|
| `chunk_size=500`, `overlap=50` | Good default for policy-style prose; small enough for precise retrieval, big enough to keep a full policy clause together most of the time. |
| `all-MiniLM-L6-v2` for embeddings | Free, open-source, runs on CPU, ~80MB — no API cost per chunk embedded, good enough quality for a portfolio project. |
| `IndexFlatL2` (exact search) | Simple and correct. Only becomes a bottleneck above ~1M vectors, far beyond a handful of policy PDFs. |
| Separate `/documents` and `/ask` endpoints | Mirrors the two distinct phases of RAG (indexing vs. querying) and matches how you'd design this in a real product. |
| Prompt explicitly says "answer only from context" | Without it, the LLM will confidently hallucinate a plausible-sounding but fabricated policy. |
| FAISS index + metadata persisted to disk | So you don't have to re-upload and re-embed documents every time you restart the server. |

## 6. Failure modes I ran into (and how I'd explain them in an interview)

These are the kinds of things interviewers actually want to hear about —
not that everything worked perfectly, but that you understand *why*
things broke and how you'd fix them.

1. **Scanned PDFs return empty text.** `pypdf` can only extract text that's
   actually embedded in the PDF. A scanned contract is just an image, so
   extraction silently returns nothing. *Fix:* detect empty extraction and
   fall back to OCR (e.g. `pytesseract` + `pdf2image`).

2. **Hallucination without a strict prompt.** Early on, without the
   "answer only from context" instruction, the LLM answered *every*
   question confidently, including ones the documents never covered —
   inventing a leave policy that sounded right but wasn't in any PDF.
   *Fix:* the explicit instruction + fallback phrase in the prompt, shown
   in `generate_answer()`.

3. **Chunk boundaries splitting key sentences.** With no overlap, a
   sentence like "employees are entitled to 18 days of leave" could get
   split so that "18 days" ends up in one chunk and "of leave" in the
   next, weakening both chunks' embeddings. *Fix:* the 50-word overlap.

4. **Irrelevant chunks pulled in for vague questions.** A very short or
   ambiguous question (e.g. "leave?") embeds to a vector that's not
   clearly close to any one chunk, so FAISS can return weakly-relevant
   results, which the LLM then has to work around. *Fix ideas:* re-ranking
   retrieved chunks with a cross-encoder, or asking the user to clarify
   when similarity scores are all low.

5. **No source-conflict handling.** If two uploaded PDFs disagree (e.g. an
   outdated handbook vs. an updated policy), both get retrieved and the
   LLM has to silently pick one — with no guarantee it explains the
   conflict. *Fix idea:* tag documents with version/date metadata and
   prefer the newest, or explicitly ask the LLM to flag contradictions.

## 7. Improvements I'd make next (great "future work" resume/interview talking points)

- **Hybrid search:** combine FAISS vector search with keyword search
  (BM25) — vector search misses exact matches like policy code numbers.
- **Re-ranking:** run the top ~20 FAISS results through a cross-encoder
  re-ranker to pick the best 5, improving precision.
- **Streaming responses:** stream the LLM's answer token-by-token to the
  client instead of waiting for the full response.
- **Chat history / multi-turn:** let users ask follow-up questions that
  reference the previous answer.
- **Evaluation:** build a small labeled set of (question, correct answer)
  pairs and measure retrieval accuracy and answer quality over time.
- **Swap FAISS for a managed vector DB** (Pinecone, Weaviate, Chroma) for
  multi-user, multi-tenant, or larger-scale deployments.
- **OCR fallback** for scanned PDFs.
- **Guardrails / citation checking:** verify the LLM's answer actually
  quotes something present in the retrieved chunks before returning it.
- **Dockerize** the app for one-command deployment.
- **Auth** on the endpoints before exposing this beyond localhost.

## 8. How to talk about this on your resume

Example bullet point:

> Built an end-to-end Retrieval-Augmented Generation (RAG) pipeline
> (FastAPI + FAISS + sentence-transformers + Claude) that answers natural-
> language questions from uploaded PDF documents, including document
> chunking, semantic embedding search, and grounded LLM generation.

Be ready to explain, in your own words:
- What retrieval-augmented generation is and why it beats fine-tuning for
  "answer questions about my documents."
- Why you chunk text and what happens if you don't.
- What an embedding is, at a conceptual level (turns meaning into
  numbers so "similar meaning" becomes "similar vectors").
- What FAISS does and why exact vs. approximate search matters at scale.
- At least one failure you hit and how you fixed it (Section 6 above).
- At least one improvement you'd make with more time (Section 7 above).
