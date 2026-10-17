# Production-Style RAG: React + FastAPI + FAISS + BM25 + RRF + Cross-Encoder + Ollama

A runnable starter application for asking questions about uploaded PDF, TXT, and Markdown documents. It includes a React chat UI, FastAPI endpoints, persistent local FAISS index/metadata, hybrid dense + lexical retrieval, reciprocal rank fusion, cross-encoder reranking, and Ollama generation with source snippets.

## Architecture

1. Upload PDF/TXT/MD → extract text with PyMuPDF or UTF-8 reader.
2. Split text into overlapping chunks and attach source/chunk metadata.
3. Embed chunks using Sentence Transformers; persist a normalized FAISS inner-product index.
4. Retrieve candidates using dense FAISS search and BM25 lexical search.
5. Merge ranked results with Reciprocal Rank Fusion (RRF).
6. Rerank candidates using a Cross-Encoder and keep the best chunks.
7. Send selected context to Ollama; instruct it to cite source filename/chunk and abstain when context is insufficient.

## Requirements

- Python 3.10+ recommended
- Node.js 20+ recommended
- Ollama installed and running
- RAM/disk for embedding and reranker models (first run downloads models)

## Run locally (Windows PowerShell)

### 1. Start Ollama

Install Ollama, then run:

```powershell
ollama pull llama3.2
ollama serve
```

If Ollama is already running as a background service, you may not need `ollama serve`. Check `http://localhost:11434`.

### 2. Start backend

```powershell
cd production-rag\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

On first startup, Sentence Transformers downloads the embedding and reranker models. API docs: http://localhost:8000/docs

### 3. Start frontend in another terminal

```powershell
cd production-rag\frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually http://localhost:5173.

## API endpoints

- `GET /health` — service health and indexed chunk count
- `GET /api/documents` — indexed documents
- `POST /api/documents` — multipart upload (`file` field)
- `POST /api/chat` — JSON body: `{"question":"What is the policy?","history":[]}`

## Configuration

Copy `backend/.env.example` to `backend/.env`. Change `OLLAMA_MODEL` to a model already pulled locally. The app uses `http://localhost:11434` by default.

## Smoke tests

With backend running:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Upload a document using the UI, then ask a question whose answer appears in the document. Check the displayed source snippets and the `/docs` endpoint.

## Important production-hardening work before real deployment

This is a production-style learning starter, not a security-reviewed production service. Before exposing it to users:

- Add authentication/authorization and tenant isolation; enforce per-user document access.
- Add malware scanning, MIME/signature validation, upload rate limits, quotas, and background ingestion jobs.
- Store documents and metadata in durable managed storage; use a database for status, ownership, audit logs, and deletion.
- Make ingestion idempotent and atomic; persist index version/model version and rebuild indexes safely.
- Add tests for chunking, retrieval, API errors, prompt injection, source citations, and evaluation datasets.
- Add structured logging, tracing, metrics, latency/cost tracking, and alerting.
- Add streaming responses, cancellation, concurrency limits, and a bounded queue for model calls.
- Treat retrieved documents as untrusted data; the prompt instruction is only one defense, not a complete prompt-injection control.
- Configure explicit CORS origins, secrets management, TLS, backups, and retention/deletion policies.
- For multi-user or large-scale workloads, consider PostgreSQL + pgvector or a managed vector store instead of a single local FAISS index.

## Known starter limitations

- The FAISS index and chunk metadata are local files, and the in-memory BM25 index is rebuilt at startup.
- Re-ingestion replaces chunks for a matching filename, but this starter does not yet implement a delete-document endpoint or transactional index updates.
- Scanned PDFs need OCR, which is not included.
- The first model download can take time and memory; CPU inference may be slow.
- The API currently returns complete answers rather than streaming tokens.
