from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .config import settings
from .rag import RAGEngine

app = FastAPI(title="Production-Style RAG API", version="1.0.0", description="Document ingestion and hybrid RAG question answering")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
engine: RAGEngine | None = None

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=20)

@app.on_event("startup")
def startup():
    global engine
    engine = RAGEngine()

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(engine.chunks) if engine else 0}

@app.get("/api/documents")
def documents():
    return {"documents": engine.list_documents()}

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not engine: raise HTTPException(503, "RAG engine is starting")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(415, "Supported file types: PDF, TXT, MD")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    safe_name = Path(file.filename or f"upload-{uuid4()}{suffix}").name
    destination = Path(settings.data_dir) / "uploads" / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    try:
        return engine.ingest(destination)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Document ingestion failed. Check backend logs.") from exc

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not engine: raise HTTPException(503, "RAG engine is starting")
    try:
        return await engine.answer(request.question, request.history)
    except Exception as exc:
        raise HTTPException(502, "Could not generate an answer. Verify Ollama is running and the configured model is available.") from exc
