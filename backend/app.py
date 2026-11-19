import os
import sys
import shutil
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(PARENT_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from backend.rag.ingest import ingest_file
from backend.rag import retrieve as rag_retrieve
from backend.rag.retrieve import get_rag_db
from backend.rag.generate import generate_answer

# Define App directories
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="Educational RAG API",
    description="Backend API for parsing student PDFs/PPTs, indexing text, and running RAG queries.",
    version="1.0.0"
)

# Enable CORS for frontend integration (Vite + Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request validation schemas
import datetime
import json

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5

class QueryMetrics(BaseModel):
    latency_seconds: float
    tokens_generated: int
    tokens_per_second: float
    provider: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metrics: QueryMetrics

# Observability File Logger Path
OBSERVABILITY_LOG_PATH = os.path.join(BASE_DIR, "rag_observability.log")
DOCUMENT_INDEX_LOCK = threading.Lock()
_INDEXED_CACHE = {}
_INDEXED_CACHE_LOCK = threading.Lock()

def write_observability_log(query: str, chunks: List[Dict[str, Any]], result: Dict[str, Any], db_type: str):
    """
    Writes a comprehensive audit entry to backend/rag_observability.log
    tracing query parameters, fused rank scores, re-ranker weights, and SLM telemetry.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "active_db": db_type,
        "metrics": result.get("metrics", {}),
        "retrieved_chunks": [
            {
                "source": c.get("metadata", {}).get("source", "unknown"),
                "type": c.get("metadata", {}).get("type", "unknown"),
                "page_or_slide": c.get("metadata", {}).get("page_number", c.get("metadata", {}).get("slide_number", "unknown")),
                "similarity_score": round(c.get("similarity", 0.0), 4),
                "vector_rank": c.get("vector_rank"),
                "bm25_rank": c.get("bm25_rank"),
                "rrf_score": round(c.get("rrf_score", 0.0), 4) if "rrf_score" in c else None,
                "cross_encoder_score": round(c.get("cross_encoder_score", 0.0), 4) if "cross_encoder_score" in c else None
            }
            for c in chunks
        ],
        "exit_status": "success" if "error" not in result.get("metrics", {}).get("provider", "").lower() else "failed"
    }
    
    with open(OBSERVABILITY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

# Endpoints
def _library_status():
    """
    Health for the UI. Disk uploads are the source of truth so this never
    waits on vector-store startup or a long indexing job.
    """
    uploads = _disk_documents()
    db_manager = get_rag_db()
    
    if db_manager is None:
        return {
            "active_db": "chromadb",
            "status": "starting",
            "document_count": len(uploads),
            "uploads_count": len(uploads),
            "bm25_active": False,
        }

    try:
        indexed_documents = len(db_manager.list_documents())
        chunks = len(db_manager.bm25_retriever.corpus)
    except Exception as exc:
        print(f"Status check could not read vector store: {exc}")
        indexed_documents = 0
        chunks = 0

    return {
        "active_db": getattr(db_manager, "db_type", "chromadb"),
        "status": "healthy",
        "document_count": len(uploads),
        "uploads_count": len(uploads),
        "indexed_documents": indexed_documents,
        "chunk_count": chunks,
        "bm25_active": chunks > 0,
    }


@app.get("/api/status")
def get_status():
    """
    Returns API health and underlying database configuration status.
    """
    return _library_status()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a PDF or PPTX document, parses it, chunks it, embeds it,
    and indexes it in the vector store.
    """
    file_name = os.path.basename(file.filename or "")
    if not file_name:
        raise HTTPException(status_code=400, detail="A filename is required.")

    ext = os.path.splitext(file_name)[1].lower()
    
    if ext not in ['.pdf', '.pptx', '.ppt', '.txt', '.md']:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Only PDFs, PPTXs, and plain text files are supported."
        )
        
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        chunks_count = await run_in_threadpool(_save_and_index_upload, file_name, contents)

        if chunks_count == 0:
            return {
                "message": f"Uploaded {file_name}. Note: No readable text was extracted (e.g. scanned image PDF or empty slides).",
                "filename": file_name,
                "chunks_count": 0,
                "status": "warning"
            }

        return {
            "message": f"Successfully indexed {file_name} ({chunks_count} chunks).",
            "filename": file_name,
            "chunks_count": chunks_count,
            "status": "success"
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

SUPPORTED_UPLOAD_EXTS = {".pdf", ".pptx", ".ppt", ".txt", ".md"}


def _disk_documents():
    docs = []
    for name in os.listdir(UPLOADS_DIR):
        if name.startswith("."):
            continue
        file_path = os.path.join(UPLOADS_DIR, name)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_UPLOAD_EXTS:
            continue
        docs.append({"source": name, "type": ext.lstrip(".")})
    return docs


def _save_upload(file_name: str, contents: bytes) -> str:
    save_path = os.path.join(UPLOADS_DIR, file_name)
    with open(save_path, "wb") as buffer:
        buffer.write(contents)
    print(f"Saved uploaded file to: {save_path}")
    return save_path


def _index_saved_file(file_name: str) -> int:
    """Index a saved upload. Raises indexing errors so the upload API can report them."""
    save_path = os.path.join(UPLOADS_DIR, file_name)
    with DOCUMENT_INDEX_LOCK:
        if not os.path.isfile(save_path):
            raise FileNotFoundError(f"Uploaded file missing on disk: {file_name}")
        chunks = ingest_file(save_path, file_name)
        if not chunks:
            print(f"No readable text extracted from {file_name}")
            return 0
        get_rag_db().add_documents(chunks)
        _remember_indexed_document(file_name, chunks)
        print(f"Indexed {file_name} ({len(chunks)} chunks).")
        return len(chunks)


def _save_and_index_upload(file_name: str, contents: bytes) -> int:
    _save_upload(file_name, contents)
    return _index_saved_file(file_name)


def _sync_indexed_cache_from_db():
    try:
        db_manager = get_rag_db()
        for doc in db_manager.list_documents():
            if isinstance(doc, dict) and doc.get("source"):
                source = doc["source"]
                with _INDEXED_CACHE_LOCK:
                    _INDEXED_CACHE[source] = {
                        "source": source,
                        "type": doc.get("type", "unknown"),
                        "total_pages": int(doc.get("total_pages") or 0),
                        "slide_count": int(doc.get("slide_count") or 0),
                        "indexed": True,
                    }
    except Exception as e:
        print(f"Syncing index cache error: {e}")


def _index_unindexed_uploads():
    with DOCUMENT_INDEX_LOCK:
        _sync_indexed_cache_from_db()
        db_manager = get_rag_db()
        indexed = {
            doc.get("source")
            for doc in db_manager.list_documents()
            if isinstance(doc, dict) and doc.get("source")
        }

        for disk_doc in _disk_documents():
            file_name = disk_doc["source"]
            if file_name in indexed:
                with _INDEXED_CACHE_LOCK:
                    if file_name not in _INDEXED_CACHE:
                        _INDEXED_CACHE[file_name] = {
                            "source": file_name,
                            "type": disk_doc.get("type", "unknown"),
                            "indexed": True,
                        }
                continue

            file_path = os.path.join(UPLOADS_DIR, file_name)
            try:
                chunks = ingest_file(file_path, file_name)
                if not chunks:
                    print(f"Skipping upload with no readable text: {file_name}")
                    continue
                db_manager.add_documents(chunks)
                indexed.add(file_name)
                _remember_indexed_document(file_name, chunks)
                print(f"Indexed existing upload {file_name} ({len(chunks)} chunks).")
            except Exception as e:
                print(f"Failed to index existing upload {file_name}: {e}")


def _index_existing_uploads_in_background():
    try:
        _index_unindexed_uploads()
    except Exception as e:
        print(f"Failed to scan existing uploads: {e}")


@app.on_event("startup")
def index_existing_uploads():
    # Initialize ChromaDB (or its SQLite fallback) before serving requests so
    # /api/status reports the real backend immediately.
    get_rag_db()
    threading.Thread(
        target=_index_existing_uploads_in_background,
        name="upload-indexer",
        daemon=True,
    ).start()


def _remember_indexed_document(file_name: str, chunks=None):
    """Record index metadata without reading the vector store."""
    meta = {"source": file_name, "indexed": True}
    pages = 0
    slides = 0
    doc_type = None
    for chunk in chunks or []:
        chunk_meta = chunk.get("metadata") or {}
        doc_type = doc_type or chunk_meta.get("type")
        try:
            pages = max(pages, int(chunk_meta.get("total_pages") or 0))
            slides = max(slides, int(chunk_meta.get("slide_count") or chunk_meta.get("slide_number") or 0))
        except (TypeError, ValueError):
            continue
    if doc_type:
        meta["type"] = str(doc_type).lstrip(".").lower()
    if pages:
        meta["total_pages"] = pages
    if slides:
        meta["slide_count"] = slides
    with _INDEXED_CACHE_LOCK:
        _INDEXED_CACHE[file_name] = meta


def _forget_indexed_document(file_name: str):
    with _INDEXED_CACHE_LOCK:
        _INDEXED_CACHE.pop(file_name, None)


def _indexed_snapshot():
    with _INDEXED_CACHE_LOCK:
        return dict(_INDEXED_CACHE)


def _delete_from_index(filename: str):
    try:
        with DOCUMENT_INDEX_LOCK:
            get_rag_db().delete_document(filename)
    except Exception as e:
        print(f"Failed to delete indexed chunks for {filename}: {e}")


@app.get("/api/documents")
def list_documents():
    """
    Lists coursework from the upload folder. This never opens the search
    database, so a busy or still-starting index cannot hide uploaded files.
    """
    indexed_details = _indexed_snapshot()

    documents = []
    seen = set()
    for disk_doc in _disk_documents():
        extra = indexed_details.get(disk_doc["source"], {})
        documents.append({
            **disk_doc,
            **extra,
            "source": disk_doc["source"],
            "type": disk_doc["type"],
            "indexed": disk_doc["source"] in indexed_details,
        })
        seen.add(disk_doc["source"])

    for source, doc in indexed_details.items():
        if source not in seen:
            documents.append({**doc, "indexed": True})

    return {"documents": documents}

@app.delete("/api/documents/{filename:path}")
def delete_document(filename: str):
    """
    Removes a document and all its chunks from the database,
    and deletes its physical uploaded file.
    """
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required.")

    try:
        file_path = os.path.join(UPLOADS_DIR, safe_name)
        with DOCUMENT_INDEX_LOCK:
            if os.path.isfile(file_path):
                os.remove(file_path)
            _forget_indexed_document(safe_name)

        threading.Thread(
            target=_delete_from_index,
            args=(safe_name,),
            name=f"delete-{safe_name}",
            daemon=True,
        ).start()

        return {
            "message": f"Document '{safe_name}' successfully deleted.",
            "filename": safe_name,
            "status": "success"
        }
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/api/observability/stats")
def get_observability_stats():
    """
    Parses the backend/rag_observability.log file and calculates P50/P95 latencies,
    average generation speed, and total logged query parameters.
    """
    import math
    latencies = []
    speeds = []
    token_counts = []
    provider_hits = {}
    
    if os.path.exists(OBSERVABILITY_LOG_PATH):
        try:
            with open(OBSERVABILITY_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        log_data = json.loads(line)
                        metrics = log_data.get("metrics", {})
                        
                        latency = metrics.get("latency_seconds")
                        if latency is not None:
                            latencies.append(float(latency))
                            
                        speed = metrics.get("tokens_per_second")
                        if speed is not None:
                            speeds.append(float(speed))
                            
                        tokens = metrics.get("tokens_generated")
                        if tokens is not None:
                            token_counts.append(int(tokens))
                            
                        provider = metrics.get("provider", "unknown").lower()
                        provider_hits[provider] = provider_hits.get(provider, 0) + 1
                    except Exception:
                        continue
        except Exception as read_err:
            print(f"Error reading observability logs: {read_err}")
            
    # Calculate statistics
    total_queries = len(latencies)
    
    def get_percentile(data: list, pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (len(sorted_data) - 1) * pct
        floor = int(math.floor(idx))
        ceil = int(math.ceil(idx))
        if floor == ceil:
            return sorted_data[floor]
        return sorted_data[floor] * (ceil - idx) + sorted_data[ceil] * (idx - floor)
        
    p50_latency = get_percentile(latencies, 0.50)
    p95_latency = get_percentile(latencies, 0.95)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    
    db_manager = get_rag_db()
    
    # Expose Langfuse tracing active status
    from backend.rag.telemetry import get_telemetry_status
    telemetry_status = get_telemetry_status()
    
    return {
        "total_queries": total_queries,
        "p50_latency_seconds": round(p50_latency, 2),
        "p95_latency_seconds": round(p95_latency, 2),
        "avg_tokens_per_second": round(avg_speed, 2),
        "avg_tokens_generated": round(avg_tokens, 1),
        "providers_breakdown": provider_hits,
        "active_db": db_manager.db_type,
        "tracing_active": telemetry_status["tracing_active"]
    }

@app.post("/api/query", response_model=QueryResponse)
def query_materials(request: QueryRequest):
    """
    Student asks a question. Retrieves relevant context sections
    from the vector database and passes them to the LLM to generate an answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        db_manager = get_rag_db()
        
        # 1. Similarity Search (Hybrid Retrieval + RRF + Cross-Encoder Re-ranking)
        retrieved_chunks = db_manager.query(request.query, n_results=request.n_results)
        
        # 2. Structured Grounded Generation using SLM (JSON formatting, Retries & Benchmarks)
        result = generate_answer(request.query, retrieved_chunks)
        
        # 3. Observability Logging
        try:
            write_observability_log(request.query, retrieved_chunks, result, db_manager.db_type)
        except Exception as log_err:
            print(f"Failed to write query observability log: {log_err}")
            
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            metrics=QueryMetrics(
                latency_seconds=result["metrics"]["latency_seconds"],
                tokens_generated=result["metrics"]["tokens_generated"],
                tokens_per_second=result["metrics"]["tokens_per_second"],
                provider=result["metrics"]["provider"]
            )
        )
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

FRONTEND_DIST = os.path.join(PARENT_DIR, "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/")
    def frontend_index():
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend build not found. Run npm run build in frontend/.")

    @app.get("/favicon.svg")
    def frontend_favicon():
        favicon_path = os.path.join(FRONTEND_DIST, "favicon.svg")
        if os.path.isfile(favicon_path):
            return FileResponse(favicon_path)
        raise HTTPException(status_code=404, detail="Favicon not found.")

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
