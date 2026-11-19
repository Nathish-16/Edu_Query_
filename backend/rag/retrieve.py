import os
import json
import sqlite3
import re
import math
import hashlib
import threading
from collections import Counter
import numpy as np
from typing import List, Dict, Any
from backend.rag.telemetry import observe

# Configure local Hugging Face model cache if present in project
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL_CACHE = os.path.join(_BASE_DIR, ".model_cache", "huggingface")
if os.path.isdir(_LOCAL_MODEL_CACHE) and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = _LOCAL_MODEL_CACHE

def _find_local_model_path(model_repo_name: str) -> str:
    """Finds the local snapshot directory for a given model repo if cached in .model_cache."""
    safe_name = "models--" + model_repo_name.replace("/", "--")
    hub_dir = os.path.join(_LOCAL_MODEL_CACHE, "hub", safe_name, "snapshots")
    if os.path.isdir(hub_dir):
        for entry in os.listdir(hub_dir):
            full_path = os.path.join(hub_dir, entry)
            if os.path.isdir(full_path):
                return full_path
    return None

# ---------------------------------------------------------------------------
# Local embedding / re-ranking models
#
# The previous implementation imported ``sentence_transformers``. On Python
# 3.13 that package pulls in ``datasets`` -> ``pyarrow`` and can terminate the
# whole server with a native access violation while importing. To keep the API
# stable we load the exact same local model snapshots with ``transformers``
# directly (no ``datasets``/``sklearn`` import), and fall back to deterministic
# hashing embeddings when the model files are unavailable.
# ---------------------------------------------------------------------------

_embedding_model = None
_embedding_model_ready = False
_model_lock = threading.Lock()

_LOCAL_EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_LOCAL_CROSS_ENCODER_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _TransformersEmbedder:
    """Small mean-pooling wrapper around a local MiniLM checkpoint."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                output = self.model(**encoded)

            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().numpy().astype("float32").tolist())
        return vectors


def get_embedding_model():
    """
    Lazy-loads and caches a local MiniLM embedder.

    Uses the cached Hugging Face snapshot when present. No network access and
    no ``sentence_transformers`` import. Falls back to deterministic embeddings
    if the model cannot be loaded so ingestion/querying never crashes.
    """
    global _embedding_model, _embedding_model_ready
    with _model_lock:
        if _embedding_model_ready:
            return _embedding_model
        _embedding_model_ready = True

        if os.getenv("EDUQUERY_DISABLE_LOCAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            print("Local embeddings disabled by configuration; using hashing embeddings.")
            return None

        local_path = _find_local_model_path(_LOCAL_EMBEDDING_REPO)
        if local_path and os.path.isdir(local_path):
            try:
                print(f"Loading local MiniLM embedder from snapshot: {local_path}")
                _embedding_model = _TransformersEmbedder(local_path)
                print("Local MiniLM embedder loaded successfully.")
                return _embedding_model
            except Exception as local_err:
                print(f"Local embedder load failed, using hashing fallback: {local_err}")

        print("Local embedding snapshot unavailable; using deterministic hashing embeddings.")
        return None


def _fallback_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Lightweight deterministic text embedding fallback (TF-IDF / character n-gram hashing)
    when neural embedding model cannot be loaded.
    Ensures vector databases never crash and can always store and search documents.
    """
    words = re.findall(r'\w+', text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    if not words:
        return vec.tolist()
    for w in words:
        h = int(hashlib.md5(w.encode('utf-8')).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using SentenceTransformer or fallback.
    """
    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode(texts)
            return [list(vector) for vector in embeddings]
        except Exception as e:
            print(f"Error generating embeddings via model: {e}")
    # Deterministic fallback embedding
    return [_fallback_embedding(t) for t in texts]


# Caching the local Cross-Encoder model for re-ranking
_cross_encoder = None
_cross_encoder_ready = False
_ce_lock = threading.Lock()


class _TransformersCrossEncoder:
    """Local MS MARCO cross-encoder without importing sentence_transformers."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        self.model.eval()

    def predict(self, pairs: List[tuple], batch_size: int = 16) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
            scores.extend(logits.reshape(-1).cpu().numpy().astype("float32").tolist())
        return scores


def get_cross_encoder():
    """Loads the cached local cross-encoder, or returns None to skip re-ranking."""
    global _cross_encoder, _cross_encoder_ready
    with _ce_lock:
        if _cross_encoder_ready:
            return _cross_encoder
        _cross_encoder_ready = True

        if os.getenv("EDUQUERY_DISABLE_RERANKING", "").lower() in {"1", "true", "yes"}:
            print("Cross-encoder re-ranking disabled by configuration.")
            return None

        local_path = _find_local_model_path(_LOCAL_CROSS_ENCODER_REPO)
        if local_path and os.path.isdir(local_path):
            try:
                print(f"Loading local cross-encoder from snapshot: {local_path}")
                _cross_encoder = _TransformersCrossEncoder(local_path)
                print("Local cross-encoder loaded successfully.")
                return _cross_encoder
            except Exception as local_err:
                print(f"Local cross-encoder load failed, skipping re-ranking: {local_err}")

        print("Local cross-encoder snapshot unavailable; using fused ranking only.")
        return None


class BM25Retriever:
    """
    A pure-Python high-performance Okapi BM25 keyword search engine.
    Precalculates TF-IDF variables locally, providing zero compile dependencies
    and sub-millisecond keyword lookup.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.corpus = []
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
        
    def fit(self, chunks: List[Dict[str, Any]]):
        self.corpus = chunks
        if not chunks:
            self.doc_len = []
            self.avg_doc_len = 0.0
            self.doc_freqs = []
            self.idf = {}
            return
            
        tokenized_docs = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = sum(self.doc_len) / len(chunks)
        
        self.doc_freqs = []
        df = Counter()
        
        for doc in tokenized_docs:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for token in frequencies.keys():
                df[token] += 1
                
        # Precompute standard Okapi BM25 IDF scores with smoothing
        N = len(chunks)
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def query(self, query_text: str, top_n: int = 15) -> List[Dict[str, Any]]:
        if not self.corpus:
            return []
            
        q_tokens = self._tokenize(query_text)
        scores = []
        
        for doc_idx, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_l = self.doc_len[doc_idx]
            
            for token in q_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0.0)
                    
                    # BM25 Term Frequency adjustment
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avg_doc_len))
                    
                    score += idf * (numerator / denominator)
                    
            scores.append((score, self.corpus[doc_idx]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, chunk) in enumerate(scores[:top_n], start=1):
            # Create a deep copy of the chunk dictionary to prevent mutations
            chunk_copy = json.loads(json.dumps(chunk))
            chunk_copy["bm25_score"] = float(score)
            chunk_copy["similarity"] = float(score)  # Backend API compliance
            results.append(chunk_copy)
            
        return results


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fuses two lists of ranked chunks using the standard Reciprocal Rank Fusion (RRF) algorithm.
    """
    scores = {}
    
    # Process vector search ranks
    for rank, res in enumerate(vector_results, start=1):
        doc_text = res["text"]
        if doc_text not in scores:
            scores[doc_text] = {"chunk": res, "vector_rank": rank, "bm25_rank": None, "score": 0.0}
        scores[doc_text]["score"] += 1.0 / (k + rank)
        scores[doc_text]["vector_rank"] = rank
        
    # Process BM25 keywords ranks
    for rank, res in enumerate(bm25_results, start=1):
        doc_text = res["text"]
        if doc_text not in scores:
            scores[doc_text] = {"chunk": res, "vector_rank": None, "bm25_rank": rank, "score": 0.0}
        scores[doc_text]["score"] += 1.0 / (k + rank)
        scores[doc_text]["bm25_rank"] = rank
        
    # Sort candidates by combined reciprocal score descending
    fused_candidates = list(scores.values())
    fused_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    results = []
    for item in fused_candidates:
        chunk = json.loads(json.dumps(item["chunk"]))
        # Store ranks & fusion score for auditing
        chunk["rrf_score"] = float(item["score"])
        chunk["vector_rank"] = item["vector_rank"]
        chunk["bm25_rank"] = item["bm25_rank"]
        results.append(chunk)
        
    return results


class SQLiteVectorStore:
    """
    A pure-Python high-speed SQLite vector store that acts as a robust
    database fallback. It uses sqlite3 to store content & metadata,
    and computes cosine similarity using NumPy for sub-millisecond retrieval.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT,
                text TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        conn.commit()
        conn.close()
        
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{chunk['metadata']['source']}_{chunk['metadata'].get('page_number', chunk['metadata'].get('slide_number', idx))}_{idx}"
            source = chunk['metadata']['source']
            text = chunk['text']
            metadata_str = json.dumps(chunk['metadata'])
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            
            cursor.execute(
                "INSERT OR REPLACE INTO chunks (id, source, text, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, source, text, metadata_str, emb_bytes)
            )
            
        conn.commit()
        conn.close()
        
    def query(self, query_vector: List[float], n_results: int = 15) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, metadata, embedding FROM chunks")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        
        results = []
        for chunk_id, text, metadata_str, emb_bytes in rows:
            emb_vec = np.frombuffer(emb_bytes, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_vec)
            
            if q_norm > 0 and emb_norm > 0:
                similarity = float(np.dot(q_vec, emb_vec) / (q_norm * emb_norm))
            else:
                similarity = 0.0
                
            results.append({
                "text": text,
                "metadata": json.loads(metadata_str),
                "similarity": similarity
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:n_results]
        
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT text, metadata FROM chunks")
        rows = cursor.fetchall()
        conn.close()
        
        return [{"text": text, "metadata": json.loads(meta)} for text, meta in rows]
        
    def delete_document(self, filename: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE source = ?", (filename,))
        conn.commit()
        conn.close()
        
    def list_documents(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source, metadata FROM chunks")
        rows = cursor.fetchall()
        conn.close()
        
        docs = {}
        for source, metadata_str in rows:
            meta = json.loads(metadata_str)
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "type": meta.get("type", "unknown"),
                    "total_pages": int(meta.get("total_pages") or 0),
                    "slide_count": int(meta.get("slide_count") or meta.get("slide_number") or 0)
                }
            else:
                if "total_pages" in meta:
                    docs[source]["total_pages"] = max(docs[source]["total_pages"], int(meta.get("total_pages") or 0))
                if "slide_count" in meta or "slide_number" in meta:
                    slides = int(meta.get("slide_count") or meta.get("slide_number") or 0)
                    docs[source]["slide_count"] = max(docs[source]["slide_count"], slides)
                    
        return list(docs.values())


class ChromaDBStore:
    """
    Adapter for ChromaDB. Thread-safe integration with Chroma's vector API.
    Uses upsert and thread synchronization to guarantee reliable concurrent access on Windows.
    """
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.lock = threading.RLock()
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="educational_rag_collection")
        
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        with self.lock:
            ids = []
            texts = []
            metadatas = []
            
            for idx, chunk in enumerate(chunks):
                source = chunk.get("metadata", {}).get("source", "unknown")
                page_slide = chunk.get("metadata", {}).get("page_number", chunk.get("metadata", {}).get("slide_number", idx))
                chunk_id = f"{source}_{page_slide}_{idx}_{abs(hash(chunk['text'][:40]))}"
                ids.append(chunk_id)
                texts.append(chunk["text"])
                
                clean_meta = {}
                for k, v in chunk.get("metadata", {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                metadatas.append(clean_meta)
                
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
    def query(self, query_vector: List[float], n_results: int = 15) -> List[Dict[str, Any]]:
        with self.lock:
            try:
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=n_results
                )
            except Exception as e:
                print(f"Chroma query error: {e}")
                return []
            
            formatted = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                
                for doc, meta, dist in zip(docs, metas, distances):
                    formatted.append({
                        "text": doc,
                        "metadata": meta or {},
                        "similarity": float(1.0 / (1.0 + max(0.0, dist)))
                    })
            return formatted
        
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        with self.lock:
            try:
                results = self.collection.get(include=["documents", "metadatas"])
                docs = results.get("documents", []) if results else []
                metas = results.get("metadatas", []) if results else []
                return [{"text": doc, "metadata": meta or {}} for doc, meta in zip(docs, metas)]
            except Exception as e:
                print(f"Chroma get_all_chunks error: {e}")
                return []
        
    def delete_document(self, filename: str):
        with self.lock:
            try:
                self.collection.delete(where={"source": filename})
            except Exception as e:
                print(f"Chroma delete error for {filename}: {e}")
        
    def list_documents(self) -> List[Dict[str, Any]]:
        with self.lock:
            try:
                results = self.collection.get(include=["metadatas"]) or {}
            except Exception as list_err:
                print(f"Chroma metadata listing failed, retrying default get(): {list_err}")
                try:
                    results = self.collection.get() or {}
                except Exception as retry_err:
                    print(f"Chroma document listing failed: {retry_err}")
                    return []
            metadatas = results.get("metadatas") or []
            
            docs = {}
            for meta in metadatas:
                if not meta or not isinstance(meta, dict):
                    continue
                source = meta.get("source")
                if not source:
                    continue
                if source not in docs:
                    docs[source] = {
                        "source": source,
                        "type": meta.get("type", "unknown"),
                        "total_pages": int(meta.get("total_pages") or 0),
                        "slide_count": int(meta.get("slide_count") or meta.get("slide_number") or 0)
                    }
                else:
                    if "total_pages" in meta:
                        docs[source]["total_pages"] = max(docs[source]["total_pages"], int(meta.get("total_pages") or 0))
                    if "slide_count" in meta or "slide_number" in meta:
                        slides = int(meta.get("slide_count") or meta.get("slide_number") or 0)
                        docs[source]["slide_count"] = max(docs[source]["slide_count"], slides)
                        
            return list(docs.values())


class RAGDatabaseManager:
    """
    Unified manager orchestrating Vector Retrieval, BM25 Keyword Search,
    RRF Fusion, and Hugging Face Cross-Encoder Re-ranking.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vectordb_dir = os.path.join(self.base_dir, "vectordb")
        os.makedirs(self.vectordb_dir, exist_ok=True)
        
        self.db = None
        self.db_type = "chromadb"
        
        # Load Vector Store
        try:
            print("Initializing ChromaDB Store...")
            chroma_dir = os.path.join(self.vectordb_dir, "chroma")
            os.makedirs(chroma_dir, exist_ok=True)
            self.db = ChromaDBStore(persist_dir=chroma_dir)
            print("ChromaDB Store initialized successfully.")
        except Exception as chroma_err:
            print(f"ChromaDB initialization failed, falling back to SQLite: {chroma_err}")
            self.db_type = "sqlite"
            sqlite_path = os.path.join(self.vectordb_dir, "vector_store.db")
            self.db = SQLiteVectorStore(db_path=sqlite_path)
            print(f"SQLite vector fallback store active at: {sqlite_path}")
            
        # Initialize BM25 Keyword Retriever
        self.bm25_retriever = BM25Retriever()
        self.rebuild_bm25()
        
    def rebuild_bm25(self):
        """
        Retrieves all documents currently stored in the active database
        and rebuilds the Okapi BM25 search indices.
        """
        try:
            print("Rebuilding BM25 keyword indexes...")
            all_chunks = self.db.get_all_chunks()
            self.bm25_retriever.fit(all_chunks)
            print(f"BM25 index successfully updated with {len(all_chunks)} chunks.")
        except Exception as e:
            print(f"Failed to rebuild BM25 index: {e}")
            
    def add_documents(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(texts)
        sources = {c["metadata"]["source"] for c in chunks}
        for source in sources:
            self.db.delete_document(source)
        self.db.add_documents(chunks, embeddings)
        self.rebuild_bm25()
        
    @observe(name="retrieve_chunks")
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid Retrieval (BM25 + Vector Search) + Reciprocal Rank Fusion (RRF)
        + Cross-Encoder Re-ranking pipeline.
        """
        # 1. Vector Search Retrieval (fetch top 15)
        query_vector = get_embeddings([query_text])[0]
        vector_results = self.db.query(query_vector, n_results=15)
        
        # 2. BM25 Keyword Retrieval (fetch top 15)
        bm25_results = self.bm25_retriever.query(query_text, top_n=15)
        
        if not vector_results and not bm25_results:
            return []
            
        # 3. Reciprocal Rank Fusion (RRF)
        fused_candidates = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        
        # Select top 15 fused candidates for Re-ranking
        candidates_to_rerank = fused_candidates[:15]
        
        # 4. Cross-Encoder Re-ranking
        cross_encoder = get_cross_encoder()
        if cross_encoder is None:
            # Fallback directly to fused ranking if cross-encoder failed to load
            print("Re-ranking skipped (using RRF scores)...")
            for c in candidates_to_rerank:
                c["similarity"] = c["rrf_score"]
            return candidates_to_rerank[:n_results]
            
        try:
            print(f"Executing Cross-Encoder re-ranking on {len(candidates_to_rerank)} candidates...")
            pairs = [(query_text, c["text"]) for c in candidates_to_rerank]
            scores = cross_encoder.predict(pairs)
            
            for chunk, score in zip(candidates_to_rerank, scores):
                chunk["cross_encoder_score"] = float(score)
                sigmoid_similarity = 1.0 / (1.0 + math.exp(-score))
                chunk["similarity"] = sigmoid_similarity
                
            candidates_to_rerank.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            print("Re-ranking complete. Most relevant chunks promoted to top.")
            return candidates_to_rerank[:n_results]
        except Exception as e:
            print(f"Error during Cross-Encoder re-ranking: {e}")
            for c in candidates_to_rerank:
                c["similarity"] = c["rrf_score"]
            return candidates_to_rerank[:n_results]
            
    def delete_document(self, filename: str):
        self.db.delete_document(filename)
        self.rebuild_bm25()
        
    def list_documents(self) -> List[Dict[str, Any]]:
        return self.db.list_documents()
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "active_db": self.db_type,
            "document_count": len(self.list_documents()),
            "bm25_active": len(self.bm25_retriever.corpus) > 0,
            "status": "healthy"
        }

# Singleton instance
rag_db = None
rag_db_lock = threading.Lock()

def get_rag_db():
    global rag_db
    with rag_db_lock:
        if rag_db is None:
            rag_db = RAGDatabaseManager()
    return rag_db
