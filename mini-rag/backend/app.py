"""
Mini RAG Pipeline - Backend (Deployment-Optimized)
Construction Marketplace AI Assistant

Architecture:
- Document loading and chunking
- Embedding generation (Google Gemini API — lightweight, no local ML)
- FAISS vector index
- LLM answer generation (OpenRouter API)
- FastAPI REST endpoints
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "documents"
INDEX_PATH = Path(__file__).parent / "vector_store" / "faiss_index.bin"
CHUNKS_PATH = Path(__file__).parent / "vector_store" / "chunks.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_FALLBACK_MODELS = [
    "google/gemma-3-4b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
CORS_ORIGINS = [
    "https://constructiq.vercel.app",
    "https://mini-rag-dun.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
]

TOP_K = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
MAX_RETRIES = 3


# ─────────────────────────────────────────────
# Gemini Embedder (lightweight — no ML frameworks)
# ─────────────────────────────────────────────
class GeminiEmbedder:
    """Generates embeddings via Google Gemini REST API.
    No local ML models needed — keeps deployment under 100MB."""

    def __init__(self, api_key: str, model: str = GEMINI_EMBED_MODEL):
        self.api_key = api_key
        self.model = model
        self.dimension = EMBEDDING_DIM
        self._base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text with retry on rate-limit."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Cannot embed queries.")

        for attempt in range(MAX_RETRIES):
            resp = requests.post(
                f"{self._base}:embedContent?key={self.api_key}",
                json={
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_QUERY",
                },
                timeout=15,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"Gemini rate-limited (429). Retrying in {wait}s… (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
            return np.array([values], dtype=np.float32)

        raise RuntimeError("Gemini API rate limit exceeded after retries. Please try again later.")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts in batches of 100 (Gemini API limit)."""
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Cannot generate embeddings.")

        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            reqs = [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
                for t in batch
            ]
            for attempt in range(MAX_RETRIES):
                resp = requests.post(
                    f"{self._base}:batchEmbedContents?key={self.api_key}",
                    json={"requests": reqs},
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Gemini rate-limited (429) on batch. Retrying in {wait}s… (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                embeddings = [e["values"] for e in resp.json()["embeddings"]]
                all_embeddings.extend(embeddings)
                logger.info(f"  Embedded batch {i // batch_size + 1} ({len(batch)} texts)")
                break
            else:
                raise RuntimeError(f"Gemini API rate limit exceeded on batch {i // batch_size + 1} after retries.")

        return np.array(all_embeddings, dtype=np.float32)


# ─────────────────────────────────────────────
# Document Chunker
# ─────────────────────────────────────────────
class DocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_and_chunk(self, docs_dir: Path) -> List[Dict]:
        """Load all .txt files from directory and chunk them."""
        all_chunks = []
        txt_files = sorted(docs_dir.glob("*.txt"))

        if not txt_files:
            logger.warning(f"No .txt files found in {docs_dir}")
            return []

        for filepath in txt_files:
            logger.info(f"Processing: {filepath.name}")
            text = filepath.read_text(encoding="utf-8")
            chunks = self._chunk_text(text, filepath.name)
            all_chunks.extend(chunks)
            logger.info(f"  → {len(chunks)} chunks from {filepath.name}")

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        chunks = []
        text = text.strip()
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break
                else:
                    sent_break = max(
                        text.rfind(". ", start, end),
                        text.rfind(".\n", start, end),
                    )
                    if sent_break > start + self.chunk_size // 2:
                        end = sent_break + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": f"{source}_{chunk_id}",
                    "text": chunk_text,
                    "source": source,
                    "chunk_index": chunk_id,
                    "char_start": start,
                    "char_end": end,
                })
                chunk_id += 1

            start = max(start + 1, end - self.overlap)

        return chunks


# ─────────────────────────────────────────────
# Vector Store (FAISS)
# ─────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.embedder = GeminiEmbedder(api_key=GEMINI_API_KEY)
        self.dimension = EMBEDDING_DIM
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        """Generate embeddings via Gemini API and build FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        logger.info(f"Generating embeddings for {len(chunks)} chunks via Gemini API...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"FAISS index built: {self.index.ntotal} vectors (dim={self.dimension})")

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Persist index and chunks to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved index → {index_path}")
        logger.info(f"Saved chunks → {chunks_path}")

    def load(self, index_path: Path, chunks_path: Path) -> bool:
        """Load pre-built index from disk with dimension safety check."""
        if not index_path.exists() or not chunks_path.exists():
            return False

        loaded_index = faiss.read_index(str(index_path))

        # Safety: reject index built with a different embedding model
        if loaded_index.d != self.dimension:
            logger.warning(
                f"Index dimension mismatch: file has {loaded_index.d}, "
                f"expected {self.dimension}. Rebuild required."
            )
            return False

        self.index = loaded_index
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        return True

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")

        query_embedding = self.embedder.embed_query(query)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results


# ─────────────────────────────────────────────
# LLM Client (OpenRouter)
# ─────────────────────────────────────────────
class LLMClient:
    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generate a grounded answer from retrieved context."""
        context_text = self._format_context(context_chunks)

        system_prompt = """You are an AI assistant for a construction marketplace platform.
Your job is to answer user questions STRICTLY based on the provided document context.

RULES:
1. Only use information explicitly stated in the provided context.
2. If the context does not contain enough information, say: "Based on the available documents, I cannot fully answer this question."
3. Do NOT use general knowledge or make assumptions beyond what is in the context.
4. Be concise, clear, and professional.
5. When citing information, you may reference the source document name.
6. Structure your answer clearly with paragraphs or bullet points as appropriate."""

        user_prompt = f"""CONTEXT FROM DOCUMENTS:
{context_text}

USER QUESTION: {query}

Please answer the question strictly based on the context above."""

        if not self.api_key:
            return self._fallback_response(query, context_chunks)

        models_to_try = [self.model] + OPENROUTER_FALLBACK_MODELS
        last_error = ""

        for model_id in models_to_try:
            try:
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://constructiq.vercel.app",
                        "X-Title": "ConstructIQ - Construction Marketplace",
                    },
                    json={
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 800,
                    },
                    timeout=30,
                )
                latency = time.time() - start_time

                if response.status_code == 404:
                    logger.warning(f"Model {model_id} returned 404, trying next...")
                    last_error = f"Model {model_id} unavailable (404)"
                    continue

                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()

                if model_id != self.model:
                    logger.info(f"Used fallback model: {model_id}")

                return {
                    "answer": answer,
                    "model": model_id,
                    "latency_seconds": round(latency, 2),
                    "token_usage": data.get("usage", {}),
                }
            except requests.exceptions.Timeout:
                logger.error(f"LLM request timed out for model {model_id}")
                last_error = "Request timed out."
                continue
            except Exception as e:
                logger.error(f"LLM error with model {model_id}: {e}")
                last_error = str(e)
                continue

        return self._fallback_response(query, context_chunks, error=last_error)

    def _format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i} | Source: {chunk['source']} | Relevance: {chunk['similarity_score']:.3f}]\n"
                f"{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _fallback_response(self, query: str, chunks: List[Dict], error: str = "") -> Dict:
        """Return a structured fallback when LLM is unavailable."""
        notice = ""
        if error:
            notice = f"\n\n⚠️ Note: LLM unavailable ({error}). Showing raw retrieved context below."
        elif not self.api_key:
            notice = "\n\n⚠️ Note: No OPENROUTER_API_KEY set. Showing raw retrieved context. Set API key for AI-generated answers."

        top_chunk = chunks[0]["text"] if len(chunks) > 0 else "No relevant information found."
        answer = f"Based on the documents, here is the most relevant information regarding your query:\n\n{top_chunk}{notice}"
        return {
            "answer": answer,
            "model": "fallback (no LLM)",
            "latency_seconds": 0,
            "token_usage": {},
        }


# ─────────────────────────────────────────────
# RAG Pipeline (orchestrator)
# ─────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()
        self.llm = LLMClient(api_key=OPENROUTER_API_KEY)
        self._initialized = False

    def initialize(self):
        """Load or build the vector index."""
        if self.vector_store.load(INDEX_PATH, CHUNKS_PATH):
            logger.info("✅ Loaded existing vector index.")
            self._initialized = True
        else:
            if not GEMINI_API_KEY:
                logger.error(
                    "⚠️ No pre-built index found (or dimension mismatch) and GEMINI_API_KEY is not set. "
                    "Cannot build index. Set GEMINI_API_KEY and call POST /rebuild-index."
                )
                return
            logger.info("Building new vector index from documents...")
            chunks = self.chunker.load_and_chunk(DOCS_DIR)
            if not chunks:
                raise RuntimeError(f"No documents found in {DOCS_DIR}")
            self.vector_store.build_index(chunks)
            self.vector_store.save(INDEX_PATH, CHUNKS_PATH)
            logger.info("✅ Index built and saved.")
            self._initialized = True

    def query(self, user_query: str, top_k: int = TOP_K) -> Dict:
        """Full RAG pipeline: retrieve + generate."""
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        retrieved_chunks = self.vector_store.search(user_query, top_k)
        retrieval_time = time.time() - t0

        if not retrieved_chunks:
            return {
                "query": user_query,
                "retrieved_chunks": [],
                "answer": "I could not find relevant information in the documents for your query.",
                "model": "N/A",
                "retrieval_time_seconds": round(retrieval_time, 3),
                "generation_time_seconds": 0,
                "total_time_seconds": round(retrieval_time, 3),
            }

        t1 = time.time()
        llm_result = self.llm.generate(user_query, retrieved_chunks)
        generation_time = time.time() - t1

        return {
            "query": user_query,
            "retrieved_chunks": retrieved_chunks,
            "answer": llm_result["answer"],
            "model": llm_result["model"],
            "token_usage": llm_result.get("token_usage", {}),
            "retrieval_time_seconds": round(retrieval_time, 3),
            "generation_time_seconds": round(generation_time, 3),
            "total_time_seconds": round(retrieval_time + generation_time, 3),
        }

    def get_stats(self) -> Dict:
        """Return pipeline statistics."""
        return {
            "total_chunks": len(self.vector_store.chunks),
            "index_size": self.vector_store.index.ntotal if self.vector_store.index else 0,
            "embedding_model": f"Gemini {GEMINI_EMBED_MODEL}",
            "embedding_dim": EMBEDDING_DIM,
            "llm_model": OPENROUTER_MODEL,
            "documents": list(set(c["source"] for c in self.vector_store.chunks)),
        }


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────
rag = RAGPipeline()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Modern lifespan handler."""
    logger.info("🚀 Starting Mini RAG API...")
    try:
        rag.initialize()
        if rag._initialized:
            logger.info("✅ RAG pipeline ready.")
        else:
            logger.warning("⚠️ RAG pipeline started WITHOUT index. Set GEMINI_API_KEY and POST /rebuild-index.")
    except Exception as e:
        logger.error(f"⚠️ Initialization failed: {e}. Server running but queries will fail.")
    yield
    logger.info("🛑 Shutting down Mini RAG API.")


app = FastAPI(
    title="Mini RAG API",
    description="Construction Marketplace AI Assistant - RAG Pipeline",
    version="2.1.0",
    lifespan=lifespan,
)

# ── Rate Limiter ──
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a moment and try again."},
    )

# ── CORS — restricted to known origins ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Secret"],
)


def require_admin(request: Request):
    """Verify admin secret header for protected endpoints."""
    if not ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret not configured on server.")
    if request.headers.get("X-Admin-Secret") != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: invalid admin credentials.")


class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RebuildRequest(BaseModel):
    confirm: bool = False


@app.get("/")
def root():
    return {"status": "ok", "message": "Mini RAG API is running. Visit /docs for API reference."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "initialized": rag._initialized,
        "stats": rag.get_stats() if rag._initialized else {},
    }


@app.post("/query")
@limiter.limit("10/minute")
def query_endpoint(request: Request, req: QueryRequest):
    """Main RAG query endpoint. Rate-limited to 10 req/min per IP."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(req.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters).")
    if req.top_k < 1 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")

    try:
        result = rag.query(req.query, top_k=req.top_k)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@app.get("/stats")
def stats_endpoint():
    """Get pipeline statistics."""
    if not rag._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")
    return rag.get_stats()


@app.post("/rebuild-index")
@limiter.limit("1/hour")
def rebuild_index(request: Request, req: RebuildRequest):
    """Force rebuild of the vector index. Admin-only, rate-limited to 1/hour."""
    require_admin(request)
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to rebuild the index.")
    try:
        chunks = rag.chunker.load_and_chunk(DOCS_DIR)
        if not chunks:
            raise HTTPException(status_code=404, detail="No documents found.")
        rag.vector_store.build_index(chunks)
        rag.vector_store.save(INDEX_PATH, CHUNKS_PATH)
        rag._initialized = True
        return {"status": "rebuilt", "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Index rebuild failed. Check server logs.")


@app.get("/chunks")
def list_chunks(request: Request, source: Optional[str] = None, limit: int = 50):
    """List indexed chunks (admin-only, for debugging)."""
    require_admin(request)
    if not rag._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    chunks = rag.vector_store.chunks
    if source:
        chunks = [c for c in chunks if c["source"] == source]
    return {
        "total": len(chunks),
        "shown": min(limit, len(chunks)),
        "chunks": chunks[:limit],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
