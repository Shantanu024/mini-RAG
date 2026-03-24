"""
Mini RAG Pipeline - Backend
Construction Marketplace AI Assistant

Architecture:
- Document loading and chunking
- Embedding generation (sentence-transformers)
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests

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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, lightweight, great quality
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
TOP_K = 5          # Number of chunks to retrieve
CHUNK_SIZE = 512   # Characters per chunk
CHUNK_OVERLAP = 80 # Character overlap between chunks


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
        txt_files = list(docs_dir.glob("*.txt"))

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

            # Try to end at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break first
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break
                else:
                    # Fall back to sentence end
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
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        """Generate embeddings and build FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Build inner product index (equivalent to cosine similarity after L2 norm)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={self.dimension})")

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Persist index and chunks to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved index to {index_path}")
        logger.info(f"Saved chunks to {chunks_path}")

    def load(self, index_path: Path, chunks_path: Path) -> bool:
        """Load pre-built index from disk."""
        if not index_path.exists() or not chunks_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        return True

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Retrieve top-k most relevant chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded.")

        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype=np.float32)
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

        try:
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "Mini RAG - Construction Marketplace",
                },
                json={
                    "model": self.model,
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
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()

            return {
                "answer": answer,
                "model": self.model,
                "latency_seconds": round(latency, 2),
                "token_usage": data.get("usage", {}),
            }
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return self._fallback_response(query, context_chunks, error="Request timed out. Using context summary.")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._fallback_response(query, context_chunks, error=str(e))

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

        top_chunk = chunks[0]["text"] if chunks else "No relevant information found."
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
        else:
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

        # Step 1: Retrieve relevant chunks
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

        # Step 2: Generate answer
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
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": OPENROUTER_MODEL,
            "documents": list(set(c["source"] for c in self.vector_store.chunks)),
        }


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────
rag = RAGPipeline()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Modern lifespan handler (replaces deprecated on_event)."""
    logger.info("🚀 Starting Mini RAG API...")
    rag.initialize()
    logger.info("✅ RAG pipeline ready.")
    yield
    logger.info("🛑 Shutting down Mini RAG API.")


app = FastAPI(
    title="Mini RAG API",
    description="Construction Marketplace AI Assistant - RAG Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def query_endpoint(req: QueryRequest):
    """Main RAG query endpoint."""
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
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/stats")
def stats_endpoint():
    """Get pipeline statistics."""
    if not rag._initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")
    return rag.get_stats()


@app.post("/rebuild-index")
def rebuild_index(req: RebuildRequest):
    """Force rebuild of the vector index from documents."""
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
    except Exception as e:
        logger.error(f"Rebuild error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
def list_chunks(source: Optional[str] = None, limit: int = 50):
    """List indexed chunks (for debugging/transparency)."""
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
