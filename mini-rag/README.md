# 🏗️ ConstructIQ — Mini RAG Pipeline

> A production-ready Retrieval-Augmented Generation (RAG) system for a Construction Marketplace AI Assistant.

---
## Live Link :- 

> [website](https://mini-rag-dun.vercel.app/)

> [backend-url](https://mini-rag-backend-5a90.onrender.com)

> **Note** : If the server doesn't respond , please hit the backend url once , as I have used the free-tier of render to host my backend server it get's put down to sleep after 15 minutes of no incoming traffic but once you hit it the server restarts . Thanks ;) .
---
## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation-results)
- [Design Decisions](#design-decisions)

---

## Project Overview

ConstructIQ answers user questions **strictly from internal construction documents** — policies, FAQs, and technical specifications. It does **not** rely on the LLM's general knowledge, preventing hallucinations and ensuring answers are grounded in verified content.

### Features

- ✅ Document chunking with smart boundary detection
- ✅ Semantic embeddings via `sentence-transformers`
- ✅ FAISS vector index (cosine similarity)
- ✅ Grounded LLM answer generation via OpenRouter
- ✅ Full transparency: retrieved chunks shown alongside every answer
- ✅ Performance metrics (retrieval time, generation time, model name)
- ✅ REST API with FastAPI
- ✅ Custom chat UI frontend
- ✅ Quality evaluation script (12 test questions)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                    │
│                                                         │
│  Documents (.txt) ──► Chunker ──► Embedder ──► FAISS   │
│                     (512 chars,   (MiniLM-L6)  Index   │
│                      80 overlap)                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                       │
│                                                         │
│  User Query ──► Embed Query ──► FAISS Search (top-k)   │
│                                       │                 │
│                                       ▼                 │
│                              Retrieved Chunks           │
│                                       │                 │
│                                       ▼                 │
│                            LLM (OpenRouter) ──► Answer  │
└─────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Technology | Reason |
|-----------|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` | Fast (384-dim), high quality on semantic tasks, runs locally |
| Vector DB | FAISS (IndexFlatIP) | No infrastructure needed, cosine similarity via L2-normalized inner product |
| LLM | Mistral-7B via OpenRouter | Free tier, good instruction following, low hallucination rate |
| API | FastAPI | Async, auto OpenAPI docs, Pydantic validation |
| Frontend | Vanilla HTML/CSS/JS | No build toolchain needed, zero dependencies |

---

## Tech Stack

**Backend:**
- Python 3.10+
- `sentence-transformers` — embedding generation
- `faiss-cpu` — vector similarity search
- `FastAPI` + `uvicorn` — REST API
- `requests` — OpenRouter API client

**Frontend:**
- HTML5, CSS3, Vanilla JavaScript
- Zero external dependencies

---

## Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set API key (optional but recommended)

```bash
export OPENROUTER_API_KEY="your-key-here"
```
Get a free key at https://openrouter.ai — the free `mistralai/mistral-7b-instruct:free` model works out of the box.

Without the key, the app runs in **fallback mode** — it still retrieves and shows chunks, but skips LLM generation.

### 3. Start the backend

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On first start, it will:
- Load documents from `/documents/`
- Generate embeddings
- Build and save the FAISS index

### 4. Open the frontend

Open `frontend/index.html` in your browser. That's it.

Or serve it:
```bash
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

### 5. Run evaluation

```bash
cd scripts
python evaluate.py
```

---

## Configuration

All configuration is at the top of `backend/app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | LLM model via OpenRouter |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Character overlap between chunks |

---

## API Reference

### `POST /query`

```json
{
  "query": "What factors affect construction project delays?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "...",
  "retrieved_chunks": [
    {
      "id": "construction_policies.txt_3",
      "text": "...",
      "source": "construction_policies.txt",
      "chunk_index": 3,
      "similarity_score": 0.847
    }
  ],
  "answer": "Based on the documents, construction project delays are caused by...",
  "model": "mistralai/mistral-7b-instruct:free",
  "retrieval_time_seconds": 0.021,
  "generation_time_seconds": 2.3,
  "total_time_seconds": 2.321
}
```

### `GET /health`
Returns pipeline status and stats.

### `GET /stats`
Returns index statistics (chunk count, document list, model info).

### `POST /rebuild-index`
Forces rebuilding the vector index from documents.

---

## Evaluation Results

The evaluation script (`scripts/evaluate.py`) tests 12 questions derived from the documents.

**Metrics:**
- **Retrieval score**: Weighted average of source match (40%) + topic coverage (60%)
- **Answer score**: Topic coverage in generated answer
- **Latency**: End-to-end query time

**Observed findings:**
1. Source matching is ~90%+ — the embedding model reliably identifies the relevant document
2. Topic coverage varies by query specificity — broad questions retrieve slightly less focused chunks
3. Hallucination is minimal — the system prompt strictly restricts the LLM to retrieved context
4. Latency: retrieval ~20-50ms, LLM generation 1-4s (network-dependent)

---

## Design Decisions

### Why `all-MiniLM-L6-v2`?
- 384-dimensional embeddings — small and fast
- Excellent semantic similarity benchmarks (SBERT leaderboard)
- Runs fully locally — no API cost for embeddings
- Downloads automatically via `sentence-transformers`

### Why FAISS over Pinecone/Weaviate?
- Zero infrastructure setup
- Sufficient for thousands to millions of chunks
- `IndexFlatIP` (inner product) with L2 normalization = exact cosine similarity
- Files saved to disk — index survives restarts

### Why overlapping chunks?
- 80-character overlap ensures sentences that span chunk boundaries are still retrievable
- Smart boundary detection: prefers paragraph breaks > sentence ends > character limit

### How is grounding enforced?
The LLM system prompt explicitly states:
> "Only use information explicitly stated in the provided context. If the context does not contain enough information, say: 'Based on the available documents, I cannot fully answer this question.'"
> "Do NOT use general knowledge or make assumptions beyond what is in the context."

This + low temperature (0.1) ensures highly grounded responses.

### Why OpenRouter?
- Free tier available (no billing required for free models)
- Single API key unlocks many open-source models (Mistral, Llama, etc.)
- Easy to swap models via config variable

---

## Adding Your Own Documents

1. Place `.txt` files in the `documents/` directory
2. Call `POST /rebuild-index` with `{"confirm": true}`
3. The system re-embeds and re-indexes automatically

---

## Project Structure

```
mini-rag/
├── backend/
│   ├── app.py              # Main application (RAG pipeline + FastAPI)
│   ├── requirements.txt
│   └── vector_store/       # Auto-created on first run
│       ├── faiss_index.bin
│       └── chunks.json
├── documents/
│   ├── construction_policies.txt
│   ├── platform_faq.txt
│   └── technical_specs.txt
├── frontend/
│   └── index.html          # Single-file chat UI
├── scripts/
│   └── evaluate.py         # Quality evaluation script
├── .env.example            # API key template
├── .gitignore
├── start.bat               # Windows launcher
├── start.sh                # Linux/macOS launcher
└── README.md
```
