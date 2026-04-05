# 🏗️ ConstructIQ — Mini RAG Pipeline

> A production-ready Retrieval-Augmented Generation (RAG) system for a Construction Marketplace AI Assistant.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Design Decisions](#design-decisions)

---

## Project Overview

ConstructIQ answers user questions **strictly from internal construction documents** — policies, FAQs, and technical specifications. It does **not** rely on the LLM's general knowledge, preventing hallucinations and ensuring answers are grounded in verified content.

### Features

- Document chunking with smart boundary detection
- Semantic embeddings via Google Gemini API (`gemini-embedding-001`, 3072-dim)
- FAISS vector index (cosine similarity)
- Grounded LLM answer generation via OpenRouter (with model fallbacks)
- Full transparency: retrieved chunks shown alongside every answer
- Performance metrics (retrieval time, generation time, model name)
- REST API with FastAPI + auto-generated OpenAPI docs
- Custom chat UI frontend (zero dependencies)
- Quality evaluation script (12 test questions)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                     │
│                                                         │
│  Documents (.txt) ──► Chunker ──► Embedder ──► FAISS   │
│                     (512 chars,   (Gemini     Index     │
│                      80 overlap)  API 3072d)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                        │
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
| Embeddings | Gemini `gemini-embedding-001` | 3072-dim, high quality, free API tier, no local ML frameworks needed |
| Vector DB | FAISS (`IndexFlatIP`) | Zero infrastructure, cosine similarity via L2-normalized inner product |
| LLM | Llama 3.3 70B via OpenRouter | Free tier, excellent instruction following, low hallucination rate |
| API | FastAPI | Async, auto OpenAPI docs, Pydantic validation |
| Frontend | Vanilla HTML/CSS/JS | No build toolchain, zero dependencies, instant deployment |

---

## Tech Stack

**Backend:**
- Python 3.10+
- `faiss-cpu` — vector similarity search
- `FastAPI` + `uvicorn` — REST API
- `requests` — Gemini & OpenRouter API client
- `numpy` — embedding array handling
- `python-dotenv` — environment variable loading

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

### 2. Set API keys

```bash
# Required — for embeddings
export GEMINI_API_KEY="your-gemini-key"

# Optional — for AI-generated answers (without this, raw chunks are shown)
export OPENROUTER_API_KEY="your-openrouter-key"
```

Get free keys at:
- **Gemini**: https://aistudio.google.com/app/apikey
- **OpenRouter**: https://openrouter.ai

Or copy `.env.example` to `.env` and fill in your keys.

### 3. Start the backend

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On first start, it will:
- Load documents from `/documents/`
- Generate embeddings via Gemini API
- Build and save the FAISS index

Subsequent starts load the pre-built index instantly.

### 4. Open the frontend

Open `frontend/index.html` in your browser. That's it.

Or serve it:
```bash
cd frontend
python -m http.server 3000
# then visit http://localhost:3000
```

### 5. Run evaluation (optional)

```bash
cd scripts
python evaluate.py
```

---

## Configuration

All configuration is at the top of `backend/app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(env var)* | **Required.** Google Gemini API key for embeddings |
| `OPENROUTER_API_KEY` | *(env var)* | Optional. OpenRouter API key for LLM answers |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `EMBEDDING_DIM` | `3072` | Embedding dimension (must match model) |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Primary LLM model |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `80` | Character overlap between chunks |

---

## API Reference

### `GET /`
Returns API status.

### `GET /health`
Returns pipeline health and statistics.

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
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "retrieval_time_seconds": 0.021,
  "generation_time_seconds": 2.3,
  "total_time_seconds": 2.321
}
```

### `GET /stats`
Returns index statistics (chunk count, document list, model info).

### `POST /rebuild-index`
Force rebuild the vector index from documents. Send `{"confirm": true}`.

### `GET /chunks`
List indexed chunks. Optional `?source=filename.txt&limit=50` query params.

### `GET /docs`
Auto-generated interactive API documentation (Swagger UI).

---

## Evaluation

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

## Deployment

### Backend — Render

| Setting | Value |
|---------|-------|
| **Root Directory** | `backend` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

**Environment Variables:**

| Variable | Required | Value |
|----------|----------|-------|
| `GEMINI_API_KEY` | ✅ Yes | Your Gemini API key |
| `OPENROUTER_API_KEY` | ⬜ Optional | Your OpenRouter API key |
| `PYTHON_VERSION` | ⬜ Optional | `3.11.0` (if needed) |

> **Note:** The pre-built FAISS index (`backend/vector_store/`) is committed to git. Render loads it on startup — no rebuild needed unless you change documents.

### Frontend — Vercel

| Setting | Value |
|---------|-------|
| **Root Directory** | `frontend` |
| **Framework Preset** | Other |
| **Build Command** | *(leave empty)* |
| **Output Directory** | `.` |

**No environment variables needed.** The backend URL is set in `index.html` line 433 (`DEPLOYED_API_URL`).

> **Important:** Before deploying, update `DEPLOYED_API_URL` in `frontend/index.html` to your Render backend URL (e.g., `https://your-app.onrender.com`).

---

## Adding Your Own Documents

1. Place `.txt` files in the `documents/` directory
2. Start the backend, then call `POST /rebuild-index` with `{"confirm": true}`
3. The system re-embeds and re-indexes automatically
4. Commit the updated `backend/vector_store/` files for deployment

---

## Design Decisions

### Why Gemini API for Embeddings?
- 3072-dimensional embeddings — high quality semantic representation
- Free API tier — no cost for moderate usage
- No local ML frameworks needed — keeps deployment under 100 MB
- REST API call — lightweight `requests` library instead of heavy `torch` + `transformers`

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
- Single API key unlocks many open-source models (Llama, Gemma, Mistral)
- Built-in model fallback chain if primary model is unavailable

---

## Project Structure

```
mini-rag/
├── backend/
│   ├── app.py              # Main application (RAG pipeline + FastAPI)
│   ├── requirements.txt
│   └── vector_store/       # Pre-built index (committed to git)
│       ├── faiss_index.bin
│       └── chunks.json
├── documents/
│   ├── construction_policies.txt
│   ├── platform_faq.txt
│   └── technical_specs.txt
├── frontend/
│   ├── index.html          # Single-file chat UI
│   └── vercel.json         # Vercel deployment config
├── scripts/
│   └── evaluate.py         # Quality evaluation script
├── .env.example            # API key template
├── .gitignore
└── README.md
```
