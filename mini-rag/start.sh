#!/bin/bash
set -e

echo "🏗️  ConstructIQ — Mini RAG Startup"
echo "======================================"

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅  Loaded .env"
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "⚠️  OPENROUTER_API_KEY not set — running in fallback mode (no LLM generation)"
  echo "   Set it with: export OPENROUTER_API_KEY=your_key"
else
  echo "✅  OpenRouter API key found"
fi

echo ""
echo "📦  Installing dependencies..."
cd backend
pip install -r requirements.txt -q

echo ""
echo "🚀  Starting API server on http://localhost:8000 ..."
echo "   Open frontend/index.html in your browser to use the chatbot"
echo ""
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
