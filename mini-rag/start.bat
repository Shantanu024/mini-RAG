@echo off
echo.
echo  ConstructIQ - Mini RAG Startup
echo ======================================

:: Load .env if present
if exist .env (
    for /f "usebackq tokens=* delims=" %%a in (".env") do (
        echo %%a | findstr /r "^#" >nul || set "%%a"
    )
    echo  Loaded .env
)

if "%OPENROUTER_API_KEY%"=="" (
    echo  WARNING: OPENROUTER_API_KEY not set - running in fallback mode (no LLM generation)
    echo    Set it with: set OPENROUTER_API_KEY=your_key
) else (
    echo  OpenRouter API key found
)

echo.
echo  Installing dependencies...
cd backend
pip install -r requirements.txt -q

echo.
echo  Starting API server on http://localhost:8000 ...
echo    Open frontend\index.html in your browser to use the chatbot
echo.
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
