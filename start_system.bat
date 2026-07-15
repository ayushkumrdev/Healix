@echo off
setlocal enableextensions enabledelayedexpansion

rem =============================================================
rem  Baymax Healthcare Super-Assistant - Start Script (Windows)
rem  This starts the Streamlit UI using the project's virtualenv
rem =============================================================

rem Change to the directory of this script (project root)
pushd "%~dp0"

rem -------------------------------------------------------------
rem Active pipeline environment (model, embeddings, reranker, threads)
rem Update these if you change artifacts or defaults.
rem -------------------------------------------------------------
rem ================= Fast Mode preset =================
rem Core performance flags with DYNAMIC token allocation (ChatGPT-style)
set BAYMAX_FAST_MODE=true
set BAYMAX_GEN_THREADS=12

rem Dynamic token allocation (variable response length based on query)
rem Max tokens: upper limit for complex questions (default: 500)
rem Min tokens: lower limit for simple greetings (default: 30)
rem System automatically adjusts between min and max based on query complexity
set BAYMAX_GEN_MAX_TOKENS=900
set BAYMAX_GEN_MIN_TOKENS=60
rem Legacy: Use max tokens as default fallback
set BAYMAX_GEN_TOKENS=900
set BAYMAX_GEN_SHORT_TOKENS=256

rem Optional: Enable token allocation debug info
rem set BAYMAX_DEBUG_TOKENS=1
set BAYMAX_GEN_CONTEXT=8192
set BAYMAX_GEN_STREAM=true

rem Model paths
set BAYMAX_GGUF_MODEL_NAME=mistral-7b-instruct-v0.2.Q4_K_M.gguf
set BAYMAX_GGUF_MODEL_DIR=artifacts
set BAYMAX_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

rem Enable reranker (use "none" to disable)
set BAYMAX_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2
set BAYMAX_RERANK_TOPN=50
set BAYMAX_RERANK_BATCH=32
set BAYMAX_RERANK_SKIP_THRESHOLD=0.86

rem Threads (compat vars)
set BAYMAX_GPT4ALL_THREADS=12

rem Generation tuning
set BAYMAX_GEN_TEMP=0.4
set BAYMAX_GEN_TOP_K=10
set BAYMAX_GEN_TOP_P=0.9
set BAYMAX_GEN_STYLE=abstractive
set BAYMAX_CONTEXT_CHARS=400
set BAYMAX_POST_ABSTRACT=0
rem Main prompt template (single source of truth)
set BAYMAX_MAIN_PROMPT_FILE=%CD%\healix_main_prompt.txt
set BAYMAX_GENERATE_ALL=1
set BAYMAX_NO_FALLBACKS=1

rem Retrieval behavior
set BAYMAX_RAG_STRICT=0

rem Optional GPU acceleration
set LLAMA_CUBLAS=1
set CUDA_VISIBLE_DEVICES=0
set BAYMAX_CUDA_DEVICE=0
set BAYMAX_FAISS_GPU=1
set BAYMAX_USE_LLAMA_CPP=1
set BAYMAX_GPU_LAYERS=99999

rem Ensure CUDA 12.6 runtime DLLs are on PATH for llama.cpp cuBLAS
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%"

rem Enable XAI (explainability package)
set BAYMAX_ENABLE_XAI=1

rem Conversation memory (persistent multi-chat history + per-turn recall window)
set BAYMAX_MEMORY_TURNS=10
set BAYMAX_MEMORY_CHARS=2400

rem Agentic Mixture-of-Experts routing (HEALIX_MOE=0 to disable / A-B test)
set HEALIX_MOE=1

rem Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with RRF
set HEALIX_HYBRID=1
rem HyDE query expansion + RAPTOR overview tree (fail-open if artifacts absent)
set HEALIX_HYDE=1
set HEALIX_RAPTOR=1

rem LLM backend: use the local Ollama server (clear BAYMAX_LLM_BACKEND for GGUF)
set BAYMAX_LLM_BACKEND=ollama
set BAYMAX_OLLAMA_URL=http://localhost:11434
set BAYMAX_OLLAMA_MODEL=qwen2.5:7b-instruct

rem Streamlit settings (use defaults for security)
rem set STREAMLIT_SERVER_ENABLECORS=false
rem set STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
rem set STREAMLIT_BROWSER_GATHERERUSAGESTATS=false

rem Port for the Healix web app
if not defined BAYMAX_PORT set BAYMAX_PORT=8504

rem Check for virtual environment Python
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment Python not found: .venv\Scripts\python.exe
  echo         Please create/activate the venv in this project first.
  echo         Example: py -m venv .venv ^&^& .\.venv\Scripts\activate.bat ^&^& pip install -r requirements.txt
  popd
  exit /b 1
)

rem Display config summary
echo ===============================================
echo  Baymax Healthcare Super-Assistant - Launcher
echo ===============================================
echo Project Dir  : %CD%
echo Using Python : .venv\Scripts\python.exe
if defined BAYMAX_GGUF_MODEL_NAME echo Model Name   : %BAYMAX_GGUF_MODEL_NAME%
if defined BAYMAX_GGUF_MODEL_DIR  echo Model Dir    : %BAYMAX_GGUF_MODEL_DIR%
echo UI Port      : %BAYMAX_PORT%
echo.

rem Start the Healix web app (FastAPI + custom chat UI) in the foreground
echo Open http://127.0.0.1:%BAYMAX_PORT% in your browser
".venv\Scripts\python.exe" -m uvicorn backend.api:app --host 127.0.0.1 --port %BAYMAX_PORT%

rem Return to original directory
popd
endlocal

