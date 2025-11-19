[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

# Resolve repo root relative to this script
$repoRoot = Split-Path -Parent $PSScriptRoot

# Configure environment for the running session
$env:BAYMAX_GGUF_MODEL_DIR = Join-Path $repoRoot "artifacts"
$env:BAYMAX_GGUF_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
$env:BAYMAX_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
$env:BAYMAX_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
$env:BAYMAX_RERANK_TOPN = "50"
$env:BAYMAX_RERANK_BATCH = "32"
$env:BAYMAX_RERANK_SKIP_THRESHOLD = "0.86"
$env:BAYMAX_MAIN_PROMPT_FILE = (Join-Path $repoRoot "healix_main_prompt.txt")
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:BAYMAX_ENABLE_XAI = "1"

# Sanity checks
$modelPath = Join-Path $env:BAYMAX_GGUF_MODEL_DIR $env:BAYMAX_GGUF_MODEL_NAME
if (-not (Test-Path $modelPath)) {
  Write-Error "Model file not found: $modelPath"
  exit 1
}

$indexDir = Join-Path $repoRoot "data"
$required = @(
  (Join-Path $indexDir "index.faiss"),
  (Join-Path $indexDir "faiss_metadata.pkl"),
  (Join-Path $indexDir "indexed_chunks.pkl"),
  (Join-Path $indexDir "index_config.json")
)
foreach ($p in $required) { if (-not (Test-Path $p)) { Write-Error "Missing required file: $p"; exit 1 } }

# Launch Streamlit app (assumes venv is already activated in this shell)
$frontend = Join-Path $repoRoot "frontend\app_professional.py"
python -m streamlit run $frontend

