[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

# Resolve repo root relative to this script
$repoRoot = Split-Path -Parent $PSScriptRoot

# Configure environment for the running session
$env:BAYMAX_GGUF_MODEL_DIR = Join-Path $repoRoot "artifacts"
$env:BAYMAX_GGUF_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
$env:BAYMAX_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"

# Quick checks
$modelPath = Join-Path $env:BAYMAX_GGUF_MODEL_DIR $env:BAYMAX_GGUF_MODEL_NAME
if (-not (Test-Path $modelPath)) { Write-Error "Model file not found: $modelPath"; exit 1 }

# Run E2E smoke test
$smoke = Join-Path $repoRoot "scripts\e2e_smoke.py"
python $smoke

