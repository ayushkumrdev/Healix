[CmdletBinding()]
param(
  [int]$Port = 8000
)
$ErrorActionPreference = "Stop"

# Resolve repo root
$repoRoot = Split-Path -Parent $PSScriptRoot

# Ensure venv python
$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv python: $py" }

# Run uvicorn for medication support API
& $py -m uvicorn backend.services.medication_support:app --host 0.0.0.0 --port $Port

