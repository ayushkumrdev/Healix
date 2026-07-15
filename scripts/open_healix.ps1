# Healix desktop launcher.
# Reuses an already-running server when possible; otherwise starts
# start_system.bat minimized and waits for the API, then opens the app.

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$port = if ($env:BAYMAX_PORT) { $env:BAYMAX_PORT } else { "8504" }
$url = "http://127.0.0.1:$port"

function Test-Healix {
    try {
        $r = Invoke-WebRequest -Uri "$url/api/info" -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

if (-not (Test-Healix)) {
    Start-Process -FilePath "$root\start_system.bat" -WorkingDirectory $root -WindowStyle Minimized
    $deadline = (Get-Date).AddSeconds(90)
    while (-not (Test-Healix) -and ((Get-Date) -lt $deadline)) {
        Start-Sleep -Milliseconds 700
    }
}

Start-Process $url
