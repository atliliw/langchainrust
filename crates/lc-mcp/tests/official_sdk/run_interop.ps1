# B1 official-SDK interop gate (Windows / PowerShell).
#
# Runs the full B1 interop matrix:
#   1. official TypeScript SDK client  <->  our Streamable HTTP server (JSON mode)
#   2. official TypeScript SDK client  <->  our Streamable HTTP server (bearer auth)
#   3. official Python SDK client      <->  our Streamable HTTP server (if `mcp` installed)
#   4. our StdioMcpClient              <->  official TypeScript SDK stdio server
#
# Usage from the repo root (or anywhere — paths are resolved by this script):
#   powershell -ExecutionPolicy Bypass -File crates\lc-mcp\tests\official_sdk\run_interop.ps1
#   ... -SkipPython   to skip the optional Python stage
#
# Requires: Rust toolchain, Node.js 18+, npm. Python 3.11+ with `mcp` is optional.
param([switch]$SkipPython)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here "..\..\..\..")).Path
$harness = $here
$failures = New-Object System.Collections.Generic.List[string]

function Section($name) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
}

# --- prerequisites -----------------------------------------------------------
$node = Get-Command node -ErrorAction SilentlyContinue
$npm = Get-Command npm -ErrorAction SilentlyContinue
if (-not $node -or -not $npm) {
    throw "Node.js + npm are required for the official-SDK interop gate."
}

if (-not (Test-Path (Join-Path $harness "node_modules"))) {
    Section "npm install (official TypeScript SDK)"
    Push-Location $harness
    try {
        npm install --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw "npm install failed" }
    }
    finally {
        Pop-Location
    }
}

# --- build the example server ------------------------------------------------
Section "cargo build --example streamable_echo_server"
Push-Location $repo
try {
    cargo build -p lc-mcp --example streamable_echo_server
    if ($LASTEXITCODE -ne 0) { throw "example build failed" }
    $targetDir = (cargo metadata --format-version 1 --no-deps | ConvertFrom-Json).target_directory
}
finally {
    Pop-Location
}
$serverExe = Join-Path $targetDir "debug\examples\streamable_echo_server.exe"
if (-not (Test-Path $serverExe)) {
    # Non-Windows layout (kept for completeness; use run_interop.sh there).
    $serverExe = Join-Path $targetDir "debug/examples/streamable_echo_server"
}

function Start-EchoServer($bearer) {
    $out = [System.IO.Path]::GetTempFileName()
    if ($bearer) {
        $proc = Start-Process -FilePath $serverExe -ArgumentList @("--bearer", $bearer) `
            -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
            -NoNewWindow -PassThru
    }
    else {
        $proc = Start-Process -FilePath $serverExe `
            -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
            -NoNewWindow -PassThru
    }
    $url = $null
    for ($i = 0; $i -lt 100 -and -not $url; $i++) {
        Start-Sleep -Milliseconds 100
        if (Test-Path $out) {
            $line = Get-Content $out -ErrorAction SilentlyContinue |
                Where-Object { $_ -like "MCP_STREAMABLE_URL=*" } |
                Select-Object -First 1
            if ($line) { $url = $line.Substring("MCP_STREAMABLE_URL=".Length) }
        }
        if ($proc.HasExited) { throw "streamable_echo_server exited early; see $out.err" }
    }
    if (-not $url) { throw "server never advertised MCP_STREAMABLE_URL (see $out)" }
    return [pscustomobject]@{ Proc = $proc; Url = $url; OutFile = $out }
}

function Stop-EchoServer($srv) {
    if ($srv.Proc -and -not $srv.Proc.HasExited) {
        Stop-Process -Id $srv.Proc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $srv.OutFile, "$($srv.OutFile).err" -ErrorAction SilentlyContinue
}

# --- 1+2: official TS SDK client -> our Streamable HTTP server ---------------
Section "official TypeScript SDK client -> langchainrust Streamable HTTP server"
$srv = Start-EchoServer $null
try {
    Push-Location $harness
    try {
        node ts_streamable_client.mjs --url $srv.Url
        if ($LASTEXITCODE -ne 0) { $failures.Add("TS streamable (anon)") }
    }
    finally {
        Pop-Location
    }
}
finally {
    Stop-EchoServer $srv
}

Section "official TypeScript SDK client -> bearer-protected server"
$token = "interop-bearer-0x42"
$srv = Start-EchoServer $token
try {
    Push-Location $harness
    try {
        node ts_streamable_client.mjs --url $srv.Url --bearer $token
        if ($LASTEXITCODE -ne 0) { $failures.Add("TS streamable (bearer)") }
    }
    finally {
        Pop-Location
    }
}
finally {
    Stop-EchoServer $srv
}

# --- 3: official Python SDK client -> our Streamable HTTP server -------------
if (-not $SkipPython) {
    Section "official Python SDK client -> langchainrust Streamable HTTP server"
    $py = Get-Command python -ErrorAction SilentlyContinue
    $mcpReady = $false
    if ($py) {
        python -c "import mcp" 2>$null
        $mcpReady = ($LASTEXITCODE -eq 0)
    }
    if (-not $mcpReady) {
        Write-Warning "python with the 'mcp' package not found; skipping (install: pip install -r requirements.txt)"
    }
    else {
        $srv = Start-EchoServer $null
        try {
            Push-Location $harness
            try {
                python py_streamable_client.py --url $srv.Url
                if ($LASTEXITCODE -ne 0) { $failures.Add("Python streamable") }
            }
            finally {
                Pop-Location
            }
        }
        finally {
            Stop-EchoServer $srv
        }
    }
}

# --- 4: our stdio client -> official TS SDK stdio server ---------------------
Section "langchainrust StdioMcpClient -> official TypeScript SDK stdio server"
Push-Location $repo
try {
    cargo test -p lc-mcp --test official_sdk_stdio_interop -- --ignored --nocapture
    if ($LASTEXITCODE -ne 0) { $failures.Add("Rust stdio -> TS SDK server") }
}
finally {
    Pop-Location
}

# --- result ------------------------------------------------------------------
if ($failures.Count -gt 0) {
    Write-Host "`nB1 INTEROP GATE FAILED:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "`nB1 INTEROP GATE PASSED" -ForegroundColor Green
