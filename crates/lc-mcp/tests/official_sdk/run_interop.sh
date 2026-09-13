#!/usr/bin/env bash
# B1 official-SDK interop gate (Linux / macOS).
#
# Runs the full B1 interop matrix:
#   1. official TypeScript SDK client  <->  our Streamable HTTP server (JSON mode)
#   2. official TypeScript SDK client  <->  our Streamable HTTP server (bearer auth)
#   3. official Python SDK client      <->  our Streamable HTTP server (if `mcp` installed)
#   4. our StdioMcpClient              <->  official TypeScript SDK stdio server
#
# Usage: crates/lc-mcp/tests/official_sdk/run_interop.sh [--skip-python]
# Requires: Rust toolchain, Node.js 18+, npm. Python 3.11+ with `mcp` is optional.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SKIP_PYTHON=0
[ "${1:-}" = "--skip-python" ] && SKIP_PYTHON=1
FAILURES=()

if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo "Node.js + npm are required for the official-SDK interop gate." >&2
  exit 2
fi

if [ ! -d "$HERE/node_modules" ]; then
  echo "=== npm install (official TypeScript SDK) ==="
  (cd "$HERE" && npm install --no-audit --no-fund) || exit 1
fi

echo "=== cargo build --example streamable_echo_server ==="
(cd "$REPO" && cargo build -p lc-mcp --example streamable_echo_server) || exit 1
TARGET_DIR="$(cd "$REPO" && cargo metadata --format-version 1 --no-deps \
  | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')"
SERVER_BIN="$TARGET_DIR/debug/examples/streamable_echo_server"
[ -x "$SERVER_BIN" ] || SERVER_BIN="$TARGET_DIR/debug/examples/streamable_echo_server.exe"

start_server() {
  local bearer="$1"
  local out err pid url
  out="$(mktemp)"; err="$(mktemp)"
  if [ -n "$bearer" ]; then
    "$SERVER_BIN" --bearer "$bearer" >"$out" 2>"$err" &
  else
    "$SERVER_BIN" >"$out" 2>"$err" &
  fi
  pid=$!
  for _ in $(seq 1 100); do
    url="$(sed -n 's/^MCP_STREAMABLE_URL=//p' "$out" | head -n1)"
    [ -n "$url" ] && break
    kill -0 "$pid" 2>/dev/null || { echo "server exited early; stderr:" >&2; cat "$err" >&2; exit 1; }
    sleep 0.1
  done
  if [ -z "$url" ]; then
    echo "server never advertised MCP_STREAMABLE_URL; stderr:" >&2; cat "$err" >&2
    exit 1
  fi
  echo "$pid $out $err $url"
}

stop_server() {
  local pid="$1" out="$2" err="$3"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  rm -f "$out" "$err"
}

run_ts() {
  local label="$1"; shift
  echo "=== official TS SDK client -> $label ==="
  (cd "$HERE" && node ts_streamable_client.mjs "$@") || FAILURES+=("$label")
}

read -r PID OUT ERR URL <<<"$(start_server "")"
run_ts "langchainrust Streamable HTTP server" --url "$URL"
stop_server "$PID" "$OUT" "$ERR"

TOKEN="interop-bearer-0x42"
read -r PID OUT ERR URL <<<"$(start_server "$TOKEN")"
run_ts "bearer-protected server" --url "$URL" --bearer "$TOKEN"
stop_server "$PID" "$OUT" "$ERR"

if [ "$SKIP_PYTHON" -eq 0 ] && command -v python >/dev/null && python -c "import mcp" 2>/dev/null; then
  echo "=== official Python SDK client -> langchainrust Streamable HTTP server ==="
  read -r PID OUT ERR URL <<<"$(start_server "")"
  (cd "$HERE" && python py_streamable_client.py --url "$URL") || FAILURES+=("Python streamable")
  stop_server "$PID" "$OUT" "$ERR"
else
  echo "=== skipping Python stage (python+mcp not installed; pip install -r requirements.txt) ==="
fi

echo "=== langchainrust StdioMcpClient -> official TS SDK stdio server ==="
(cd "$REPO" && cargo test -p lc-mcp --test official_sdk_stdio_interop -- --ignored --nocapture) \
  || FAILURES+=("Rust stdio -> TS SDK server")

if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "B1 INTEROP GATE FAILED:" >&2
  printf ' - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
echo "B1 INTEROP GATE PASSED"
