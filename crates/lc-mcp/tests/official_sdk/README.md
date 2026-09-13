# Official MCP SDK interop gate (B1)

This directory is the **external-SDK half** of the B1 transport gate. Cargo
does not compile anything here (no Rust targets live in this subtree); the
assets are driven by the runner scripts and by the ignored Rust test one
directory up.

## The matrix

| Direction | Client | Server | Asset |
|---|---|---|---|
| official SDK → ours | TypeScript SDK `StreamableHTTPClientTransport` | `examples/streamable_echo_server.rs` | `ts_streamable_client.mjs` |
| official SDK → ours, bearer | TypeScript SDK with `Authorization: Bearer` | same example (`--bearer`) | same |
| official SDK → ours | Python SDK `streamablehttp_client` + `ClientSession` | same example | `py_streamable_client.py` |
| ours → real SDK server | `StdioMcpClient` (Rust) | TypeScript SDK `Server` + `StdioServerTransport` | `ts_stdio_echo_server.mjs` |

The in-tree half (our own client ↔ our own servers, plus raw HTTP status/SSE
matrix coverage) lives in `../streamable_http_interop.rs`,
`../streamable_http_server_interop.rs` and `../stdio_interop.rs` and runs with
every `cargo test -p lc-mcp`.

## Running

Prerequisites: Rust toolchain, Node.js 18+, npm. Python 3.11+ is optional.

```powershell
# Windows (from repo root)
powershell -ExecutionPolicy Bypass -File crates\lc-mcp\tests\official_sdk\run_interop.ps1
```

```bash
# Linux / macOS
crates/lc-mcp/tests/official_sdk/run_interop.sh            # includes Python if available
crates/lc-mcp/tests/official_sdk/run_interop.sh --skip-python
```

The runner builds the example server, `npm install`s the pinned
`@modelcontextprotocol/sdk` into this directory on first use, spawns the
server on an ephemeral port (parsing its `MCP_STREAMABLE_URL=` line), and runs
each stage.

Python stage only:

```bash
pip install -r requirements.txt
python py_streamable_client.py --url http://127.0.0.1:PORT/mcp
```

Rust-against-TS-stdio stage only:

```bash
cd crates/lc-mcp/tests/official_sdk && npm install
cd -
cargo test -p lc-mcp --test official_sdk_stdio_interop -- --ignored --nocapture
```

The Rust test is `#[ignore]`d so ordinary `cargo test` runs need neither Node
nor npm; the gate is explicit.

## What each stage asserts

- the complete official handshake (`initialize` → `notifications/initialized`)
  succeeds against our session lifecycle, including `Mcp-Session-Id` echo,
  202 notifications and content negotiation;
- `tools/list` returns the `echo` tool;
- `tools/call` round-trips a unique marker through real SDK serialization;
- `ping` is answered (the SDK server answers it in its protocol layer; our
  `MCPServer` answers it explicitly);
- the bearer-protected run proves the SDK accepts our 401 +
  `WWW-Authenticate` challenge flow once it holds the right token;
- the Rust client against the TS stdio server additionally asserts
  `serverInfo` negotiation, JSON-RPC `-32601` on an unknown tool, and clean
  child shutdown.

`node_modules/`, `__pycache__/` and virtualenvs are git-ignored here.
