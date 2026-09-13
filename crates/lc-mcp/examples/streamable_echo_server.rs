//! Spec-based demo MCP server over the **official Streamable HTTP** transport
//! (B1, 0.22.4).
//!
//! Serves one `echo` tool at `http://127.0.0.1:<port>/mcp` using
//! [`MCPServer::serve_streamable_http`] (initialize handshake,
//! `Mcp-Session-Id`, JSON/SSE content negotiation, HTTP 202 notifications) —
//! the fixture for the official TS/Python SDK client interop matrix.
//!
//! ```text
//! cargo run -p lc-mcp --example streamable_echo_server -- [--port 0] [--bearer TOKEN]
//! ```
//!
//! On startup it prints exactly one machine-readable line to stdout and
//! flushes, so a parent process can parse the endpoint:
//!
//! ```text
//! MCP_STREAMABLE_URL=http://127.0.0.1:51837/mcp
//! ```
//!
//! - `--port <n>`: bind port (default `0` = ephemeral).
//! - `--bearer <token>`: require `Authorization: Bearer <token>` on every
//!   request (401 + `WWW-Authenticate` otherwise).

use std::sync::Arc;

use lc_core::tools::ToolError;
use lc_core::BaseTool;
use lc_mcp::auth::StaticBearerValidator;
use lc_mcp::MCPServer;
use serde_json::{json, Value};
use tokio::net::TcpListener;

struct EchoTool;

#[async_trait::async_trait]
impl BaseTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echoes the `msg` argument back to the caller."
    }
    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "msg": {"type": "string"}
            },
            "required": ["msg"]
        }))
    }
    async fn run(&self, input: String) -> Result<String, ToolError> {
        // Echo the raw arguments JSON; interop harnesses assert on substring.
        Ok(input)
    }
}

#[tokio::main]
async fn main() {
    let mut port = 0u16;
    let mut bearer: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                if let Some(value) = args.next() {
                    port = value.parse().expect("--port must be a u16");
                }
            }
            "--bearer" => bearer = args.next(),
            _ => {}
        }
    }

    let mut server = MCPServer::new()
        .with_server_info("echo-streamable", env!("CARGO_PKG_VERSION"))
        .with_tool(Arc::new(EchoTool));
    if let Some(token) = bearer {
        server = server.with_token_validator(Arc::new(StaticBearerValidator::new(token)));
    }

    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind 127.0.0.1");
    let url = Arc::new(server).serve_streamable_http(listener);

    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "MCP_STREAMABLE_URL={url}").expect("write url");
    stdout.flush().expect("flush url");

    // The accept loop runs on a background task; park until killed.
    tokio::signal::ctrl_c().await.ok();
}
