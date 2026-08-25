//! Example / deployable MCP SSE server.
//!
//! Exposes langchainrust's built-in tools as a networked MCP server via `MCPServer::serve_sse`,
//! callable by MCP clients (MCPClient / Cursor / Claude Desktop, etc.).
//!
//! # Run (local testing)
//!
//! ```powershell
//! cargo run -p langchainrust --example mcp_sse_server
//! ```
//!
//! # Build a standalone executable (for deployment)
//!
//! ```powershell
//! cargo build --release -p langchainrust --example mcp_sse_server
//! # Artifact: target/release/examples/mcp_sse_server.exe, copy it to the remote server and run
//! ```
//!
//! # Runtime configuration (environment variables, not hardcoded)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `MCP_SERVER_HOST` | `127.0.0.1` | Bind address (default local-only; for remote access set it explicitly to `0.0.0.0` and configure auth / network whitelist yourself) |
//! | `MCP_SERVER_PORT` | `8788` | Listening port |
//! | `MCP_SERVER_PUBLIC_URL` | see below | Base URL clients use to reach this server |
//!
//! For remote deployment you **must set** `MCP_SERVER_PUBLIC_URL`, otherwise the POST address
//! the server sends to clients would be written as `0.0.0.0` and clients could not connect.
//! Local testing can omit it.
//!
//! ```powershell
//! $env:MCP_SERVER_PORT = "8788"
//! $env:MCP_SERVER_PUBLIC_URL = "http://<your-server-public-ip-or-domain>:8788"
//! .\target\release\examples\mcp_sse_server.exe
//! ```
//!
//! On startup it prints the client connection endpoint: `http://<host>:<port>/sse`.

use langchainrust::mcp::{MCPRequest, MCPServer};
use langchainrust::{
    Calculator, DateTimeTool, DuckDuckGoSearchTool, SimpleMathTool, URLFetchTool, WikipediaTool,
};
use std::sync::Arc;
use tokio::net::TcpListener;

/// Registers a set of ready-to-use built-in tools.
///
/// Deliberately does not register `PythonREPLTool` (it can execute arbitrary code remotely,
/// a security risk); add more tools here with `with_tool` when needed.
fn build_server() -> MCPServer {
    MCPServer::new()
        .with_server_info("langchainrust-mcp-server", env!("CARGO_PKG_VERSION"))
        .with_tool(Arc::new(Calculator))
        .with_tool(Arc::new(SimpleMathTool))
        .with_tool(Arc::new(DateTimeTool))
        .with_tool(Arc::new(URLFetchTool::new()))
        .with_tool(Arc::new(WikipediaTool::new()))
        .with_tool(Arc::new(DuckDuckGoSearchTool::new()))
}

#[tokio::main]
async fn main() {
    // 1. Read the configuration (environment variables with defaults)
    // Default binds to the local loopback only; this server has no auth and mounts
    // SSRF-reachable tools like url_fetch. Binding 0.0.0.0 would expose your internal
    // network and cloud metadata to anyone who can reach this port, so it must be
    // an explicit choice.
    let host = std::env::var("MCP_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("MCP_SERVER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8788);

    // 2. Bind the listen address (default 127.0.0.1 = local only; 0.0.0.0 = all interfaces, must be explicit)
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| {
            eprintln!("failed to bind {host}:{port}: {e}");
            std::process::exit(1);
        });
    let bound = listener.local_addr().unwrap();

    // 3. Base URL for clients: required for remote deployment, otherwise 0.0.0.0 cannot be reached
    let public_base = std::env::var("MCP_SERVER_PUBLIC_URL").unwrap_or_else(|_| {
        eprintln!("⚠ MCP_SERVER_PUBLIC_URL is not set; remote clients will be unable to reach the POST address.");
        eprintln!("  set it to: http://<server-public-ip-or-domain>:<port>");
        format!("http://{bound}")
    });

    // 4. Build the server, print the registered tools, and start serving
    let server = Arc::new(build_server());
    let names = registered_tool_names(&server).await;
    println!("registered {} tools: {}", names.len(), names.join(", "));

    let sse_url = server.serve_sse(listener, public_base);
    println!("MCP SSE server started ✅");
    println!("client connection endpoint: {sse_url}");
    println!("press Ctrl+C to stop.");

    // 5. Keep the process alive (the receive loop runs in a background task)
    std::future::pending::<()>().await;
}

/// Fetches the actually registered tool names via tools/list (what a client would see).
async fn registered_tool_names(server: &MCPServer) -> Vec<String> {
    let resp = server
        .handle_request(MCPRequest::new(1, "tools/list", None))
        .await;
    resp.result
        .as_ref()
        .and_then(|r| r.get("tools"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}
