//! Example: MCP Server — stdio (pipe) transport.
//!
//! Shows how to expose langchainrust's built-in tools as a **local-pipe** MCP server via
//! [`MCPServer::serve_stdio`]: read JSON-RPC requests from stdin, process them, write the
//! results back to stdout — never touching the network. External clients (Claude Desktop /
//! Cursor, etc.) start this binary over the pipe and talk JSON-RPC line framing.
//!
//! # Build
//!
//! ```powershell
//! cargo build --release -p langchainrust --example mcp_stdio_server
//! # Artifact: target/release/examples/mcp_stdio_server.exe
//! ```
//!
//! # Manual verification (write one JSON-RPC line to stdin)
//!
//! ```powershell
//! echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | .\target\release\examples\mcp_stdio_server.exe
//! # it replies with an initialize response (capabilities/serverInfo)
//! ```
//!
//! # In-framework access
//!
//! The 0.22.0 stateless track has no in-framework stdio client — in-framework
//! integration goes through the HTTP track: run `mcp_http_server` and connect
//! with `StatelessMcpClient::connect("http://127.0.0.1:8788/mcp")`.
//!
//! ```no_run
//! use langchainrust::mcp::StatelessMcpClient;
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = StatelessMcpClient::connect("http://127.0.0.1:8788/mcp");
//! let tools = client.list_tools().await?;
//! println!("{} tools", tools.len());
//! # Ok(())
//! # }
//! ```

use langchainrust::mcp::MCPServer;
use langchainrust::{
    Calculator, DateTimeTool, DuckDuckGoSearchTool, SimpleMathTool, URLFetchTool, WikipediaTool,
};
use std::sync::Arc;

/// Registers a set of ready-to-use built-in tools (the same set as `mcp_http_server`).
fn build_server() -> MCPServer {
    MCPServer::new()
        .with_server_info("langchainrust-stdio-server", env!("CARGO_PKG_VERSION"))
        .with_tool(Arc::new(Calculator))
        .with_tool(Arc::new(SimpleMathTool))
        .with_tool(Arc::new(DateTimeTool))
        .with_tool(Arc::new(URLFetchTool::new()))
        .with_tool(Arc::new(WikipediaTool::new()))
        .with_tool(Arc::new(DuckDuckGoSearchTool::new()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = build_server();

    // Read JSON-RPC requests from stdin, process them, write the results back to stdout,
    // looping until EOF. Exits when the client closes the pipe (or this process's stdin
    // reaches EOF).
    server.serve_stdio().await?;
    Ok(())
}
