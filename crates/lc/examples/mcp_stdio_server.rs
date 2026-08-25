//! 示例:MCP Server —— stdio(pipe)传输。
//!
//! 展示如何把 langchainrust 的内置工具通过 [`MCPServer::serve_stdio`] 暴露为
//! **本地管道** MCP 服务器:从 stdin 读 JSON-RPC 请求,处理后写回 stdout,
//! 全程不经过网络。客户端(MCPClient / Claude Desktop / Cursor 等)用
//! `MCPConfig::stdio(命令, 参数)` 启动本二进制,通过管道与之通信。
//!
//! # 构建
//!
//! ```powershell
//! cargo build --release -p langchainrust --example mcp_stdio_server
//! # 产物:target/release/examples/mcp_stdio_server.exe
//! ```
//!
//! # 手动验证(直接向 stdin 写一行 JSON-RPC)
//!
//! ```powershell
//! echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | .\target\release\examples\mcp_stdio_server.exe
//! # 会回一行 initialize 响应(capabilities/serverInfo)
//! ```
//!
//! # 客户端接入(框架,走管道)
//!
//! ```no_run
//! use langchainrust::mcp::{MCPClient, MCPConfig};
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = MCPClient::connect(MCPConfig::stdio(
//!     "target/release/examples/mcp_stdio_server".into(),
//!     Vec::new(),
//! ))
//! .await?;
//! let tools = client.list_tools().await?;
//! println!("管道连上,共 {} 个工具", tools.len());
//! # Ok(())
//! # }
//! ```

use langchainrust::mcp::MCPServer;
use langchainrust::{
    Calculator, DateTimeTool, DuckDuckGoSearchTool, SimpleMathTool, URLFetchTool, WikipediaTool,
};
use std::sync::Arc;

/// 注册一组开箱即用的内置工具(与 `mcp_sse_server` 一致)。
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

    // 从 stdin 读 JSON-RPC 请求,处理后写回 stdout,循环直到 EOF。
    // 客户端关闭管道(或本进程 stdin 到达 EOF)即退出。
    server.serve_stdio().await?;
    Ok(())
}
