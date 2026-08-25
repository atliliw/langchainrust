//! 示例 / 可部署的 MCP SSE 服务器。
//!
//! 把 langchainrust 的内置工具通过 `MCPServer::serve_sse` 暴露为网络 MCP server,
//! 供 MCP 客户端(MCPClient / Cursor / Claude Desktop 等)调用。
//!
//! # 运行(本地联调)
//!
//! ```powershell
//! cargo run -p langchainrust --example mcp_sse_server
//! ```
//!
//! # 构建独立可执行文件(部署用)
//!
//! ```powershell
//! cargo build --release -p langchainrust --example mcp_sse_server
//! # 产物:target/release/examples/mcp_sse_server.exe,拷到远程服务器即可运行
//! ```
//!
//! # 运行配置(环境变量,不写死在代码里)
//!
//! | 变量 | 默认 | 说明 |
//! |---|---|---|
//! | `MCP_SERVER_HOST` | `127.0.0.1` | 绑定地址(默认仅本机;远程访问需显式设为 `0.0.0.0`,且必须自行配置鉴权/网络白名单) |
//! | `MCP_SERVER_PORT` | `8788` | 监听端口 |
//! | `MCP_SERVER_PUBLIC_URL` | 见下 | 客户端访问本服务器的基地址 |
//!
//! 远程部署时**必须设置** `MCP_SERVER_PUBLIC_URL`,否则服务端发给客户端的
//! POST 地址会写成 `0.0.0.0`,客户端连不上。本地联调可不设。
//!
//! ```powershell
//! $env:MCP_SERVER_PORT = "8788"
//! $env:MCP_SERVER_PUBLIC_URL = "http://<你的服务器公网IP或域名>:8788"
//! .\target\release\examples\mcp_sse_server.exe
//! ```
//!
//! 启动后打印客户端连接入口:`http://<host>:<port>/sse`。

use langchainrust::mcp::{MCPRequest, MCPServer};
use langchainrust::{
    Calculator, DateTimeTool, DuckDuckGoSearchTool, SimpleMathTool, URLFetchTool, WikipediaTool,
};
use std::sync::Arc;
use tokio::net::TcpListener;

/// 注册一组开箱即用的内置工具。
///
/// 刻意不注册 `PythonREPLTool`(远程可执行任意代码,有安全风险);
/// 需要更多工具时在这里加 `with_tool`。
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
    // 1. 读取配置(环境变量,带默认值)
    // 默认仅绑定本机回环;该 server 无鉴权且挂载 url_fetch 等 SSRF 可达工具,
    // 绑 0.0.0.0 会把内部网络和云元数据暴露给任何能到达该端口的人,须显式选择
    let host = std::env::var("MCP_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("MCP_SERVER_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8788);

    // 2. 绑定监听地址(默认 127.0.0.1 = 仅本机;0.0.0.0 = 所有网卡,须显式设置)
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| {
            eprintln!("绑定 {host}:{port} 失败: {e}");
            std::process::exit(1);
        });
    let bound = listener.local_addr().unwrap();

    // 3. 客户端访问基地址:远程部署必须显式给出,否则 0.0.0.0 连不上
    let public_base = std::env::var("MCP_SERVER_PUBLIC_URL").unwrap_or_else(|_| {
        eprintln!("⚠ 未设置 MCP_SERVER_PUBLIC_URL,远程部署时客户端将无法回连 POST 地址。");
        eprintln!("  请设为: http://<服务器公网IP或域名>:<port>");
        format!("http://{bound}")
    });

    // 4. 建 server、打印已注册工具、开服
    let server = Arc::new(build_server());
    let names = registered_tool_names(&server).await;
    println!("已注册 {} 个工具: {}", names.len(), names.join(", "));

    let sse_url = server.serve_sse(listener, public_base);
    println!("MCP SSE server 已启动 ✅");
    println!("客户端连接入口: {sse_url}");
    println!("按 Ctrl+C 停止。");

    // 5. 保持进程存活(接收循环在后台任务里运行)
    std::future::pending::<()>().await;
}

/// 通过 tools/list 拉取实际注册的工具名(与客户端看到的一致)。
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
