//! MCP 集成测试 - 需要真实 MCP Server
//!
//! 测试前确保:
//! - Node.js / npx 已安装
//! - 可访问 @anthropic/mcp-server-filesystem(npx 会自动下载)
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_mcp -- --ignored
//! ```

use langchainrust::mcp::{MCPClient, MCPConfig};
use serde_json::json;

fn filesystem_config() -> MCPConfig {
    // Windows 上 npx 是 npx.cmd,需要加后缀
    let npx_cmd = if cfg!(target_os = "windows") {
        "npx.cmd"
    } else {
        "npx"
    };
    MCPConfig::stdio(
        npx_cmd,
        vec![
            "@modelcontextprotocol/server-filesystem".to_string(),
            std::env::temp_dir().to_string_lossy().to_string(),
        ],
    )
}

/// 测试 Stdio 传输连接 filesystem server 并列出工具
#[tokio::test]
#[ignore = "需要 npx 和 @anthropic/mcp-server-filesystem"]
async fn test_mcp_stdio_list_tools() {
    let client = MCPClient::connect(filesystem_config())
        .await
        .expect("连接 MCP Server 失败");
    let tools = client.list_tools().await.expect("列出工具失败");
    assert!(!tools.is_empty(), "filesystem server 应暴露工具");
    println!("工具数量: {}", tools.len());
    for t in &tools {
        println!("  - {}: {}", t.name, t.description);
    }
    client.close().await.unwrap();
}

/// 测试 as_tools 转换为 BaseTool 列表
///
/// P0-3: `as_tools` 自动发现工具,无需先手动调用 `list_tools`。
#[tokio::test]
#[ignore = "需要 npx 和 @anthropic/mcp-server-filesystem"]
async fn test_mcp_as_tools() {
    let client = MCPClient::connect(filesystem_config())
        .await
        .expect("连接失败");
    let tools: Vec<_> = client.as_tools().await.expect("as_tools 失败");
    assert!(!tools.is_empty(), "应转换出 BaseTool");
    println!("BaseTool 数量: {}", tools.len());
}

/// 测试调用工具(list_directory)
#[tokio::test]
#[ignore = "需要 npx 和 @anthropic/mcp-server-filesystem"]
async fn test_mcp_call_tool() {
    let client = MCPClient::connect(filesystem_config())
        .await
        .expect("连接失败");
    client.list_tools().await.expect("列出工具失败");
    let result = client
        .call_tool(
            "list_directory",
            json!({"path": std::env::temp_dir().to_string_lossy()}),
        )
        .await;
    if let Ok(r) = result {
        println!("工具结果: {}", r.text());
        assert!(!r.is_error, "list_directory 不应返回错误");
    }
    client.close().await.unwrap();
}

/// P2-6: Agent 工具从 MCP server 加载 —— 进程内打通。
///
/// 真实 `MCPServer` 暴露 Calculator → `MCPClient::with_transport` 进程内连接
/// (无子进程 / 网络)→ `as_tools()` 得到 `Vec<Arc<dyn BaseTool>>`(正是
/// `AgentExecutor` 的 `tools` 参数类型)→ 经 `BaseTool::run` 调用,证明
/// `tools/call` 变成 `BaseTool::run`,Agent 的 `tools` 可直接放 MCP 工具。
#[tokio::test]
async fn test_mcp_tools_load_into_agent_tools_in_process() {
    use langchainrust::mcp::{InMemoryTransport, MCPClient, MCPServer};
    use langchainrust::{AgentExecutor, BaseTool, Calculator};
    use std::sync::Arc;

    // 1. 进程内 MCP Server,暴露 Calculator 工具
    let server = Arc::new(
        MCPServer::new()
            .with_server_info("calc-mcp", "0.1.0")
            .with_tool(Arc::new(Calculator::new()) as Arc<dyn BaseTool>),
    );

    // 2. 进程内连接 + MCP 协议握手(initialize / initialized)
    let client = MCPClient::with_transport(Box::new(InMemoryTransport::new(server)))
        .await
        .expect("进程内连接 MCP Server 失败");

    // 3. as_tools():MCP 工具 → BaseTool 列表(自动发现)
    let mcp_tools: Vec<Arc<dyn BaseTool>> = client.as_tools().await.expect("as_tools 失败");
    assert_eq!(mcp_tools.len(), 1, "MCPServer 应暴露 1 个工具");

    // 4. 工具元数据就位(Agent 据此生成 tool 定义)
    let calc = &mcp_tools[0];
    assert_eq!(calc.name(), "calculator");
    assert!(calc.description().contains("math"));
    assert!(calc.args_schema().is_some());

    // 5. tools/call 变成 BaseTool::run —— 直接经 BaseTool 调用 MCP 工具
    let out = calc
        .run(r#"{"expression": "2 + 3 * 4"}"#.to_string())
        .await
        .expect("调用 MCP 工具失败");
    assert!(out.contains("= 14"), "计算结果应正确, 实际: {}", out);

    // 6. 类型契约:`as_tools()` 产物正是 AgentExecutor::new 的 tools 参数类型,
    //    说明 Agent 的 tools 可直接放 MCP 工具(真实跑 Agent 需配 LLM)。
    fn agent_tools(tools: Vec<Arc<dyn BaseTool>>) -> Vec<Arc<dyn BaseTool>> {
        tools
    }
    let _kept: Vec<Arc<dyn BaseTool>> = agent_tools(mcp_tools);
    // AgentExecutor 构造签名保持(仅为编译期类型契约的锚点)
    let _assert_type: fn(
        Arc<dyn langchainrust::BaseAgent>,
        Vec<Arc<dyn BaseTool>>,
    ) -> AgentExecutor = AgentExecutor::new;
    let _ = _assert_type;
}
