//! MCP 集成测试 —— 进程内打通(无子进程 / 网络),默认运行。

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
    #[allow(clippy::type_complexity)] // 刻意写死签名做契约断言
    let _assert_type: fn(
        Arc<dyn langchainrust::BaseAgent>,
        Vec<Arc<dyn BaseTool>>,
    ) -> AgentExecutor = AgentExecutor::new;
    let _ = _assert_type;
}
