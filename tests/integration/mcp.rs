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
#[tokio::test]
#[ignore = "需要 npx 和 @anthropic/mcp-server-filesystem"]
async fn test_mcp_as_tools() {
    let client = MCPClient::connect(filesystem_config())
        .await
        .expect("连接失败");
    client.list_tools().await.expect("列出工具失败");
    let tools: Vec<_> = client.as_tools().await;
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
