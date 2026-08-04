//! MCP Server 示例
//!
//! 展示如何使用 MCP Server 暴露本地工具给外部 host 调用。
//!
//! # 运行
//! ```bash
//! cargo run --example mcp_server
//! ```

use langchainrust::{BaseTool, Calculator};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建工具
    let calculator = Arc::new(Calculator::new()) as Arc<dyn BaseTool>;

    println!("=== MCP Server 示例 ===\n");

    // 展示工具信息(MCP Server 会将这些工具暴露给 host)
    println!("工具名称: {}", calculator.name());
    println!("工具描述: {}", calculator.description());

    // 模拟 MCP 调用
    let input = r#"{"expression": "2 + 3 * 4"}"#;
    let result = calculator.run(input.to_string()).await?;
    println!("\n调用 {}: 结果 = {}", input, result);

    let input2 = r#"{"expression": "100 / 5"}"#;
    let result2 = calculator.run(input2.to_string()).await?;
    println!("调用 {}: 结果 = {}", input2, result2);

    println!("\nMCP Server 可通过 stdio 或 SSE 传输暴露这些工具。");
    println!("Host 端使用 MCP Client 连接后即可调用这些工具。");
    Ok(())
}
