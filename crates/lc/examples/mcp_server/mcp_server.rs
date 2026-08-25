//! MCP Server example
//!
//! Shows how to expose local tools to an external host with MCP Server.
//!
//! # Run
//! ```bash
//! cargo run --example mcp_server
//! ```

use langchainrust::{BaseTool, Calculator};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tool
    let calculator = Arc::new(Calculator::new()) as Arc<dyn BaseTool>;

    println!("=== MCP Server example ===\n");

    // Show the tool info (MCP Server will expose these tools to the host)
    println!("Tool name: {}", calculator.name());
    println!("Tool description: {}", calculator.description());

    // Simulate an MCP call
    let input = r#"{"expression": "2 + 3 * 4"}"#;
    let result = calculator.run(input.to_string()).await?;
    println!("\nCall {}: result = {}", input, result);

    let input2 = r#"{"expression": "100 / 5"}"#;
    let result2 = calculator.run(input2.to_string()).await?;
    println!("Call {}: result = {}", input2, result2);

    println!("\nMCP Server can expose these tools over stdio or SSE transport.");
    println!("A host connects with an MCP Client and can then call these tools.");
    Ok(())
}
