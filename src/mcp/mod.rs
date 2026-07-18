//! MCP (Model Context Protocol) 支持
//!
//! MCP 是 Anthropic 推出的工具协议标准,已成行业事实标准。
//! 本模块提供 MCP Client,可连接任意 MCP Server 获取工具能力,
//! 并将 MCP 工具适配为 `BaseTool` 供 Agent 使用。
//!
//! # 示例
//! ```no_run
//! use langchainrust::mcp::{MCPClient, MCPConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MCPConfig::stdio(
//!     "npx",
//!     vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
//! );
//! let mut client = MCPClient::connect(config).await?;
//! let tools = client.list_tools().await?;
//! println!("MCP 工具数量: {}", tools.len());
//! # Ok(())
//! # }
//! ```

pub mod protocol;
pub mod types;
pub mod transport;
pub mod client;
pub mod tool_adapter;
pub mod server;

pub use protocol::{MCPError, MCPRequest, MCPResponse, MCP_VERSION};
pub use types::{MCPConfig, MCPContent, MCPToolDefinition, MCPToolResult};
pub use transport::{MCPTransport, SseTransport, StdioTransport};
pub use client::MCPClient;
pub use tool_adapter::MCPToolAdapter;
pub use server::MCPServer;
