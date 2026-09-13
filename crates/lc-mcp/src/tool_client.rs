//! Common tool-calling surface across MCP transport tracks (B1, 0.22.4).
//!
//! Every MCP client — the self-contained stateless HTTP track
//! ([`crate::StatelessMcpClient`]) and the official handshake tracks
//! ([`crate::StdioMcpClient`] today, Streamable HTTP next) — can list and call
//! tools. [`McpToolClient`] is the narrow shared surface the rest of the
//! framework programs against, so adapters, timeouts and gateways need no
//! knowledge of which transport a tool lives on.

use async_trait::async_trait;
use serde_json::Value;

use crate::protocol::MCPError;
use crate::types::{MCPToolDefinition, MCPToolResult};

/// The tool-oriented operations every MCP client track supports.
///
/// Deliberately narrow: handshake/session details differ per track and stay on
/// the concrete clients; tool execution is identical.
#[async_trait]
pub trait McpToolClient: Send + Sync {
    /// Lists the tools the server exposes (`tools/list`).
    async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError>;

    /// Invokes one server tool by its server-side name (`tools/call`).
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError>;
}
