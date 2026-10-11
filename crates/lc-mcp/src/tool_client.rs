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
use crate::types::{ListToolsResult, MCPToolDefinition, MCPToolResult};

/// The tool-oriented operations every MCP client track supports.
///
/// Deliberately narrow: handshake/session details differ per track and stay on
/// the concrete clients; tool execution is identical.
#[async_trait]
pub trait McpToolClient: Send + Sync {
    /// Lists the tools the server exposes (`tools/list`).
    async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError>;

    /// `tools/list` with cursor pagination (#79).
    ///
    /// Default implementation issues `tools/list` with the cursor and returns
    /// the full envelope including `nextCursor`, so a caller can page through a
    /// large tool set. Concrete clients override `send`-backed pagination; the
    /// fallback here degrades to a single page via `list_tools` when a client
    /// does not implement the cursor-aware path.
    async fn list_tools_page(
        &self,
        _cursor: Option<String>,
    ) -> Result<ListToolsResult, MCPError> {
        let tools = self.list_tools().await?;
        Ok(ListToolsResult {
            tools,
            next_cursor: None,
        })
    }

    /// Invokes one server tool by its server-side name (`tools/call`).
    async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError>;
}
