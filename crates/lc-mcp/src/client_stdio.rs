//! Official MCP stdio client (B1, 0.22.4).
//!
//! High-level client over [`StdioTransport`]: performs the standard MCP
//! handshake (`initialize` → `notifications/initialized`), records the
//! negotiated [`ProtocolInfo`], and exposes the same tool surface as
//! [`crate::StatelessMcpClient`] (`list_tools` / `call_tool`) so callers can
//! drive local subprocess MCP servers (Claude Desktop style: filesystem, git,
//! database, ... servers distributed as executables).
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use lc_mcp::{StdioCommand, StdioMcpClient};
//!
//! let client = StdioMcpClient::connect(
//!     StdioCommand::new("uvx").args(["mcp-server-fetch"]),
//! )
//! .await?;
//! for tool in client.list_tools().await? {
//!     println!("{}: {}", tool.name, tool.description);
//! }
//! client.close().await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use serde_json::{json, Value};

use crate::protocol::{
    negotiate_protocol_version, JsonRpcId, MCPError, ProtocolInfo, VersionPolicy, MCP_VERSION,
};
use crate::transport::stdio::{StdioCommand, StdioTransport};
use crate::types::{MCPToolDefinition, MCPToolResult};

/// Default per-request timeout for stdio sessions.
const STDIO_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// MCP client driving a local subprocess over the official stdio transport.
#[derive(Debug)]
pub struct StdioMcpClient {
    transport: StdioTransport,
    info: ProtocolInfo,
    /// Full `initialize` result (capabilities / serverInfo), for callers that
    /// need server-declared capability details.
    initialize_result: Value,
    request_timeout: Duration,
    /// T10: optional `tools/call` OTel instrumentation.
    #[cfg(feature = "opentelemetry")]
    instrumentation: Option<std::sync::Arc<crate::instrument::McpInstrumentation>>,
}

impl StdioMcpClient {
    /// Spawns `command` and completes the MCP handshake with the default
    /// version policy ([`VersionPolicy::Degrade`]) and a 60s request timeout.
    pub async fn connect(command: StdioCommand) -> Result<Self, MCPError> {
        Self::connect_with(command, VersionPolicy::default(), STDIO_REQUEST_TIMEOUT).await
    }

    /// Spawns `command` and completes the MCP handshake with an explicit
    /// version policy and per-request timeout.
    ///
    /// Under [`VersionPolicy::Reject`], a server declaring a protocol version
    /// outside [`crate::protocol::SUPPORTED_PROTOCOL_VERSIONS`] fails the
    /// connection (-32005) and the child is shut down.
    pub async fn connect_with(
        command: StdioCommand,
        policy: VersionPolicy,
        request_timeout: Duration,
    ) -> Result<Self, MCPError> {
        let transport = StdioTransport::spawn(command)?;

        let init_params = json!({
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "langchainrust-mcp-client",
                "version": env!("CARGO_PKG_VERSION"),
            },
        });

        let result = match transport
            .request("initialize", Some(init_params), request_timeout)
            .await
        {
            Ok(result) => result,
            Err(e) => {
                transport.shutdown().await;
                return Err(e);
            }
        };

        let server_version = result
            .get("protocolVersion")
            .and_then(|v| v.as_str())
            .unwrap_or(MCP_VERSION)
            .to_string();
        let (negotiated, supported) = match negotiate_protocol_version(&server_version, policy) {
            Ok(x) => x,
            Err(e) => {
                transport.shutdown().await;
                return Err(e);
            }
        };

        // The initialized notification has no response per the MCP spec.
        if let Err(e) = transport.notify("notifications/initialized", None).await {
            transport.shutdown().await;
            return Err(e);
        }

        Ok(Self {
            transport,
            info: ProtocolInfo {
                requested: MCP_VERSION.to_string(),
                server_version,
                negotiated,
                supported,
            },
            initialize_result: result,
            request_timeout,
            #[cfg(feature = "opentelemetry")]
            instrumentation: None,
        })
    }

    /// Attaches OpenTelemetry `tools/call` instrumentation (T10).
    #[cfg(feature = "opentelemetry")]
    pub fn with_instrumentation(
        mut self,
        instrumentation: std::sync::Arc<crate::instrument::McpInstrumentation>,
    ) -> Self {
        self.instrumentation = Some(instrumentation);
        self
    }

    /// Protocol negotiation result of the completed handshake.
    pub fn protocol_info(&self) -> &ProtocolInfo {
        &self.info
    }

    /// Full `initialize` result: `protocolVersion`, `capabilities`,
    /// `serverInfo`.
    pub fn initialize_result(&self) -> &Value {
        &self.initialize_result
    }

    /// Server-declared capabilities (empty object when absent).
    pub fn capabilities(&self) -> &Value {
        self.initialize_result
            .get("capabilities")
            .unwrap_or(&Value::Null)
    }

    /// Sends a raw JSON-RPC request, returning the result payload.
    ///
    /// Public for methods without a typed wrapper; honors the configured
    /// per-request timeout.
    pub async fn send(&self, method: &str, params: Option<Value>) -> Result<Value, MCPError> {
        self.transport
            .request(method, params, self.request_timeout)
            .await
    }

    /// `ping` liveness probe.
    pub async fn ping(&self) -> Result<(), MCPError> {
        self.transport
            .request("ping", None, self.request_timeout)
            .await
            .map(|_| ())
    }

    /// B8: best-effort `notifications/cancelled` for an in-flight request id.
    /// The `requestId` is serialized as-is (numeric ids stay numeric) so the
    /// server can match it against its `requestId → Notify` table. Delivery is
    /// best-effort — a server that already responded may drop it.
    pub async fn cancel(&self, request_id: impl Into<JsonRpcId>) -> Result<(), MCPError> {
        let rid = serde_json::to_value(request_id.into())
            .map_err(|e| MCPError::new(-32603, format!("failed to encode request id: {e}")))?;
        self.transport
            .notify("notifications/cancelled", Some(json!({ "requestId": rid })))
            .await
    }

    /// `tools/list` (uncached), mirroring the stateless client.
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send("tools/list", None).await?;
        let tools_value = result
            .get("tools")
            .ok_or_else(|| MCPError::new(-1, "tools/list response missing 'tools' field"))?;
        let tools: Vec<MCPToolDefinition> = serde_json::from_value(tools_value.clone())
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool list: {e}")))?;
        Ok(tools)
    }

    /// `tools/call`, mirroring the stateless client (emits a `tools/call
    /// {name}` OTel client span when instrumentation is attached — T10).
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        #[cfg(feature = "opentelemetry")]
        if let Some(instr) = &self.instrumentation {
            return instr
                .record_tool_call(
                    name,
                    &arguments,
                    crate::instrument::NETWORK_TRANSPORT_PIPE,
                    &self.info.negotiated,
                    None,
                    self.call_tool_direct(name, arguments.clone()),
                )
                .await;
        }
        self.call_tool_direct(name, arguments).await
    }

    async fn call_tool_direct(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool result: {e}")))
    }

    /// Shuts the session down: closes stdin, waits for the child, kills it if
    /// needed. Idempotent.
    pub async fn close(&self) -> Result<(), MCPError> {
        self.transport.shutdown().await;
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::tool_client::McpToolClient for StdioMcpClient {
    async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        StdioMcpClient::list_tools(self).await
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        StdioMcpClient::call_tool(self, name, arguments).await
    }
}
