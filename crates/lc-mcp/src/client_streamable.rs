//! Official MCP Streamable HTTP client (B1, 0.22.4).
//!
//! High-level client over [`StreamableHttpTransport`]: performs the standard
//! MCP handshake (`initialize` → `notifications/initialized`) over the
//! Streamable HTTP transport, tracks the server-assigned `Mcp-Session-Id`,
//! and exposes the same tool surface as [`crate::StdioMcpClient`] /
//! [`crate::StatelessMcpClient`] (`list_tools` / `call_tool`) so remote
//! Streamable HTTP servers (the official TS/Python SDK deployment shape) drop
//! into [`crate::MCPToolAdapter::from_client`].
//!
//! Both response shapes are handled transparently: direct `application/json`
//! bodies and `text/event-stream` responses with interleaved notifications.
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use lc_mcp::{StreamableMcpClient, StaticBearerToken};
//! use std::sync::Arc;
//!
//! let client = StreamableMcpClient::connect("https://example.com/mcp").await?;
//! for tool in client.list_tools().await? {
//!     println!("{}: {}", tool.name, tool.description);
//! }
//! let _authed = StreamableMcpClient::connect_with_token_provider(
//!     "https://example.com/mcp",
//!     Arc::new(StaticBearerToken("token".into())),
//! )
//! .await?;
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde_json::{json, Value};

use crate::protocol::{
    negotiate_protocol_version, MCPError, MCPRequest, ProtocolInfo, VersionPolicy,
    MCP_ERROR_REQUEST_TIMEOUT, MCP_VERSION,
};
use crate::transport::streamable_http::StreamableHttpTransport;
use crate::types::{MCPToolDefinition, MCPToolResult};

/// Default per-request timeout for Streamable HTTP sessions.
const STREAMABLE_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// MCP client driving a remote server over Streamable HTTP.
pub struct StreamableMcpClient {
    transport: Arc<StreamableHttpTransport>,
    info: RwLock<ProtocolInfo>,
    initialize_result: RwLock<Value>,
    policy: VersionPolicy,
    request_timeout: Duration,
    next_id: AtomicU64,
}

impl std::fmt::Debug for StreamableMcpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamableMcpClient")
            .field("url", &self.transport.url())
            .field("session_id", &self.transport.session_id())
            .field("info", &self.info)
            .field("request_timeout", &self.request_timeout)
            .finish_non_exhaustive()
    }
}

impl StreamableMcpClient {
    /// Completes the MCP handshake against an unauthenticated endpoint with
    /// the default version policy ([`VersionPolicy::Degrade`]) and a 60s
    /// per-request timeout.
    pub async fn connect(url: impl Into<String>) -> Result<Self, MCPError> {
        let transport = Arc::new(StreamableHttpTransport::new(url));
        Self::handshake(
            transport,
            VersionPolicy::default(),
            STREAMABLE_REQUEST_TIMEOUT,
        )
        .await
    }

    /// Completes the handshake with an explicit version policy and per-request
    /// timeout (unauthenticated endpoint).
    pub async fn connect_with(
        url: impl Into<String>,
        policy: VersionPolicy,
        request_timeout: Duration,
    ) -> Result<Self, MCPError> {
        let transport = Arc::new(StreamableHttpTransport::new(url));
        Self::handshake(transport, policy, request_timeout).await
    }

    /// Completes the handshake against an OAuth 2.1-protected endpoint using
    /// the supplied bearer-token provider (401 → invalidate/retry once).
    pub async fn connect_with_token_provider(
        url: impl Into<String>,
        provider: Arc<dyn crate::oauth2::BearerTokenProvider>,
    ) -> Result<Self, MCPError> {
        Self::connect_token(
            url,
            provider,
            VersionPolicy::default(),
            STREAMABLE_REQUEST_TIMEOUT,
        )
        .await
    }

    /// Completes the handshake with a token provider, an explicit version
    /// policy and a per-request timeout.
    pub async fn connect_token(
        url: impl Into<String>,
        provider: Arc<dyn crate::oauth2::BearerTokenProvider>,
        policy: VersionPolicy,
        request_timeout: Duration,
    ) -> Result<Self, MCPError> {
        let transport = Arc::new(StreamableHttpTransport::with_token_provider(url, provider));
        Self::handshake(transport, policy, request_timeout).await
    }

    async fn handshake(
        transport: Arc<StreamableHttpTransport>,
        policy: VersionPolicy,
        request_timeout: Duration,
    ) -> Result<Self, MCPError> {
        let (info, result) = initialize(&transport, policy, request_timeout).await?;
        Ok(Self {
            transport,
            info: RwLock::new(info),
            initialize_result: RwLock::new(result),
            policy,
            request_timeout,
            next_id: AtomicU64::new(2), // initialize used id 1
        })
    }

    /// Protocol negotiation result of the completed handshake (or the latest
    /// [`StreamableMcpClient::reconnect`]).
    pub fn protocol_info(&self) -> ProtocolInfo {
        self.info.read().unwrap().clone()
    }

    /// Full `initialize` result: `protocolVersion`, `capabilities`,
    /// `serverInfo`.
    pub fn initialize_result(&self) -> Value {
        self.initialize_result.read().unwrap().clone()
    }

    /// Server-declared capabilities (empty object when absent).
    pub fn capabilities(&self) -> Value {
        self.initialize_result()
            .get("capabilities")
            .cloned()
            .unwrap_or(Value::Null)
    }

    /// The server-assigned session id, when the server is stateful.
    pub fn session_id(&self) -> Option<String> {
        self.transport.session_id()
    }

    /// Re-runs the initialize handshake after a session-lost error
    /// ([`crate::MCP_ERROR_SESSION_LOST`]): clears the old
    /// `Mcp-Session-Id`, negotiates a fresh session, and pins its result.
    pub async fn reconnect(&self) -> Result<(), MCPError> {
        self.transport.clear_session();
        let (info, result) = initialize(&self.transport, self.policy, self.request_timeout).await?;
        *self.info.write().unwrap() = info;
        *self.initialize_result.write().unwrap() = result;
        Ok(())
    }

    /// Sends a raw JSON-RPC request, returning the result payload. Honors the
    /// configured per-request timeout.
    pub async fn send(&self, method: &str, params: Option<Value>) -> Result<Value, MCPError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req = MCPRequest::new(id, method, params);
        match tokio::time::timeout(self.request_timeout, self.transport.request(&req)).await {
            Ok(Ok(response)) => response.into_result(),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(MCPError::new(
                MCP_ERROR_REQUEST_TIMEOUT,
                format!(
                    "MCP Streamable HTTP request '{method}' timed out after {:?}",
                    self.request_timeout
                ),
            )),
        }
    }

    /// Sends a JSON-RPC notification (202 accepted; no result).
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), MCPError> {
        self.transport.notify(method, params).await
    }

    /// `ping` liveness probe.
    pub async fn ping(&self) -> Result<(), MCPError> {
        self.send("ping", None).await.map(|_| ())
    }

    /// `tools/list` (uncached).
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send("tools/list", None).await?;
        let tools_value = result
            .get("tools")
            .ok_or_else(|| MCPError::new(-1, "tools/list response missing 'tools' field"))?;
        let tools: Vec<MCPToolDefinition> = serde_json::from_value(tools_value.clone())
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool list: {e}")))?;
        Ok(tools)
    }

    /// `tools/call`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool result: {e}")))
    }

    /// Releases the session. Streamable HTTP has no explicit session-delete
    /// verb: this is a best-effort no-op (the server reaps expired sessions);
    /// kept for API symmetry with the stdio client.
    pub async fn close(&self) -> Result<(), MCPError> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl crate::tool_client::McpToolClient for StreamableMcpClient {
    async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        StreamableMcpClient::list_tools(self).await
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        StreamableMcpClient::call_tool(self, name, arguments).await
    }
}

/// Runs one initialize handshake over `transport` and sends the initialized
/// notification; shared by connect and reconnect.
async fn initialize(
    transport: &StreamableHttpTransport,
    policy: VersionPolicy,
    request_timeout: Duration,
) -> Result<(ProtocolInfo, Value), MCPError> {
    let init_params = json!({
        "protocolVersion": MCP_VERSION,
        "capabilities": {},
        "clientInfo": {
            "name": "langchainrust-mcp-client",
            "version": env!("CARGO_PKG_VERSION"),
        },
    });
    let init_req = MCPRequest::new(1, "initialize", Some(init_params));

    let response = match tokio::time::timeout(request_timeout, transport.request(&init_req)).await {
        Ok(r) => r?,
        Err(_) => {
            return Err(MCPError::new(
                MCP_ERROR_REQUEST_TIMEOUT,
                format!("MCP initialize timed out after {request_timeout:?}"),
            ))
        }
    };
    let result = response.into_result()?;

    let server_version = result
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or(MCP_VERSION)
        .to_string();
    let (negotiated, supported) = negotiate_protocol_version(&server_version, policy)?;

    // The initialized notification has no response (HTTP 202); best-effort
    // like the stdio track.
    transport.notify("notifications/initialized", None).await?;

    let info = ProtocolInfo {
        requested: MCP_VERSION.to_string(),
        server_version,
        negotiated,
        supported,
    };
    Ok((info, result))
}
