//! MCP Client - connects to an MCP Server to fetch and invoke tools

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::{broadcast, Mutex};
use tokio::time::{timeout, Duration};

use super::protocol::{
    MCPError, MCPRequest, ProtocolInfo, VersionPolicy, MCP_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
use super::stream::{parse_partial_notification, PartialContent, ToolStream};
use super::tool_adapter::MCPToolAdapter;
use super::transport::{MCPEvent, MCPTransport, SseTransport, StdioTransport};
use super::types::{MCPConfig, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// Default timeout for MCP client connection and initialization (30 seconds).
const MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Tool-list cache TTL: beyond this duration the cache is stale and the next `as_tools` refetches (P1-8).
const TOOLS_CACHE_TTL: Duration = Duration::from_secs(60);

struct MCPClientInner {
    transport: Box<dyn MCPTransport + Send + Sync>,
    tools: Mutex<Vec<MCPToolDefinition>>,
    /// Time of the last successful tool-list fetch; `None` = never fetched or invalidated (P1-8).
    tools_fetched_at: Mutex<Option<Instant>>,
    request_id: AtomicU64,
    /// Streaming tool-output broadcast (P2-9): the background event listener turns
    /// `notifications/tool_partial` into [`PartialContent`] broadcasts; `subscribe_tool_stream` filters by tool name.
    partial_tx: broadcast::Sender<PartialContent>,
    /// Protocol-version negotiation policy (P2-10): degrade or reject when the Server version is unsupported.
    version_policy: VersionPolicy,
    /// Protocol-version negotiation result locked in after the handshake (P2-10); `None` before the handshake.
    protocol_info: RwLock<Option<ProtocolInfo>>,
}

/// MCP Client - connects to an MCP Server to obtain tool capabilities
pub struct MCPClient {
    inner: Arc<MCPClientInner>,
}

impl Clone for MCPClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl MCPClient {
    /// Connects to an MCP Server
    ///
    /// After establishing the transport connection, sends the MCP `initialize` handshake request,
    /// then the `notifications/initialized` notification, completing the protocol handshake.
    /// Protocol-version negotiation uses the default degrade policy ([`VersionPolicy::Degrade`]): when
    /// the Server version is unsupported it degrades to this implementation's version. For strict mode
    /// see `connect_with_policy`.
    pub async fn connect(config: MCPConfig) -> Result<Self, MCPError> {
        Self::connect_with_policy(config, VersionPolicy::Degrade).await
    }

    /// Connects with a specified version-negotiation policy (P2-10).
    ///
    /// With `VersionPolicy::Reject`, a Server declaring a version outside the supported list fails the
    /// handshake and refuses the connection, no silent degradation.
    pub async fn connect_with_policy(
        config: MCPConfig,
        policy: VersionPolicy,
    ) -> Result<Self, MCPError> {
        // Wrap the entire connect + initialize flow in a timeout
        timeout(MCP_CONNECT_TIMEOUT, Self::connect_inner(config, policy))
            .await
            .map_err(|_| {
                MCPError::new(
                    -1,
                    "MCP connect timeout: handshake not completed within 30 seconds",
                )
            })?
    }

    async fn connect_inner(config: MCPConfig, policy: VersionPolicy) -> Result<Self, MCPError> {
        let transport: Box<dyn MCPTransport + Send + Sync> = match &config {
            MCPConfig::Stdio { .. } => Box::new(StdioTransport::new(&config).await?),
            MCPConfig::Sse { .. } => Box::new(SseTransport::new(&config)?),
        };
        Self::with_transport_policy(transport, policy).await
    }

    /// Constructs the client with a custom transport and completes the handshake (P2-6).
    ///
    /// For in-process embedding: e.g. plug [`InMemoryTransport`](crate::InMemoryTransport)
    /// directly into a local `MCPServer`, bypassing subprocess / network; or inject a custom transport layer.
    /// Like `connect`, completes `reconnect → initialize handshake → listen for pushes`; the caller
    /// gets a ready-to-use client.
    pub async fn with_transport(
        transport: Box<dyn MCPTransport + Send + Sync>,
    ) -> Result<Self, MCPError> {
        Self::with_transport_policy(transport, VersionPolicy::Degrade).await
    }

    /// Custom-transport construction with a specified version-negotiation policy (P2-10).
    ///
    /// Same as `with_transport`, but negotiates the protocol version with the given policy.
    pub async fn with_transport_policy(
        transport: Box<dyn MCPTransport + Send + Sync>,
        policy: VersionPolicy,
    ) -> Result<Self, MCPError> {
        let client = Self {
            inner: Arc::new(MCPClientInner {
                transport,
                tools: Mutex::new(Vec::new()),
                tools_fetched_at: Mutex::new(None),
                request_id: AtomicU64::new(1),
                partial_tx: broadcast::channel(64).0,
                version_policy: policy,
                protocol_info: RwLock::new(None),
            }),
        };

        // Establish the transport connection first (SSE lazily starts the background read loop / the
        // Stdio subprocess is ready / the in-process transport is ready), so the first initialize
        // request doesn't race the connection setup.
        client.inner.transport.reconnect().await?;
        // MCP protocol handshake: initialize + initialized notification (negotiate the protocol version)
        client.handshake().await?;
        // P1-8: listen to server pushes in the background; invalidate the tool cache on `tools/list_changed`.
        client.spawn_event_listener();

        Ok(client)
    }

    fn next_id(&self) -> u64 {
        self.inner.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// MCP protocol handshake: initialize + initialized notification, negotiating the protocol version (P2-10).
    ///
    /// The requested version is this library's current [`MCP_VERSION`]; a Server response inside the
    /// supported list is adopted, otherwise it degrades or rejects per [`VersionPolicy`](crate::VersionPolicy).
    /// The negotiated result is locked into `protocol_info`, read via `protocol_info()` / `protocol_version()`.
    /// Reused by connection setup and by disconnect-reconnect (re-handshake).
    async fn handshake(&self) -> Result<(), MCPError> {
        let requested = MCP_VERSION.to_string();
        let init_params = json!({
            "protocolVersion": requested,
            "capabilities": {},
            "clientInfo": {
                "name": "langchainrust-mcp-client",
                "version": "0.3.0"
            }
        });
        let req = MCPRequest::new(self.next_id(), "initialize", Some(init_params));
        let resp = self.inner.transport.request(req).await?;
        let result = resp.into_result()?;

        // Version negotiation: a Server-declared version ∈ supported list → adopt it; otherwise degrade / reject per policy.
        let server_version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(MCP_VERSION)
            .to_string();
        let supported = SUPPORTED_PROTOCOL_VERSIONS.contains(&server_version.as_str());
        let negotiated = if supported {
            server_version.clone()
        } else {
            match self.inner.version_policy {
                VersionPolicy::Degrade => MCP_VERSION.to_string(),
                VersionPolicy::Reject => {
                    return Err(MCPError::new(
                        -32600,
                        format!("unsupported MCP protocol version: {server_version}"),
                    ));
                }
            }
        };
        *self
            .inner
            .protocol_info
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(ProtocolInfo {
            requested,
            server_version,
            negotiated,
            supported,
        });

        // Send the initialized notification (no id, no response awaited)
        self.inner
            .transport
            .notify("notifications/initialized", None)
            .await?;
        Ok(())
    }

    /// The locked-in protocol-version negotiation result (P2-10); `None` before the handshake.
    pub fn protocol_info(&self) -> Option<ProtocolInfo> {
        self.inner
            .protocol_info
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The protocol version actually in effect (P2-10); `None` before the handshake.
    pub fn protocol_version(&self) -> Option<String> {
        self.protocol_info().map(|p| p.negotiated)
    }

    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value, MCPError> {
        let req = MCPRequest::new(self.next_id(), method, params);
        let resp = self.inner.transport.request(req.clone()).await;
        match resp {
            Ok(r) => r.into_result(),
            Err(e) if e.is_connection_lost() => {
                // P0-2: connection lost → reconnect + re-handshake + refresh the tool cache + retry once
                log::warn!(
                    "MCP connection lost, reconnecting and re-handshaking before retry for {}",
                    req.method
                );
                self.reconnect_and_retry(req).await
            }
            Err(e) => Err(e),
        }
    }

    /// Disconnect recovery: wait for the transport to reconnect → re-handshake → clear the tool cache
    /// (the tool list may change after a subprocess restart) → retry the original request once.
    async fn reconnect_and_retry(&self, req: MCPRequest) -> Result<Value, MCPError> {
        self.inner.transport.reconnect().await?;
        self.handshake().await?;
        self.invalidate_tools_cache().await;
        let resp = self.inner.transport.request(req).await?;
        resp.into_result()
    }

    /// Invalidates the tool cache: clears the list and resets the fetch time (P1-8).
    ///
    /// The next `as_tools` / `list_tools` refetches instead of continuing with the stale list.
    /// Callers are all in async contexts (tokio Mutex), so the await holds the lock briefly.
    async fn invalidate_tools_cache(&self) {
        *self.inner.tools.lock().await = Vec::new();
        *self.inner.tools_fetched_at.lock().await = None;
    }

    /// P1-8 background listener: subscribes to transport push events; invalidates the tool cache on
    /// `tools/list_changed` or disconnect, so the agent never uses a stale tool list.
    fn spawn_event_listener(&self) {
        let client = self.clone();
        let mut rx = self.inner.transport.subscribe_events();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(MCPEvent::Message { method, params })
                        if method == "notifications/tool_partial" =>
                    {
                        // P2-9 streaming tool output: the server pushes incremental chunks as it runs;
                        // forward them to subscribe_tool_stream subscribers; malformed chunks are dropped silently.
                        if let Some(partial) = parse_partial_notification(params) {
                            let _ = client.inner.partial_tx.send(partial);
                        }
                    }
                    Ok(MCPEvent::Message { method, .. })
                        if method == "notifications/tools/list_changed" =>
                    {
                        log::info!("received tools/list_changed, invalidating tool cache");
                        client.invalidate_tools_cache().await;
                    }
                    Ok(MCPEvent::Disconnected) => {
                        // After a disconnect the tool list may have changed; let the reconnect-handshake path refetch it
                        client.invalidate_tools_cache().await;
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Backlogged events are dropped: invalidate conservatively to avoid using a stale list
                        client.invalidate_tools_cache().await;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    /// Fetches the available tool list (`tools/list`)
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send_request("tools/list", None).await?;
        let tools_value = result
            .get("tools")
            .ok_or_else(|| MCPError::new(-1, "tools/list response missing 'tools' field"))?;
        let tools: Vec<MCPToolDefinition> = serde_json::from_value(tools_value.clone())
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool list: {}", e)))?;
        *self.inner.tools.lock().await = tools.clone();
        // P1-8: record the fetch time so as_tools can judge cache freshness.
        *self.inner.tools_fetched_at.lock().await = Some(Instant::now());
        Ok(tools)
    }

    /// Invokes a tool (`tools/call`)
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send_request("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool result: {}", e)))
    }

    /// Closes the connection
    pub async fn close(&self) -> Result<(), MCPError> {
        self.inner.transport.close().await
    }

    /// Subscribes to transport push events (P2-4).
    ///
    /// Used by per-tool timeout (P2-4), streaming tool output (P2-9), etc. to listen to server
    /// pushes: `notifications/progress` / disconnect / tool changes, etc.
    pub fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
        self.inner.transport.subscribe_events()
    }

    /// Subscribes to the streaming incremental output of one tool (P2-9).
    ///
    /// A long-running tool "pushes as it runs": the server splits partial results into chunks, pushed via
    /// `notifications/tool_partial`. The returned [`ToolStream`] only delivers increments belonging to
    /// `tool`; pushes from other tools are filtered. With P1-7, each chunk carries its own
    /// [`MCPContent`](crate::MCPContent) (text / image / resource).
    ///
    /// Note: the broadcast channel only delivers pushes made *after* subscribing — **subscribe before
    /// calling the tool**; when pushes are too fast and frames are dropped, [`ToolStream::next`] returns
    /// [`ToolStreamError::Lagged`](crate::ToolStreamError::Lagged).
    pub fn subscribe_tool_stream(&self, tool: &str) -> ToolStream {
        ToolStream::new(self.inner.partial_tx.subscribe(), tool.to_string())
    }

    /// Converts to a `BaseTool` list (for use by Agents)
    ///
    /// P0-3: auto-discovery — if the tool cache is empty, automatically calls `list_tools` instead of
    /// silently returning an empty list; failures are exposed explicitly via `Result`.
    ///
    /// P1-8: cache with TTL — refetches when the cache is empty or not refreshed within `TOOLS_CACHE_TTL`;
    /// the cache is invalidated immediately on a `tools/list_changed` notification (background listener)
    /// or a disconnect.
    pub async fn as_tools(&self) -> Result<Vec<Arc<dyn BaseTool>>, MCPError> {
        let tools = {
            let cached = self.inner.tools.lock().await;
            let fetched = self.inner.tools_fetched_at.lock().await;
            let fresh = matches!(fetched.as_ref(), Some(t) if t.elapsed() < TOOLS_CACHE_TTL);
            if cached.is_empty() || !fresh {
                drop(cached);
                drop(fetched);
                self.list_tools().await?
            } else {
                cached.clone()
            }
        };
        Ok(tools
            .into_iter()
            .map(|def| Arc::new(MCPToolAdapter::new(self.clone(), def)) as Arc<dyn BaseTool>)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::MCPResponse;
    use crate::test_support::{start_fake_sse_server, PostMode};

    #[tokio::test]
    async fn test_connect_invalid_stdio_command() {
        let config = MCPConfig::stdio("nonexistent_command_xyz_zzz", vec![]);
        let result = MCPClient::connect(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_as_tools_uses_cache_within_ttl() {
        // P1-8: a Quiet-mode server sends no list_changed → the second as_tools within the TTL hits the
        // cache and never calls tools/list (tools_list_count stays 1).
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let tools = client
            .as_tools()
            .await
            .expect("initial fetch should succeed");
        assert_eq!(tools.len(), 1);
        assert_eq!(server.tools_list_count.load(Ordering::SeqCst), 1);

        let tools2 = client.as_tools().await.expect("cache hit should succeed");
        assert_eq!(tools2.len(), 1);
        assert_eq!(
            server.tools_list_count.load(Ordering::SeqCst),
            1,
            "cache hit within TTL, tools/list should not be called again"
        );
    }

    #[tokio::test]
    async fn test_tools_cache_invalidated_on_list_changed() {
        // P1-8: the server pushes tools/list_changed → the background listener invalidates the cache →
        // the next as_tools refetches (tools_list_count >= 2). Poll until then to avoid timing flakiness.
        let server = start_fake_sse_server(PostMode::NotifyChanged).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let tools = client
            .as_tools()
            .await
            .expect("initial fetch should succeed");
        assert_eq!(tools.len(), 1);

        timeout(Duration::from_secs(5), async {
            loop {
                // Each poll goes through as_tools: an invalidated cache triggers a refetch
                let _ = client.as_tools().await;
                if server.tools_list_count.load(Ordering::SeqCst) >= 2 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("list_changed should invalidate cache and refetch tools/list");
    }

    #[tokio::test]
    async fn test_connect_list_and_call_tool_via_sse_push_responses() {
        // F4 acceptance: the server replies 202 to POST and JSON-RPC responses arrive via SSE
        // `event: message` → connect(initialize)/list_tools/call_tool all work. The POST body is always
        // empty, so results can only come from SSE pushes — success proves the id-correlated push path works.
        let server = start_fake_sse_server(PostMode::PushResponse).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connect (initialize) should succeed via SSE-pushed response");

        let tools = client
            .list_tools()
            .await
            .expect("list_tools should succeed via SSE-pushed response");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = client
            .call_tool("echo", json!({"msg": "hi"}))
            .await
            .expect("call_tool should succeed via SSE-pushed response");
        assert!(!result.is_error, "server tool should not error");
    }

    // ---- P2-10 protocol-version negotiation tests ----

    /// Stub transport for version-negotiation tests: initialize returns a configurable `protocolVersion`,
    /// other methods are empty, avoiding SSE timing and a real server.
    struct StubVersionTransport {
        server_version: String,
    }

    #[async_trait::async_trait]
    impl MCPTransport for StubVersionTransport {
        async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
            if req.method == "initialize" {
                return Ok(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": self.server_version,
                        "capabilities": {},
                        "serverInfo": { "name": "stub", "version": "0.0.1" }
                    })),
                    error: None,
                });
            }
            Ok(MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: Some(json!({})),
                error: None,
            })
        }
        async fn notify(&self, _m: &str, _p: Option<Value>) -> Result<(), MCPError> {
            Ok(())
        }
        async fn close(&self) -> Result<(), MCPError> {
            Ok(())
        }
        fn is_connected(&self) -> bool {
            true
        }
        async fn reconnect(&self) -> Result<(), MCPError> {
            Ok(())
        }
        fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
            broadcast::channel(4).1
        }
    }

    /// A version inside the supported list: negotiation succeeds, supported=true, locking the requested version.
    #[tokio::test]
    async fn test_handshake_negotiates_supported_version() {
        let client = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: MCP_VERSION.to_string(),
            }),
            VersionPolicy::Degrade,
        )
        .await
        .expect("supported version should handshake successfully");

        let info = client
            .protocol_info()
            .expect("negotiation result should be available after handshake");
        assert!(info.supported, "2024-11-05 should be in the supported list");
        assert_eq!(info.negotiated, MCP_VERSION);
        assert_eq!(info.server_version, MCP_VERSION);
        assert_eq!(client.protocol_version().as_deref(), Some(MCP_VERSION));
    }

    /// Degrade policy: the Server declares an unknown version → supported=false, degrades to this implementation's version.
    #[tokio::test]
    async fn test_handshake_degrades_unsupported_version() {
        let client = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: "2099-01-01".to_string(),
            }),
            VersionPolicy::Degrade,
        )
        .await
        .expect("degrade policy should tolerate unknown version");

        let info = client
            .protocol_info()
            .expect("negotiation result should be available after handshake");
        assert!(
            !info.supported,
            "2099-01-01 should not be in the supported list"
        );
        assert_eq!(info.server_version, "2099-01-01");
        assert_eq!(
            info.negotiated, MCP_VERSION,
            "should lock to this implementation version after degrade"
        );
        assert_eq!(client.protocol_version().as_deref(), Some(MCP_VERSION));
    }

    /// Reject policy: the Server declares an unknown version → handshake fails, connection refused.
    #[tokio::test]
    async fn test_handshake_rejects_unsupported_version() {
        let err = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: "2099-01-01".to_string(),
            }),
            VersionPolicy::Reject,
        )
        .await
        .err()
        .expect("reject policy should error on unknown version");

        assert!(
            err.to_string().contains("unsupported MCP protocol version"),
            "error message should state the version is unsupported: {err}"
        );
    }
}
