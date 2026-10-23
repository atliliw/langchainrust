//! MCP Server - exposes local `BaseTool`s as an MCP Server for other Hosts (Claude Desktop/Cursor etc.) to call
//!
//! Symmetric to `StatelessMcpClient`: the Client connects to another's Server to use tools, the Server exposes its own
//! tools to others.
//! Supports the `initialize` handshake, `tools/list`, `tools/call`, plus registration-based primitives
//! `resources/*` / `prompts/*` / `completion/complete` (still returning `method_not_found` when unregistered,
//! an honest boundary) and the server→host direction `sampling::create_message` /
//! `elicitation::create` (requires an injected callback).

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, Notify, Semaphore};

use super::auth::TokenValidator;
use super::completion::{CompletionProvider, CompletionRequest};
use super::elicitation::{ElicitationHandler, ElicitationRequest, ElicitationResponse};
use super::prompts::PromptProvider;
use super::protocol::{
    JsonRpcId, MCPError, MCPRequest, MCPResponse, MCP_VERSION, MCP_VERSION_STATELESS,
    SUPPORTED_PROTOCOL_VERSIONS,
};
use super::resources::{ReadResourceResult, ResourceProvider};
use super::tasks::TaskProvider;
use super::sampling::{SamplingHandler, SamplingRequest, SamplingResult};
use super::logging::{message_envelope, LogLevel, LogSink, LOGGING_SETLEVEL};
use super::subscriptions::{
    ResourcePushSink, ResourceSubscriptions, RESOURCES_SUBSCRIBE, RESOURCES_UNSUBSCRIBE,
};
use super::types::{MCPContent, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// 0.22.0 C8 hardening: maximum accepted request body (1 MiB). A larger
/// `Content-Length` is rejected with 413 and a stream exceeding it aborts —
/// without the cap a single connection could grow `buf` unboundedly (OOM).
pub(crate) const HTTP_MAX_BODY_BYTES: usize = 1024 * 1024;
/// 0.22.0 C8 hardening: per-request socket read deadline. A slow-loris
/// connection that never finishes its headers is reaped instead of holding
/// a task + file descriptor forever.
pub(crate) const HTTP_READ_TIMEOUT_SECS: u64 = 30;
/// 0.22.0 C8 hardening: maximum concurrently served connections. Accept waits
/// for a permit instead of spawning unbounded tasks.
pub(crate) const HTTP_MAX_CONCURRENT_CONNECTIONS: usize = 256;
/// M-3: maximum request-header region before the terminating `\r\n\r\n`. The
/// body is already capped by [`HTTP_MAX_BODY_BYTES`]; this bounds the header
/// prefix so a hostile client that never sends the blank line cannot grow the
/// buffer (and the O(n) terminator scan per read) without limit.
pub(crate) const HTTP_MAX_HEADER_BYTES: usize = 64 * 1024;

/// M-2: fail-fast public-bind guard, mirroring lc-a2a's B7 guard. An MCP server
/// without any token validator exposes arbitrary tool invocation to the whole
/// network; a non-loopback bind is refused up front (panic with an actionable
/// message) so the mistake is caught at startup, not after exposure. Loopback
/// binds stay allowed — they are a local/CI-only surface that needs no auth.
fn assert_safe_bind(listener: &tokio::net::TcpListener, has_auth: bool) {
    if let Ok(addr) = listener.local_addr() {
        if !addr.ip().is_loopback() && !has_auth {
            panic!(
                "refusing to serve MCP on non-loopback bind '{addr}' without a token \
                 validator: attach one via with_token_validator(..) for a public deployment \
                 (or bind a loopback address)"
            );
        }
    }
}

/// Private alias factoring the `(session scope, request id) → wakeup` cancellation table
/// so the field type stays readable. The table is only ever touched inside the server.
type CancellationTable = Arc<Mutex<HashMap<(Option<String>, JsonRpcId), Arc<Notify>>>>;

/// MCP Server - exposes a set of `BaseTool`s as MCP tools
pub struct MCPServer {
    tools: Vec<Arc<dyn BaseTool>>,
    server_name: String,
    server_version: String,
    /// 0.22.0 C8: optional request authenticator. When set, `serve_http`
    /// validates the `Authorization: Bearer <token>` header on **every**
    /// request and answers 401 otherwise — the server no longer "deploys
    /// naked". `initialize` is not exempt: the stateless track has no
    /// session, so every request is authenticated.
    auth_validator: Option<Arc<dyn TokenValidator>>,
    /// Optional resource provider (S10): once registered, enables `resources/list` / `resources/read`.
    resources: Option<Arc<dyn ResourceProvider>>,
    /// Optional prompt provider (S10): once registered, enables `prompts/list` / `prompts/get`.
    prompts: Option<Arc<dyn PromptProvider>>,
    /// Optional completion provider (S10): once registered, enables `completion/complete`.
    completion: Option<Arc<dyn CompletionProvider>>,
    /// Optional task provider (T3 Tasks extension): once registered, enables
    /// `tasks/get` / `tasks/cancel` for durable long-running operations.
    tasks: Option<Arc<dyn TaskProvider>>,
    /// 0.27.0 T1: optional resource-subscription registry. Once attached via
    /// [`Self::with_subscription_registry`], it enables `resources/subscribe` /
    /// `resources/unsubscribe` and lets the server push
    /// `notifications/resources/list_changed` to interested subscribers via
    /// [`Self::notify_resource_changed`]. When unregistered, both subscribe
    /// methods still return `method_not_found` (an honest boundary, matching the
    /// other optional primitives).
    subscriptions: Option<Arc<ResourceSubscriptions>>,
    /// 0.27.0 T1: optional delivery sink for resource-change notifications.
    /// When attached (via [`Self::with_push_sink`]), a subscribed-resource
    /// change flows envelope→sink instead of dying at registry boundary: the
    /// sink physically delivers `notifications/resources/list_changed` to each
    /// interested subscriber. Without a sink, [`Self::notify_resource_changed`]
    /// still returns the envelope but delivery stops there (transport layer
    /// must pick it up itself).
    push_sink: Option<Arc<dyn ResourcePushSink>>,
    /// 0.27.0 #79 item ④: optional MCP `logging` sink. Once attached via
    /// [`Self::with_log_sink`], `logging/setLevel` is served and [`Self::emit_log`]
    /// delivers `notifications/message` frames to the sink (below the configured
    /// threshold). When absent, `logging/setLevel` returns `method_not_found`
    /// (honest boundary, matching the other optional primitives).
    log_sink: Option<Arc<dyn LogSink>>,
    /// 0.27.0 #79 item ④: the `logging/setLevel` threshold set by the client
    /// (`None` = no filtering — every [`Self::emit_log`] frame passes). Guarded
    /// by a `Mutex` because `setLevel` mutates it while concurrent requests
    /// read it; the lock is never held across an `await`.
    log_level: std::sync::Mutex<Option<LogLevel>>,
    /// Optional sampling callback (S10, server→host direction): once injected, `create_message` can fire.
    sampling_handler: Option<Arc<dyn SamplingHandler>>,
    /// Optional elicitation callback (S10, server→host direction): once injected, `create_elicitation` can fire.
    elicitation_handler: Option<Arc<dyn ElicitationHandler>>,
    /// B8 cancellation: `(session scope, requestId) → Notify` table. A request
    /// being processed registers a `Notify` keyed by its JSON-RPC id (and, when
    /// run over Streamable HTTP, its session scope) before awaiting the tool;
    /// the client's `notifications/cancelled` for that same scope+id fires it
    /// and the in-flight `tools/call` aborts (the tool future is dropped).
    ///
    /// F1/H3: scoping by `(session, id)` prevents two concurrent sessions that
    /// happen to reuse the same JSON-RPC id from cancelling each other's
    /// in-flight tool call. The HTTP transport passes the `Mcp-Session-Id`;
    /// connection-local transports (stdio) and unscoped callers use `None`,
    /// where ids are unique by construction within the single connection.
    cancellations: CancellationTable,
    /// 0.26.0 P0-1 (BLOCK): per-`tools/call` wall-clock deadline. When set
    /// (`Some(d)`), the in-flight tool future is dropped once it exceeds `d`
    /// and the call returns a hard error observation, so a single hung tool
    /// (a remote dependency, a blocking host call, a runaway loop) cannot
    /// permanently hold a concurrency permit and take the whole server down.
    /// Defaults to `Some(60s)` (hardening); adjust per deployment via
    /// [`Self::with_tool_timeout`] to suit genuinely long-running tools.
    tool_timeout: Option<Duration>,
}

/// Default page size for cursor-paginated `tools/list` (#79). A server exposes
/// every tool across pages; clients that ignore `nextCursor` simply page again.
const TOOL_PAGE_SIZE: usize = 100;

/// Upper bound (bytes) on any single JSON-RPC line read from a stdio stream
/// (stdin / child stdout). A peer sending an arbitrarily long line would
/// otherwise grow the read buffer without limit (OOM); the read is failed
/// cleanly once a single line exceeds this cap (see #H-5).
pub(crate) const MAX_STDIO_LINE_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

/// Reads one newline-terminated line from `reader` into `buf`, bounded so a
/// single frame cannot exceed `max_bytes` (memory DoS guard). The cap is
/// enforced *while* reading, so the buffer is never allowed to grow past the
/// limit, and any offending bytes are consumed so the stream stays aligned for
/// the next line. Returns the number of bytes read (0 = EOF); fails with
/// `InvalidData` when a single line exceeds `max_bytes`.
pub(crate) async fn read_bounded_line<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    buf: &mut String,
    max_bytes: usize,
) -> std::io::Result<usize> {
    buf.clear();
    let mut total = 0usize;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(total); // EOF
        }
        // Consume up to and including the next newline, or the whole buffered
        // chunk when the line is still unterminated.
        let nl = available.iter().position(|&b| b == b'\n');
        let consumed = match nl {
            Some(pos) => pos + 1,
            None => available.len(),
        };
        total += consumed;
        if total > max_bytes {
            // Drain the offending bytes to keep the stream aligned, then fail.
            reader.consume(consumed);
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("stdio line exceeds {max_bytes} bytes"),
            ));
        }
        buf.push_str(&String::from_utf8_lossy(&available[..consumed]));
        reader.consume(consumed);
        if nl.is_some() {
            // Consumed through the terminating newline: the line is complete
            // (even when the newline is the chunk's very last byte).
            return Ok(total);
        }
    }
}

impl MCPServer {
    /// Creates an empty server
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            server_name: "langchainrust-mcp-server".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
            auth_validator: None,
            resources: None,
            prompts: None,
            tasks: None,
            completion: None,
            sampling_handler: None,
            elicitation_handler: None,
            subscriptions: None,
            push_sink: None,
            log_sink: None,
            log_level: std::sync::Mutex::new(None),
            cancellations: Arc::new(Mutex::new(HashMap::new())),
            // 0.26.0 P0-1: bound every `tools/call` by default so a lone hung
            // tool cannot exhaust the concurrency surface (BLOCK). Users with
            // genuinely longer tools opt into a larger/`None` budget.
            tool_timeout: Some(Duration::from_secs(60)),
        }
    }

    /// 0.26.0 P0-1: set the per-`tools/call` deadline. `None` disables the
    /// bound (not recommended — a hung tool then holds a concurrency permit
    /// indefinitely and can strand the whole server). The hardened default is
    /// 60s; raise it for deployments whose tools legitimately run longer.
    pub fn with_tool_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.tool_timeout = timeout.into();
        self
    }

    /// 0.22.0 C8: requires authenticated requests on [`Self::serve_http`] and
    /// [`Self::serve_streamable_http`].
    ///
    /// Every request must carry `Authorization: Bearer <token>` and pass
    /// `validator.validate(token)`; failures get HTTP 401. Recommended for
    /// any non-loopback binding (see the deployment guide).
    pub fn with_token_validator(mut self, validator: Arc<dyn TokenValidator>) -> Self {
        self.auth_validator = Some(validator);
        self
    }

    /// Authenticator shared by the stateless and Streamable HTTP servers.
    pub(crate) fn token_validator(&self) -> Option<&Arc<dyn TokenValidator>> {
        self.auth_validator.as_ref()
    }

    /// Registers a tool
    pub fn with_tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Sets serverInfo (name/version)
    pub fn with_server_info(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.server_name = name.into();
        self.server_version = version.into();
        self
    }

    /// Registers a resource provider, enabling `resources/list` / `resources/read` (S10).
    ///
    /// When unregistered, both primitives still return `method_not_found` (an honest boundary).
    pub fn with_resource_provider(mut self, provider: Arc<dyn ResourceProvider>) -> Self {
        self.resources = Some(provider);
        self
    }

    /// Registers a prompt provider, enabling `prompts/list` / `prompts/get` (S10).
    ///
    /// When unregistered, both primitives still return `method_not_found` (an honest boundary).
    pub fn with_prompt_provider(mut self, provider: Arc<dyn PromptProvider>) -> Self {
        self.prompts = Some(provider);
        self
    }

    /// Registers a completion provider, enabling `completion/complete` (S10).
    ///
    /// When unregistered, the primitive still returns `method_not_found` (an honest boundary).
    /// Registers a task provider, enabling `tasks/get` / `tasks/cancel` (T3 Tasks
    /// extension) for durable long-running operations. When unregistered, both
    /// primitives still return `method_not_found` (an honest boundary).
    pub fn with_task_provider(mut self, provider: Arc<dyn TaskProvider>) -> Self {
        self.tasks = Some(provider);
        self
    }

    /// Registers a completion provider, enabling `completion/complete` (S10).
    ///
    /// When unregistered, the primitive still returns `method_not_found` (an honest boundary).
    pub fn with_completion_provider(mut self, provider: Arc<dyn CompletionProvider>) -> Self {
        self.completion = Some(provider);
        self
    }

    /// 0.27.0 T1: attaches a resource-subscription registry, enabling
    /// `resources/subscribe` / `resources/unsubscribe` and
    /// [`Self::notify_resource_changed`] push. When unregistered, the subscribe
    /// methods still return `method_not_found` (an honest boundary).
    pub fn with_subscription_registry(mut self, subscriptions: Arc<ResourceSubscriptions>) -> Self {
        self.subscriptions = Some(subscriptions);
        self
    }

    /// 0.27.0 T1: attaches a delivery sink so a subscribed-resource change is
    /// physically pushed to each interested subscriber (SSE write, WebSocket
    /// frame, channel send, …). Without a sink the change still produces the
    /// envelope via [`Self::notify_resource_changed`], but the transport must
    /// deliver it itself.
    pub fn with_push_sink(mut self, sink: Arc<dyn ResourcePushSink>) -> Self {
        self.push_sink = Some(sink);
        self
    }

    /// 0.27.0 #79 item ④: attaches an MCP `logging` sink. Once attached,
    /// `logging/setLevel` is served (setting the [`LogLevel`] threshold) and
    /// [`Self::emit_log`] delivers `notifications/message` frames at or above
    /// that threshold to the sink. Without a sink, both the method and the emit
    /// are honest no-ops/boundaries.
    pub fn with_log_sink(mut self, sink: Arc<dyn LogSink>) -> Self {
        self.log_sink = Some(sink);
        self
    }

    /// 0.27.0 T1: emits `notifications/resources/list_changed` for `uri` when
    /// at least one subscriber is watching it, returning the JSON-RPC
    /// notification envelope. When a [`ResourcePushSink`] is attached, the
    /// envelope is also delivered to every interested subscriber via the sink —
    /// the change is physically pushed, not just materialized. Returns `None`
    /// when nothing is subscribed or no registry is attached — an honest no-op,
    /// never traffic to an unwatched resource.
    pub fn notify_resource_changed(&self, uri: &str) -> Option<Value> {
        let subs = self.subscriptions.as_ref()?;
        let envelope = subs.emit_list_changed(uri)?;
        if let Some(sink) = &self.push_sink {
            for subscriber in subs.subscribers_of(uri) {
                sink.deliver(&subscriber, &envelope);
            }
        }
        Some(envelope)
    }

    /// 0.27.0 #79 item ④: emits `notifications/resources/updated` for `uri` when
    /// at least one subscriber is watching it — the *content of a subscribed
    /// resource* changed (distinct from [`Self::notify_resource_changed`], which
    /// is about the resource *list*). Same registry/sink flow: returns the
    /// envelope, and delivers it to every interested subscriber when a
    /// [`ResourcePushSink`] is attached. Returns `None` when unwatched.
    pub fn notify_resource_updated(&self, uri: &str) -> Option<Value> {
        let subs = self.subscriptions.as_ref()?;
        let envelope = subs.emit_updated(uri)?;
        if let Some(sink) = &self.push_sink {
            for subscriber in subs.subscribers_of(uri) {
                sink.deliver(&subscriber, &envelope);
            }
        }
        Some(envelope)
    }

    /// 0.27.0 #79 item ④: emits one `notifications/message` frame at `level`
    /// when a sink is attached and the frame clears the configured
    /// `logging/setLevel` threshold. `logger` is an optional namespace for the
    /// sink to route on; `data` is the free-form payload. Returns the envelope
    /// (for tests/inspection) or `None` when it was filtered out or no sink is
    /// attached — an honest no-op, never fabricated traffic.
    pub fn emit_log(&self, level: LogLevel, logger: Option<&str>, data: Value) -> Option<Value> {
        let sink = self.log_sink.as_ref()?;
        let threshold = self.log_level.lock().unwrap_or_else(|e| e.into_inner());
        if !level.passes(*threshold) {
            return None;
        }
        let envelope = message_envelope(level, logger, data);
        sink.emit(&envelope);
        Some(envelope)
    }

    /// Injects a sampling callback (server→host direction), enabling [`Self::create_message`].
    ///
    /// The callback delivers `sampling/createMessage` to the connected Host and retrieves the response;
    /// without it, `create_message` returns a clear error.
    pub fn with_sampling_handler(mut self, handler: Arc<dyn SamplingHandler>) -> Self {
        self.sampling_handler = Some(handler);
        self
    }

    /// Injects an elicitation callback (server→host direction), enabling [`Self::create_elicitation`].
    ///
    /// The callback delivers `elicitation/create` to the connected Host (collecting input from the user via its
    /// UI) and retrieves the response; without it, `create_elicitation` returns a clear error.
    pub fn with_elicitation_handler(mut self, handler: Arc<dyn ElicitationHandler>) -> Self {
        self.elicitation_handler = Some(handler);
        self
    }

    /// Builds an MCP tool definition from a BaseTool
    fn tool_definition(tool: &dyn BaseTool) -> MCPToolDefinition {
        MCPToolDefinition {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            input_schema: tool
                .args_schema()
                .unwrap_or_else(|| json!({"type":"object"})),
        }
    }

    /// Handles one JSON-RPC request, returning the response
    ///
    /// For direct calls from unit tests; `serve_stdio` also uses it to process each request line.
    pub async fn handle_request(&self, req: MCPRequest) -> MCPResponse {
        // F1/H3: unscoped by default — the single-connection stdio transport
        // and direct callers, whose JSON-RPC ids are unique by construction.
        // The Streamable HTTP transport routes through handle_request_scoped
        // instead so cancellation is keyed by (session, request id).
        self.handle_request_scoped(None, req).await
    }

    /// Session-scoped dispatch (F1/H3). The Streamable HTTP transport threads
    /// the `Mcp-Session-Id` here so the B8 cancellation table is keyed by
    /// `(session, request id)`: two concurrent sessions that reuse the same
    /// JSON-RPC id cannot cancel each other's in-flight tool call.
    pub async fn handle_request_scoped(
        &self,
        session: Option<&str>,
        req: MCPRequest,
    ) -> MCPResponse {
        match req.method.as_str() {
            // P2-10 version negotiation: echo the requested version when it is in the support list, otherwise
            // degrade to this implementation's version (a Server only ever replies with a version it supports).
            "initialize" => {
                let requested = req
                    .params
                    .as_ref()
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(Value::as_str);
                let protocol_version = match requested {
                    Some(v) if SUPPORTED_PROTOCOL_VERSIONS.contains(&v) => v.to_string(),
                    _ => MCP_VERSION.to_string(),
                };
                // S10 capability declaration: client→server primitives are added per actual registration;
                // sampling/elicitation are client capabilities and do not go into server capabilities.
                let mut capabilities = json!({ "tools": {} });
                if self.resources.is_some() {
                    capabilities["resources"] = json!({});
                }
                if self.prompts.is_some() {
                    capabilities["prompts"] = json!({});
                }
                if self.completion.is_some() {
                    capabilities["completion"] = json!({});
                }
                if self.tasks.is_some() {
                    capabilities["tasks"] = json!({});
                }
                if self.log_sink.is_some() {
                    capabilities["logging"] = json!({});
                }
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": protocol_version,
                        "capabilities": capabilities,
                        "serverInfo": { "name": self.server_name, "version": self.server_version }
                    })),
                    error: None,
                }
            }
            // MCP lifecycle liveness probe: every server MUST answer `ping`
            // with an empty result (used by official SDK keep-alives and the
            // Streamable HTTP client).
            "ping" => MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: Some(json!({})),
                error: None,
            },
            "tools/list" => {
                // Cursor-paginated (#79): the `cursor` param is an opaque
                // continuation token; when more tools remain the result carries
                // a `nextCursor` the client passes back to continue. A client
                // that ignores `nextCursor` still gets every tool via repeated
                // pages, so non-paginated peers keep working.
                let params = req.params.clone().unwrap_or(Value::Null);
                let cursor = params.get("cursor").and_then(Value::as_str);
                let offset = cursor
                    .and_then(|c| c.parse::<usize>().ok())
                    .unwrap_or(0);
                let tools: Vec<MCPToolDefinition> = self
                    .tools
                    .iter()
                    .map(|t| Self::tool_definition(t.as_ref()))
                    .collect();
                // The `cursor` is client-controlled, so clamp it to the tool
                // list *before* any arithmetic: a hostile `usize::MAX` cursor
                // must never overflow `offset + TOOL_PAGE_SIZE` (a panic in
                // debug, silent wrap + out-of-range slice in release) or slice
                // `tools[start..end]` with `start > end` (see #H-4).
                let start = offset.min(tools.len());
                let end = start
                    .checked_add(TOOL_PAGE_SIZE)
                    .unwrap_or(usize::MAX)
                    .min(tools.len());
                let page: Vec<_> = tools[start..end].to_vec();
                let mut result = json!({ "tools": page });
                if end < tools.len() {
                    result["nextCursor"] = Value::String(end.to_string());
                }
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(result),
                    error: None,
                }
            }
            "tools/call" => {
                self.handle_tools_call(session.map(str::to_string), req)
                    .await
            }
            // 0.27.0 #79 item ④ logging: with a sink, accepts the level and
            // remembers it as the emit threshold; without a sink,
            // method_not_found (honest boundary).
            LOGGING_SETLEVEL => self.handle_logging_set_level(req),
            // 2026-07-28 stateless track: on-demand capability discovery —
            // self-contained requests ask what the server can do without a
            // prior initialize handshake. 0.27.0 T1 pulls the tool manifest
            // into the result so a single discover both reveals capabilities
            // and names every exposed tool (the "capability + tool listing"
            // shape of the new spec), even before any initialize.
            "server/discover" => {
                let mut capabilities = json!({ "tools": {} });
                if self.resources.is_some() {
                    capabilities["resources"] = json!({});
                }
                if self.prompts.is_some() {
                    capabilities["prompts"] = json!({});
                }
                if self.completion.is_some() {
                    capabilities["completion"] = json!({});
                }
                if self.tasks.is_some() {
                    capabilities["tasks"] = json!({});
                }
                if self.log_sink.is_some() {
                    capabilities["logging"] = json!({});
                }
                let tools: Vec<MCPToolDefinition> = self
                    .tools
                    .iter()
                    .map(|t| Self::tool_definition(t.as_ref()))
                    .collect();
                let tools_val =
                    serde_json::to_value(&tools).unwrap_or_else(|_| Value::Array(vec![]));
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": MCP_VERSION_STATELESS,
                        "capabilities": capabilities,
                        "serverInfo": { "name": self.server_name, "version": self.server_version },
                        "tools": tools_val
                    })),
                    error: None,
                }
            }
            // 0.27.0 T1: resource subscriptions — registered registry → correct
            // structure, unregistered → method_not_found. The subscriber is the
            // session scope (Streamable HTTP) or a per-connection token; unscoped
            // callers (stdio / direct) share the "default" subscriber.
            RESOURCES_SUBSCRIBE => self.handle_resource_subscription(session, req, true),
            RESOURCES_UNSUBSCRIBE => self.handle_resource_subscription(session, req, false),
            // S10 five client→server primitives: registered → correct structure, unregistered → method_not_found.
            "resources/list" => self.handle_resources_list(req).await,
            "resources/read" => self.handle_resources_read(req).await,
            "prompts/list" => self.handle_prompts_list(req).await,
            "prompts/get" => self.handle_prompts_get(req).await,
            "completion/complete" => self.handle_completion_complete(req).await,
            "tasks/get" => self.handle_tasks_get(req).await,
            "tasks/cancel" => self.handle_tasks_cancel(req).await,
            _ => Self::method_not_found_response(req.id),
        }
    }

    /// `resources/list`: lists the registered resources (cursor-paginated).
    async fn handle_resources_list(&self, req: MCPRequest) -> MCPResponse {
        match &self.resources {
            Some(provider) => {
                // Read the optional `cursor` param so a client paging through
                // results can continue from its previous `nextCursor` (#79).
                let params = req.params.clone().unwrap_or(Value::Null);
                let cursor = params
                    .get("cursor")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                match provider.list_resources_page(cursor).await {
                    Ok(result) => {
                        let result =
                            serde_json::to_value(result).unwrap_or(Value::Null);
                        Self::ok_response(req.id, result)
                    }
                    Err(e) => Self::error_response(req.id, e),
                }
            }
            None => Self::method_not_found_response(req.id),
        }
    }

    /// `resources/read`: reads a resource's content by URI.
    async fn handle_resources_read(&self, req: MCPRequest) -> MCPResponse {
        let provider = match &self.resources {
            Some(p) => p,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let uri = match params.get("uri").and_then(Value::as_str) {
            Some(u) => u.to_string(),
            None => {
                return Self::invalid_params_response(req.id, "missing uri parameter");
            }
        };
        match provider.read_resource(&uri).await {
            Ok(contents) => {
                let result =
                    serde_json::to_value(ReadResourceResult { contents }).unwrap_or(Value::Null);
                Self::ok_response(req.id, result)
            }
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `prompts/list`: lists the registered prompts (cursor-paginated).
    async fn handle_prompts_list(&self, req: MCPRequest) -> MCPResponse {
        match &self.prompts {
            Some(provider) => {
                // Read the optional `cursor` param so a client paging through
                // results can continue from its previous `nextCursor` (#79).
                let params = req.params.clone().unwrap_or(Value::Null);
                let cursor = params
                    .get("cursor")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                match provider.list_prompts_page(cursor).await {
                    Ok(result) => {
                        let result = serde_json::to_value(result).unwrap_or(Value::Null);
                        Self::ok_response(req.id, result)
                    }
                    Err(e) => Self::error_response(req.id, e),
                }
            }
            None => Self::method_not_found_response(req.id),
        }
    }

    /// `prompts/get`: generates prompt messages by name + arguments.
    async fn handle_prompts_get(&self, req: MCPRequest) -> MCPResponse {
        let provider = match &self.prompts {
            Some(p) => p,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let name = match params.get("name").and_then(Value::as_str) {
            Some(n) => n.to_string(),
            None => {
                return Self::invalid_params_response(req.id, "missing name parameter");
            }
        };
        let arguments = params.get("arguments").cloned();
        match provider.get_prompt(&name, arguments.as_ref()).await {
            Ok(result) => {
                let result = serde_json::to_value(result).unwrap_or(Value::Null);
                Self::ok_response(req.id, result)
            }
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `completion/complete`: provides completion suggestions for prompt arguments / resource URIs.
    async fn handle_completion_complete(&self, req: MCPRequest) -> MCPResponse {
        let provider = match &self.completion {
            Some(p) => p,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let request: CompletionRequest = match serde_json::from_value(params) {
            Ok(r) => r,
            Err(e) => {
                return Self::invalid_params_response(
                    req.id,
                    format!("invalid completion request: {e}"),
                );
            }
        };
        match provider.complete(&request).await {
            Ok(result) => {
                let result = serde_json::to_value(result).unwrap_or(Value::Null);
                Self::ok_response(req.id, result)
            }
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `tasks/get` (T3 Tasks extension): polls a durable task handle by id.
    async fn handle_tasks_get(&self, req: MCPRequest) -> MCPResponse {
        let provider = match &self.tasks {
            Some(p) => p,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let id = match params.get("id").and_then(Value::as_str) {
            Some(i) => i.to_string(),
            None => return Self::invalid_params_response(req.id, "missing id parameter"),
        };
        match provider.get_task(&id).await {
            Ok(result) => {
                let result = serde_json::to_value(result).unwrap_or(Value::Null);
                Self::ok_response(req.id, result)
            }
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `tasks/cancel` (T3 Tasks extension): cancels a running/queued task.
    async fn handle_tasks_cancel(&self, req: MCPRequest) -> MCPResponse {
        let provider = match &self.tasks {
            Some(p) => p,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let id = match params.get("id").and_then(Value::as_str) {
            Some(i) => i.to_string(),
            None => return Self::invalid_params_response(req.id, "missing id parameter"),
        };
        match provider.cancel_task(&id).await {
            Ok(cancelled) => Self::ok_response(req.id, json!({ "cancelled": cancelled })),
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `resources/subscribe` / `resources/unsubscribe` (0.27.0 T1): registers /
    /// clears the caller's interest in a resource URI so a later
    /// [`Self::notify_resource_changed`] can reach them.
    ///
    /// Unregistered registry → `method_not_found` (honest boundary). Missing
    /// `uri` param → `invalid_params`. A successful subscribe/unsubscribe
    /// answers `{}` per the MCP shape.
    fn handle_resource_subscription(
        &self,
        session: Option<&str>,
        req: MCPRequest,
        subscribe: bool,
    ) -> MCPResponse {
        let subs = match &self.subscriptions {
            Some(s) => s,
            None => return Self::method_not_found_response(req.id),
        };
        let params = req.params.clone().unwrap_or(Value::Null);
        let uri = match params.get("uri").and_then(Value::as_str) {
            Some(u) => u.to_string(),
            None => {
                return Self::invalid_params_response(req.id, "missing uri parameter");
            }
        };
        // The subscriber id is the session scope when present (Streamable HTTP
        // threads the Mcp-Session-Id here); single-connection transports and
        // direct callers collapse to one "default" subscriber.
        let subscriber = session.unwrap_or("default").to_string();
        let result = if subscribe {
            subs.subscribe(&uri, subscriber)
        } else {
            subs.unsubscribe(&uri, subscriber)
        };
        match result {
            Ok(()) => Self::ok_response(req.id, json!({})),
            Err(e) => Self::error_response(req.id, e),
        }
    }

    /// `logging/setLevel` (0.27.0 #79 item ④): reads the `level` param (a name
    /// or an integer), stores it as the [`Self::emit_log`] threshold, and answers
    /// `{}`. A missing/invalid `level` → `invalid_params`; no sink registered →
    /// `method_not_found` (honest boundary).
    fn handle_logging_set_level(&self, req: MCPRequest) -> MCPResponse {
        if self.log_sink.is_none() {
            return Self::method_not_found_response(req.id);
        }
        let params = req.params.clone().unwrap_or(Value::Null);
        let level = match params.get("level") {
            Some(v) => match crate::logging::parse_level(v) {
                Ok(l) => l,
                Err(e) => return Self::error_response(req.id, e),
            },
            None => return Self::invalid_params_response(req.id, "missing level parameter"),
        };
        *self.log_level.lock().unwrap_or_else(|e| e.into_inner()) = Some(level);
        Self::ok_response(req.id, json!({}))
    }

    /// Fires one sampling request (server→host direction, S10).
    ///
    /// Per MCP semantics, `sampling/createMessage` is initiated by the Server and the Host runs the LLM inference.
    /// This method hands the request to the injected [`SamplingHandler`]; without an injected handler it returns
    /// a clear error, never silently. Real interaction depends on the host environment's UI/models and is wired
    /// in by the user via [`Self::with_sampling_handler`].
    pub async fn create_message(
        &self,
        request: &SamplingRequest,
    ) -> Result<SamplingResult, MCPError> {
        match &self.sampling_handler {
            Some(handler) => handler.create_message(request).await,
            None => Err(MCPError::new(
                -32603,
                "sampling handler not configured: register one via \
                 MCPServer::with_sampling_handler() before create_message",
            )),
        }
    }

    /// Fires one elicitation request (server→host direction, S10).
    ///
    /// Per MCP semantics, `elicitation/create` is initiated by the Server and the Host collects input from the
    /// user via its UI. This method hands the request to the injected [`ElicitationHandler`]; without an injected
    /// handler it returns a clear error, never silently. Real interaction depends on the host UI and is wired in
    /// by the user via [`Self::with_elicitation_handler`].
    pub async fn create_elicitation(
        &self,
        request: &ElicitationRequest,
    ) -> Result<ElicitationResponse, MCPError> {
        match &self.elicitation_handler {
            Some(handler) => handler.create(request).await,
            None => Err(MCPError::new(
                -32603,
                "elicitation handler not configured: register one via \
                 MCPServer::with_elicitation_handler() before create_elicitation",
            )),
        }
    }

    /// Builds a success response.
    fn ok_response(id: JsonRpcId, result: Value) -> MCPResponse {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    /// Builds a response carrying a JSON-RPC error.
    fn error_response(id: JsonRpcId, error: MCPError) -> MCPResponse {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: None,
            error: Some(error),
        }
    }

    /// Builds a `method_not_found` (-32601) response: shared by unregistered capabilities / unknown methods.
    fn method_not_found_response(id: JsonRpcId) -> MCPResponse {
        Self::error_response(id, MCPError::method_not_found())
    }

    /// Builds an `invalid_params` (-32602) response.
    fn invalid_params_response(id: JsonRpcId, msg: impl Into<String>) -> MCPResponse {
        Self::error_response(id, MCPError::invalid_params(msg))
    }

    async fn handle_tools_call(&self, session: Option<String>, req: MCPRequest) -> MCPResponse {
        let params = req.params.clone().unwrap_or(Value::Null);
        let name = params.get("name").and_then(|v| v.as_str());
        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

        let name = match name {
            Some(n) => n,
            None => {
                return MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: None,
                    error: Some(MCPError::invalid_params("missing name parameter")),
                }
            }
        };

        let tool = self.tools.iter().find(|t| t.name() == name);
        match tool {
            Some(t) => {
                // B8/M-6: register a cancellation signal for this request id *before*
                // any dispatch work (argument serialization), so an early
                // `notifications/cancelled` is not dropped as "unknown request id"
                // (TOCTOU between request receipt and Notify insertion). Removed
                // once the tool completes.
                let notify = {
                    let mut map = self.cancellations.lock().await;
                    map.entry((session.clone(), req.id.clone()))
                        .or_insert_with(|| Arc::new(Notify::new()))
                        .clone()
                };
                let input_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "null".into());
                let mut run = Box::pin(t.run(input_str));
                let mut cancelled = Box::pin(notify.notified());
                let mcp_result = match self.tool_timeout {
                    // 0.26.0 P0-1: a configured deadline adds a `timeout` branch to
                    // the select. Firing it drops the in-flight tool future (freeing
                    // its concurrency permit) and reports a hard error observation,
                    // so a single hung tool can no longer strand the whole server.
                    Some(timeout) => {
                        let mut deadline = Box::pin(tokio::time::sleep(timeout));
                        tokio::select! {
                            result = &mut run => match result {
                                Ok(text) => MCPToolResult {
                                    content: vec![MCPContent::Text { text }],
                                    is_error: false,
                                },
                                Err(e) => MCPToolResult {
                                    content: vec![MCPContent::Text { text: e.to_string() }],
                                    is_error: true,
                                },
                            },
                            _ = &mut cancelled => MCPToolResult {
                                content: vec![MCPContent::Text { text: "cancelled by client".into() }],
                                is_error: true,
                            },
                            _ = &mut deadline => MCPToolResult {
                                content: vec![MCPContent::Text {
                                    text: format!("tool '{name}' exceeded its {timeout:?} execution deadline"),
                                }],
                                is_error: true,
                            },
                        }
                    }
                    None => tokio::select! {
                        result = &mut run => match result {
                            Ok(text) => MCPToolResult {
                                content: vec![MCPContent::Text { text }],
                                is_error: false,
                            },
                            Err(e) => MCPToolResult {
                                content: vec![MCPContent::Text { text: e.to_string() }],
                                is_error: true,
                            },
                        },
                        _ = &mut cancelled => MCPToolResult {
                            content: vec![MCPContent::Text { text: "cancelled by client".into() }],
                            is_error: true,
                        },
                    },
                };
                self.cancellations
                    .lock()
                    .await
                    .remove(&(session, req.id.clone()));
                let result_val = serde_json::to_value(&mcp_result).unwrap_or(Value::Null);
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(result_val),
                    error: None,
                }
            }
            None => MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: None,
                error: Some(MCPError::invalid_params(format!("unknown tool: {}", name))),
            },
        }
    }

    /// Runs the server on stdio: reads JSON-RPC from stdin, processes it, writes responses back to stdout
    ///
    /// Notifications (messages without an id, such as `notifications/initialized`) are dispatched to
    /// `handle_notification` and get no response; requests (with an id) get a response.
    pub async fn serve_stdio(&self) -> Result<(), MCPError> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut stdout = tokio::io::stdout();

        loop {
            let mut line = String::new();
            let n = read_bounded_line(&mut reader, &mut line, MAX_STDIO_LINE_BYTES)
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to read stdin: {}", e)))?;
            if n == 0 {
                break; // EOF
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse leniently: notifications have no id, requests have one
            let msg: ServerMessage = match serde_json::from_str(trimmed) {
                Ok(m) => m,
                Err(e) => {
                    // Per JSON-RPC 2.0 spec: if the request could not be parsed,
                    // the response id MUST be null.
                    let resp = MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(MCPError::invalid_params(format!(
                            "failed to parse request: {}",
                            e
                        ))),
                    };
                    let json = serde_json::to_string(&resp)
                        .map_err(|e| MCPError::new(-1, format!("serialization failed: {}", e)))?;
                    let _ = write_line(&mut stdout, &json).await;
                    continue;
                }
            };

            // Notification (no id): P0-4 dispatches to handle_notification instead of dropping
            let id = match msg.id {
                Some(id) => id,
                None => {
                    self.handle_notification(&msg.method, msg.params).await;
                    continue;
                }
            };

            let req = MCPRequest {
                jsonrpc: "2.0".to_string(),
                id,
                method: msg.method,
                params: msg.params,
                meta: None,
            };
            let resp = self.handle_request(req).await;
            let json = serde_json::to_string(&resp)
                .map_err(|e| MCPError::new(-1, format!("serialization failed: {}", e)))?;
            write_line(&mut stdout, &json)
                .await
                .map_err(|e| MCPError::new(-1, format!("failed to write stdout: {}", e)))?;
        }
        Ok(())
    }

    /// Serves MCP as a stateless HTTP service on an already-bound TCP listener,
    /// returning the endpoint URL clients connect to.
    ///
    /// This is the "deployable MCP server" entry point (2026-07-28 stateless
    /// track): it exposes this server as an HTTP service that any stateless
    /// MCP client (`StatelessMcpClient::connect(url)`) can POST JSON-RPC to.
    /// No handshake, no session — every request is self-contained.
    ///
    /// - `listener`: a `TcpListener` already bound to an address. For local
    ///   debugging bind `127.0.0.1:0`; for a network deployment bind
    ///   `0.0.0.0:PORT` **and** attach a token validator first (M-2) — a
    ///   non-loopback bind without one is refused up front.
    ///
    /// Returns the endpoint URL immediately after startup; the accept loop
    /// runs on a background task until the process exits.
    pub fn serve_http(self: Arc<Self>, listener: tokio::net::TcpListener) -> String {
        assert_safe_bind(&listener, self.auth_validator.is_some());
        let addr = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        // C8: bounded concurrency — accept blocks for a permit instead of
        // spawning an unbounded task per connection.
        let semaphore = Arc::new(Semaphore::new(HTTP_MAX_CONCURRENT_CONNECTIONS));
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => break,
                };
                let server = self.clone();
                let permit = match semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                tokio::spawn(async move {
                    let _permit = permit; // released on connection close
                    loop {
                        // C8: reap connections that never finish a request.
                        let request = match tokio::time::timeout(
                            std::time::Duration::from_secs(HTTP_READ_TIMEOUT_SECS),
                            read_http_request(&mut sock),
                        )
                        .await
                        {
                            Err(_) => {
                                let _ = sock
                                        .write_all(
                                            b"HTTP/1.1 408 Request Timeout\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                                        )
                                        .await;
                                return;
                            }
                            Ok(Err(_)) => return, // closed / malformed transport
                            Ok(Ok(req)) => req,
                        };
                        if request.first_line.is_empty() {
                            return;
                        }
                        if !request.first_line.starts_with("POST ") {
                            let _ = sock
                                .write_all(
                                    b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n",
                                )
                                .await;
                            continue;
                        }

                        // C8: authenticate every request when a validator is configured.
                        if let Some(validator) = &server.auth_validator {
                            let token = request
                                .headers
                                .iter()
                                .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                                .and_then(|(_, v)| v.strip_prefix("Bearer "))
                                .map(str::to_string);
                            let token = match token {
                                Some(t) => t,
                                None => {
                                    let _ = sock
                                        .write_all(
                                            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n",
                                        )
                                        .await;
                                    return;
                                }
                            };
                            if validator.validate(&token).await.is_err() {
                                let _ = sock
                                    .write_all(
                                        b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n",
                                    )
                                    .await;
                                return;
                            }
                        }

                        let body = match request.body {
                            Some(b) => b,
                            None => {
                                // C8: truncated body — reply instead of slicing out of bounds.
                                let _ = sock
                                    .write_all(
                                        b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n",
                                    )
                                    .await;
                                return;
                            }
                        };

                        // JSON-RPC over HTTP (M2, 0.26.0 §1.1): distinguish a
                        // request from a notification. A message with no `id` is a
                        // *notification* — lenient-parse (same `ServerMessage` the
                        // stdio path uses) and dispatch to `handle_notification`
                        // instead of failing `MCPRequest`'s required-id field with
                        // `-32700`. Per JSON-RPC, notifications get NO response.
                        //
                        // Unparseable body → error envelope with `id: null`
                        // (spec-compliant; previously `id: 0`).
                        let msg: ServerMessage = match serde_json::from_str(&body) {
                            Ok(m) => m,
                            Err(e) => {
                                let resp = MCPResponse {
                                    jsonrpc: "2.0".to_string(),
                                    id: None,
                                    result: None,
                                    error: Some(MCPError::new(-32700, format!("parse error: {e}"))),
                                };
                                let payload = serde_json::to_string(&resp)
                                    .unwrap_or_else(|_| "{}".to_string());
                                let http = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                    payload.len(),
                                    payload
                                );
                                if sock.write_all(http.as_bytes()).await.is_err() {
                                    return;
                                }
                                continue;
                            }
                        };

                        if msg.id.is_none() {
                            // Notification (no id): no response per JSON-RPC — ACK
                            // with 202 Accepted, empty body.
                            server.handle_notification(&msg.method, msg.params).await;
                            if sock
                                .write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
                                .await
                                .is_err()
                            {
                                return;
                            }
                            continue;
                        }

                        let req = MCPRequest {
                            jsonrpc: "2.0".to_string(),
                            id: msg.id.expect("checked Some above"),
                            method: msg.method,
                            params: msg.params,
                            meta: None,
                        };
                        let resp = server.handle_request(req).await;
                        let payload =
                            serde_json::to_string(&resp).unwrap_or_else(|_| "{}".to_string());
                        let http = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                            payload.len(),
                            payload
                        );
                        if sock.write_all(http.as_bytes()).await.is_err() {
                            return;
                        }
                    }
                });
            }
        });
        format!("http://{addr}/mcp")
    }

    /// Serves this MCP server over the **official Streamable HTTP transport**
    /// (B1, 0.22.4): one POST endpoint speaking the 2025-03-26 MCP streamable
    /// HTTP protocol — `initialize` handshake, per-session `Mcp-Session-Id`,
    /// JSON or SSE response bodies (content-negotiated via `Accept`), and
    /// HTTP 202 for notifications — so official TS/Python SDK clients and
    /// [`crate::StreamableMcpClient`] can connect.
    ///
    /// This is distinct from [`Self::serve_http`], which remains the
    /// self-contained 2026-07-28 stateless track (no handshake).
    ///
    /// - `listener`: an already-bound `TcpListener`; bind `127.0.0.1:0` for
    ///   local/CI use.
    ///
    /// Returns the endpoint URL immediately; the accept loop runs on a
    /// background task until the listener is closed.
    pub fn serve_streamable_http(self: Arc<Self>, listener: tokio::net::TcpListener) -> String {
        assert_safe_bind(&listener, self.auth_validator.is_some());
        crate::transport::streamable_http_server::serve(self, listener)
    }

    /// Handles a notification the server receives (a message without an id).
    ///
    /// P0-4: explicitly dispatches MCP standard notifications instead of dropping them:
    /// - `notifications/cancelled` — the client requests cancelling a tool call
    /// - `notifications/progress` — the client reports tool execution progress
    /// - `notifications/roots/list_changed` — the roots list changed
    /// - `notifications/initialized` — the client finished the handshake
    ///
    /// The current implementation logs and leaves extension points; derived types can override it later to hook
    /// in cancel/progress callbacks.
    pub async fn handle_notification(&self, method: &str, params: Option<Value>) {
        // F1/H3: unscoped default — single-connection transports only. The
        // Streamable HTTP transport routes through handle_notification_scoped
        // so the cancellation lookup is keyed by (session, request id).
        self.handle_notification_scoped(None, method, params).await
    }

    /// Session-scoped notification dispatch (F1/H3): matches
    /// [`Self::handle_request_scoped`] so a `notifications/cancelled` for a
    /// given session only fires that session's own request with the same id.
    pub async fn handle_notification_scoped(
        &self,
        session: Option<&str>,
        method: &str,
        params: Option<Value>,
    ) {
        match method {
            "notifications/cancelled" => {
                // B8: carries requestId; fire the request's Notify so an
                // in-flight tools/call aborts instead of running to completion.
                let request_id = params
                    .as_ref()
                    .and_then(|p| p.get("requestId"))
                    .and_then(|v| {
                        v.as_u64()
                            .map(JsonRpcId::Number)
                            .or_else(|| v.as_str().map(|s| JsonRpcId::String(s.to_string())))
                    });
                match request_id {
                    Some(rid) => {
                        let key = (session.map(str::to_string), rid.clone());
                        let notify = self.cancellations.lock().await.get(&key).cloned();
                        if let Some(notify) = notify {
                            notify.notify_one();
                            log::debug!("MCP cancelled request {rid}");
                        } else {
                            log::debug!(
                                "MCP cancelled notification for unknown/unregistered request id {rid}"
                            );
                        }
                    }
                    None => log::info!(
                        "MCP received cancelled notification without a requestId: {:?}",
                        params
                    ),
                }
            }
            "notifications/progress" => {
                // Carries token + progress/estimatedTotal
                log::info!("MCP received progress notification: {:?}", params);
            }
            "notifications/roots/list_changed" => {
                log::info!("MCP received roots/list_changed notification: {:?}", params);
            }
            "notifications/initialized" => {
                log::debug!("MCP received initialized notification");
            }
            _ => {
                log::debug!("ignoring unknown notification: {}", method);
            }
        }
    }
}

impl Default for MCPServer {
    fn default() -> Self {
        Self::new()
    }
}

/// A lenient inbound message: notifications have no id
#[derive(Deserialize)]
struct ServerMessage {
    #[serde(default)]
    id: Option<JsonRpcId>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

async fn write_line<W: AsyncWriteExt + Unpin>(w: &mut W, json: &str) -> Result<(), std::io::Error> {
    w.write_all(json.as_bytes()).await?;
    w.write_all(b"\n").await?;
    w.flush().await?;
    Ok(())
}

/// One parsed HTTP request (0.22.0 C8: `body` is `None` when the connection
/// closed mid-body — the caller answers 400 instead of slicing out of bounds).
pub(crate) struct HttpRequest {
    pub(crate) first_line: String,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) body: Option<String>,
}

/// Reads one HTTP request from the socket.
/// `Err(())` = transport closed (or an oversized header line); the caller
/// closes the connection. Body reads are bounded by [`HTTP_MAX_BODY_BYTES`].
pub(crate) async fn read_http_request(sock: &mut tokio::net::TcpStream) -> Result<HttpRequest, ()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let mut end = None;
        for i in 0..buf.len().saturating_sub(3) {
            if &buf[i..i + 4] == b"\r\n\r\n" {
                end = Some(i);
                break;
            }
        }
        // M-3: bound the header region. The blank line was never seen and the
        // accumulated prefix already exceeds the cap → abort (transport closed),
        // so a hostile client cannot grow `buf` (and the O(n) scan) without limit.
        if end.is_none() && buf.len() > HTTP_MAX_HEADER_BYTES {
            return Err(());
        }
        if let Some(pos) = end {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let mut lines = head.lines();
            let first_line = lines.next().unwrap_or_default().to_string();
            let mut headers = Vec::new();
            for line in lines {
                if let Some((k, v)) = line.split_once(':') {
                    headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            let content_length = headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(0);
            // C8: reject oversized bodies up front.
            if content_length > HTTP_MAX_BODY_BYTES {
                return Ok(HttpRequest {
                    first_line,
                    headers,
                    body: None,
                });
            }
            let body_start = pos + 4;
            while buf.len() < body_start + content_length {
                let n = sock.read(&mut tmp).await.unwrap_or(0);
                if n == 0 {
                    // Truncated: the caller must not slice `buf` short.
                    return Ok(HttpRequest {
                        first_line,
                        headers,
                        body: None,
                    });
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = String::from_utf8_lossy(&buf[body_start..(body_start + content_length)])
                .to_string();
            return Ok(HttpRequest {
                first_line,
                headers,
                body: Some(body),
            });
        }
        let n = sock.read(&mut tmp).await.unwrap_or(0);
        if n == 0 {
            return Err(());
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::{CompletionResult, CompletionValue};
    use crate::elicitation::ElicitationAction;
    use crate::prompts::{
        GetPromptResult, ListPromptsResult, Prompt, PromptContent, PromptMessage,
    };
    use crate::resources::{ListResourcesResult, Resource, ResourceContent};
    use crate::sampling::{SamplingContent, SamplingMessage, SamplingRole};
    use crate::tasks::{ShippingLabel, TaskResponse, TaskState, TaskStatus};
    use lc_core::tools::ToolError;

    /// A test tool that echoes its input
    struct EchoTool;
    #[async_trait::async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "回显输入"
        }
        fn args_schema(&self) -> Option<Value> {
            Some(json!({"type":"object","properties":{"text":{"type":"string"}}}))
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(input)
        }
    }

    fn server_with_echo() -> MCPServer {
        MCPServer::new().with_tool(Arc::new(EchoTool))
    }

    #[tokio::test]
    async fn test_initialize() {
        let server = server_with_echo();
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", None))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        assert!(result.get("protocolVersion").is_some());
        assert!(result.get("capabilities").is_some());
        assert!(result.get("serverInfo").is_some());
    }

    /// P2-10 version negotiation: echoes when a supported version is requested, replies with the current
    /// implementation version when none is requested.
    #[tokio::test]
    async fn test_initialize_echoes_supported_version() {
        let server = server_with_echo();
        let params = serde_json::json!({ "protocolVersion": MCP_VERSION });
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", Some(params)))
            .await;
        let version = resp
            .result
            .as_ref()
            .and_then(|r| r.get("protocolVersion"))
            .and_then(Value::as_str)
            .map(str::to_string);
        assert_eq!(version.as_deref(), Some(MCP_VERSION));
    }

    /// P2-10 version negotiation: degrades to the current implementation version when an unsupported version is
    /// requested (a Server only replies with a version it supports, never echoing an unknown version).
    #[tokio::test]
    async fn test_initialize_degrades_unsupported_version() {
        let server = server_with_echo();
        let params = serde_json::json!({ "protocolVersion": "2099-01-01" });
        let resp = server
            .handle_request(MCPRequest::new(1, "initialize", Some(params)))
            .await;
        let version = resp
            .result
            .as_ref()
            .and_then(|r| r.get("protocolVersion"))
            .and_then(Value::as_str)
            .map(str::to_string);
        assert_eq!(
            version.as_deref(),
            Some(MCP_VERSION),
            "unknown version should degrade to the current implementation version"
        );
    }

    #[tokio::test]
    async fn test_tools_list() {
        let server = server_with_echo();
        let resp = server
            .handle_request(MCPRequest::new(2, "tools/list", None))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
    }

    /// #79: `tools/list` paginates — every tool is reachable across pages and
    /// `nextCursor` only appears while more results remain.
    #[tokio::test]
    async fn test_tools_list_paginates() {
        let server = server_with_echo();
        let mut server = server;
        for _ in 0..(TOOL_PAGE_SIZE + 5) {
            server = server.with_tool(Arc::new(EchoTool));
        }

        let mut collected = 0usize;
        let mut cursor: Option<Value> = None;
        loop {
            let params = cursor
                .map(|c| json!({ "cursor": c }))
                .unwrap_or_else(|| json!({}));
            let resp = server
                .handle_request(MCPRequest::new(1, "tools/list", Some(params)))
                .await;
            assert!(!resp.is_error());
            let result = resp.result.unwrap();
            let page = result.get("tools").unwrap().as_array().unwrap();
            assert!(!page.is_empty(), "page must not be empty");
            collected += page.len();
            match result.get("nextCursor").cloned() {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
        assert_eq!(collected, TOOL_PAGE_SIZE + 6);
    }

    /// #H-4: a hostile `usize::MAX` cursor must not overflow the pagination
    /// arithmetic (a panic / DoS) and must not slice `tools[start..end]` with
    /// `start > end`. Instead it degrades to an empty (or first) page.
    #[tokio::test]
    async fn test_tools_list_usize_max_cursor_no_panic() {
        let server = server_with_echo();
        let params = json!({ "cursor": usize::MAX.to_string() });
        let resp = server
            .handle_request(MCPRequest::new(1, "tools/list", Some(params)))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        // start clamps to `tools.len()`, so the page is empty and no burst.
        assert!(tools.is_empty());
        assert!(result.get("nextCursor").is_none());
    }

    #[tokio::test]
    async fn test_tools_list_large_cursor_beyond_end_is_empty() {
        // A cursor past the end must yield an empty page without panicking.
        let server = server_with_echo();
        let params = json!({ "cursor": "1000000000000" });
        let resp = server
            .handle_request(MCPRequest::new(1, "tools/list", Some(params)))
            .await;
        assert!(!resp.is_error());
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert!(tools.is_empty());
        assert!(result.get("nextCursor").is_none());
    }

    #[tokio::test]
    async fn test_tools_call_success() {
        let server = server_with_echo();
        let params = json!({"name":"echo","arguments":{"text":"hello"}});
        let resp = server
            .handle_request(MCPRequest::new(3, "tools/call", Some(params)))
            .await;
        assert!(!resp.is_error());
        let mcp_result: MCPToolResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert!(!mcp_result.is_error);
        // echo returns the input (the JSON string of arguments)
        assert_eq!(mcp_result.text(), r#"{"text":"hello"}"#);
    }

    #[tokio::test]
    async fn test_tools_call_unknown_tool() {
        let server = server_with_echo();
        let params = json!({"name":"nonexistent","arguments":{}});
        let resp = server
            .handle_request(MCPRequest::new(4, "tools/call", Some(params)))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602); // invalid_params
    }

    #[tokio::test]
    async fn test_tools_call_missing_name() {
        let server = server_with_echo();
        let params = json!({"arguments":{}});
        let resp = server
            .handle_request(MCPRequest::new(5, "tools/call", Some(params)))
            .await;
        assert!(resp.is_error());
    }

    #[tokio::test]
    async fn test_unknown_method() {
        let server = MCPServer::new();
        let resp = server
            .handle_request(MCPRequest::new(6, "foo/bar", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601); // method_not_found
    }

    // ============================================================================
    // S10 five client→server primitives: registered → correct structure, unregistered → method_not_found
    // ============================================================================

    struct MockResources;
    #[async_trait::async_trait]
    impl ResourceProvider for MockResources {
        async fn list_resources(&self) -> Result<Vec<Resource>, MCPError> {
            Ok(vec![Resource {
                uri: "file:///a.txt".to_string(),
                name: "a.txt".to_string(),
                description: Some("a sample resource".to_string()),
                mime_type: Some("text/plain".to_string()),
            }])
        }
        async fn read_resource(&self, uri: &str) -> Result<Vec<ResourceContent>, MCPError> {
            Ok(vec![ResourceContent {
                uri: uri.to_string(),
                mime_type: Some("text/plain".to_string()),
                text: Some("hello from resource".to_string()),
                blob: None,
            }])
        }
    }

    struct MockPrompts;
    #[async_trait::async_trait]
    impl PromptProvider for MockPrompts {
        async fn list_prompts(&self) -> Result<Vec<Prompt>, MCPError> {
            Ok(vec![Prompt {
                name: "greet".to_string(),
                description: Some("Greet someone".to_string()),
                arguments: vec![],
            }])
        }
        async fn get_prompt(
            &self,
            name: &str,
            arguments: Option<&Value>,
        ) -> Result<GetPromptResult, MCPError> {
            if name != "greet" {
                return Err(MCPError::invalid_params(format!("unknown prompt: {name}")));
            }
            let who = arguments
                .and_then(|a| a.get("who"))
                .and_then(Value::as_str)
                .map(|w| format!(", {w}"))
                .unwrap_or_default();
            Ok(GetPromptResult {
                description: Some("Greet someone".to_string()),
                messages: vec![PromptMessage {
                    role: "user".to_string(),
                    content: PromptContent::Text {
                        text: format!("Hello{who}"),
                    },
                }],
            })
        }
    }

    struct MockTasks;
    #[async_trait::async_trait]
    impl TaskProvider for MockTasks {
        async fn get_task(&self, id: &str) -> Result<TaskResponse, MCPError> {
            if id != "t-1" {
                return Err(MCPError::invalid_params(format!("unknown task: {id}")));
            }
            Ok(TaskResponse {
                id: "t-1".to_string(),
                title: "synthesize report".to_string(),
                status: TaskStatus {
                    state: TaskState::Completed,
                    reason: None,
                    extensions: None,
                },
                kind: "task".to_string(),
                shipping_label: ShippingLabel {
                    method: "tasks/get".to_string(),
                    protocol_version: MCP_VERSION.to_string(),
                },
                last_updated: None,
                restorable: false,
                runner_choice: None,
                scheduled_for_time: None,
                extensions: None,
            })
        }
        async fn cancel_task(&self, id: &str) -> Result<bool, MCPError> {
            if id != "t-1" {
                return Err(MCPError::invalid_params(format!("unknown task: {id}")));
            }
            Ok(true)
        }
    }

    struct MockCompletion;
    #[async_trait::async_trait]
    impl CompletionProvider for MockCompletion {
        async fn complete(
            &self,
            request: &CompletionRequest,
        ) -> Result<CompletionResult, MCPError> {
            // Filter candidates by prefix (the common shape of real completion).
            let prefix = &request.argument.value;
            let candidates = ["rust", "ruby", "python"];
            let values: Vec<CompletionValue> = candidates
                .iter()
                .filter(|s| s.starts_with(prefix.as_str()))
                .map(|s| CompletionValue {
                    label: s.to_string(),
                    description: None,
                })
                .collect();
            let count = values.len();
            Ok(CompletionResult {
                values,
                total: Some(count),
                has_more: false,
            })
        }
    }

    #[tokio::test]
    async fn test_resources_list_registered() {
        let server = MCPServer::new().with_resource_provider(Arc::new(MockResources));
        let resp = server
            .handle_request(MCPRequest::new(10, "resources/list", None))
            .await;
        assert!(
            !resp.is_error(),
            "resources/list 注册后应成功: {:?}",
            resp.error
        );
        let result: ListResourcesResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.resources.len(), 1);
        assert_eq!(result.resources[0].uri, "file:///a.txt");
    }

    #[tokio::test]
    async fn test_resources_list_not_registered() {
        let server = MCPServer::new();
        let resp = server
            .handle_request(MCPRequest::new(11, "resources/list", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_resources_read_registered() {
        let server = MCPServer::new().with_resource_provider(Arc::new(MockResources));
        let params = json!({"uri": "file:///a.txt"});
        let resp = server
            .handle_request(MCPRequest::new(12, "resources/read", Some(params)))
            .await;
        assert!(
            !resp.is_error(),
            "resources/read 注册后应成功: {:?}",
            resp.error
        );
        let result: ReadResourceResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.contents.len(), 1);
        assert_eq!(
            result.contents[0].text.as_deref(),
            Some("hello from resource")
        );
    }

    #[tokio::test]
    async fn test_resources_read_missing_uri() {
        let server = MCPServer::new().with_resource_provider(Arc::new(MockResources));
        let resp = server
            .handle_request(MCPRequest::new(13, "resources/read", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_tasks_get_registered() {
        let server = MCPServer::new().with_task_provider(Arc::new(MockTasks));
        let params = json!({"id": "t-1"});
        let resp = server
            .handle_request(MCPRequest::new(90, "tasks/get", Some(params)))
            .await;
        assert!(
            !resp.is_error(),
            "tasks/get 注册后应成功: {:?}",
            resp.error
        );
        let result: TaskResponse = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.id, "t-1");
        assert_eq!(result.status.state, TaskState::Completed);
        assert_eq!(result.shipping_label.method, "tasks/get");
    }

    #[tokio::test]
    async fn test_tasks_get_not_registered() {
        let server = MCPServer::new();
        let resp = server
            .handle_request(MCPRequest::new(91, "tasks/get", Some(json!({"id": "t-1"}))))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_tasks_cancel_registered() {
        let server = MCPServer::new().with_task_provider(Arc::new(MockTasks));
        let resp = server
            .handle_request(MCPRequest::new(92, "tasks/cancel", Some(json!({"id": "t-1"}))))
            .await;
        assert!(!resp.is_error(), "tasks/cancel 注册后应成功");
        let result = resp.result.unwrap();
        assert_eq!(result["cancelled"], true);
    }

    #[tokio::test]
    async fn test_tasks_cancel_missing_id() {
        let server = MCPServer::new().with_task_provider(Arc::new(MockTasks));
        let resp = server
            .handle_request(MCPRequest::new(93, "tasks/cancel", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_prompts_list_registered() {
        let server = MCPServer::new().with_prompt_provider(Arc::new(MockPrompts));
        let resp = server
            .handle_request(MCPRequest::new(14, "prompts/list", None))
            .await;
        assert!(
            !resp.is_error(),
            "prompts/list 注册后应成功: {:?}",
            resp.error
        );
        let result: ListPromptsResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(result.prompts[0].name, "greet");
    }

    #[tokio::test]
    async fn test_prompts_list_not_registered() {
        let server = MCPServer::new();
        let resp = server
            .handle_request(MCPRequest::new(15, "prompts/list", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_prompts_get_registered() {
        let server = MCPServer::new().with_prompt_provider(Arc::new(MockPrompts));
        let params = json!({"name": "greet", "arguments": {"who": "world"}});
        let resp = server
            .handle_request(MCPRequest::new(16, "prompts/get", Some(params)))
            .await;
        assert!(
            !resp.is_error(),
            "prompts/get 注册后应成功: {:?}",
            resp.error
        );
        let result: GetPromptResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.messages.len(), 1);
        match &result.messages[0].content {
            PromptContent::Text { text } => assert_eq!(text, "Hello, world"),
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_prompts_get_missing_name() {
        let server = MCPServer::new().with_prompt_provider(Arc::new(MockPrompts));
        let resp = server
            .handle_request(MCPRequest::new(17, "prompts/get", None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_prompts_get_not_registered() {
        let server = MCPServer::new();
        let params = json!({"name": "greet"});
        let resp = server
            .handle_request(MCPRequest::new(18, "prompts/get", Some(params)))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_completion_complete_registered() {
        let server = MCPServer::new().with_completion_provider(Arc::new(MockCompletion));
        let params = json!({
            "reference": {"type": "ref/prompt", "uri": "prompt://greet"},
            "argument": {"name": "who", "value": "ru"}
        });
        let resp = server
            .handle_request(MCPRequest::new(19, "completion/complete", Some(params)))
            .await;
        assert!(
            !resp.is_error(),
            "completion/complete 注册后应成功: {:?}",
            resp.error
        );
        let result: CompletionResult = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.values.len(), 2, "ru 前缀应筛出 rust/ruby");
        assert_eq!(result.values[0].label, "rust");
        assert_eq!(result.total, Some(2));
    }

    #[tokio::test]
    async fn test_completion_complete_not_registered() {
        let server = MCPServer::new();
        let params = json!({
            "reference": {"type": "ref/prompt", "uri": "prompt://greet"},
            "argument": {"name": "who", "value": "ru"}
        });
        let resp = server
            .handle_request(MCPRequest::new(20, "completion/complete", Some(params)))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_initialize_capabilities_reflect_registration() {
        let server = MCPServer::new()
            .with_resource_provider(Arc::new(MockResources))
            .with_prompt_provider(Arc::new(MockPrompts))
            .with_completion_provider(Arc::new(MockCompletion));
        let resp = server
            .handle_request(MCPRequest::new(21, "initialize", None))
            .await;
        let caps = resp.result.unwrap().get("capabilities").unwrap().clone();
        assert!(caps.get("tools").is_some(), "tools 恒声明");
        assert!(caps.get("resources").is_some(), "注册后应声明 resources");
        assert!(caps.get("prompts").is_some(), "注册后应声明 prompts");
        assert!(caps.get("completion").is_some(), "注册后应声明 completion");

        let plain = MCPServer::new();
        let resp = plain
            .handle_request(MCPRequest::new(22, "initialize", None))
            .await;
        let caps = resp.result.unwrap().get("capabilities").unwrap().clone();
        assert!(caps.get("tools").is_some(), "tools 恒声明");
        assert!(caps.get("resources").is_none(), "未注册不声明 resources");
        assert!(caps.get("prompts").is_none(), "未注册不声明 prompts");
        assert!(caps.get("completion").is_none(), "未注册不声明 completion");
    }

    // ============================================================================
    // 0.27.0 T1: server/discover (tool manifest) and resources subscribe/unsubscribe
    // ============================================================================

    /// server/discover is self-contained (no handshake): it reveals the
    /// stateless protocol version, the declared capabilities, and — with the
    /// T1 tool manifest — every exposed tool.
    #[tokio::test]
    async fn test_server_discover_returns_tool_manifest() {
        let server = server_with_echo();
        let resp = server
            .handle_request(MCPRequest::new(7, "server/discover", None))
            .await;
        assert!(!resp.is_error(), "{:?}", resp.error);
        let result = resp.result.unwrap();
        assert_eq!(
            result["protocolVersion"], MCP_VERSION_STATELESS,
            "discover speaks the stateless version"
        );
        assert!(result["capabilities"]["tools"].is_object(), "{result}");
        // T1: the tool manifest is present and names the registered tool.
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
        assert_eq!(tools[0]["inputSchema"]["type"], "object");
    }

    /// server/discover is usable without any prior initialize — the stateless
    /// premise (a self-contained discovery of an anonymous server).
    #[tokio::test]
    async fn test_server_discover_works_without_initialize() {
        // No test here needs an initialize handshake; discover must succeed on a
        // fresh server that has never seen a request.
        let server = MCPServer::new().with_tool(Arc::new(EchoTool));
        let resp = server
            .handle_request(MCPRequest::new(8, "server/discover", None))
            .await;
        assert!(!resp.is_error());
    }

    /// resources/subscribe on a registered registry succeeds and a later
    /// notify_resource_changed emits the list_changed notification envelope.
    #[tokio::test]
    async fn test_resources_subscribe_then_notify() {
        let subs = Arc::new(ResourceSubscriptions::new());
        let server = MCPServer::new()
            .with_resource_provider(Arc::new(MockResources))
            .with_subscription_registry(subs.clone());
        let params = json!({"uri": "file:///a.txt"});
        let resp = server
            .handle_request(MCPRequest::new(30, RESOURCES_SUBSCRIBE, Some(params)))
            .await;
        assert!(!resp.is_error(), "{:?}", resp.error);
        // Subscribe answers the MCP empty object.
        assert!(subs.is_subscribed("file:///a.txt", "default"));

        // A change to the watched resource yields the notification envelope.
        let envelope = server
            .notify_resource_changed("file:///a.txt")
            .expect("watched uri must emit");
        assert_eq!(
            envelope["method"],
            crate::NOTIFICATIONS_RESOURCES_LIST_CHANGED
        );
        assert_eq!(envelope["params"]["uri"], "file:///a.txt");
    }

    /// notify_resource_updated is the per-resource content-change notification —
    /// distinct envelope (method) from list_changed, delivered to the same sink.
    #[test]
    fn test_notify_resource_updated_uses_updated_method() {
        let subs = Arc::new(ResourceSubscriptions::new());
        subs.subscribe("file:///a.txt", "default").unwrap();
        let server = MCPServer::new().with_subscription_registry(subs.clone());

        let list_changed = server
            .notify_resource_changed("file:///a.txt")
            .expect("watched uri emits list_changed");
        assert_eq!(list_changed["method"], crate::NOTIFICATIONS_RESOURCES_LIST_CHANGED);

        let updated = server
            .notify_resource_updated("file:///a.txt")
            .expect("watched uri emits updated");
        assert_eq!(updated["method"], crate::NOTIFICATIONS_RESOURCES_UPDATED);
        assert_eq!(updated["params"]["uri"], "file:///a.txt");
        assert!(updated.get("id").is_none(), "notifications carry no id");

        // An unwatched resource never emits (honest, no fabricated traffic).
        assert!(server.notify_resource_updated("file:///other.txt").is_none());
    }

    // ============================================================================
    // 0.27.0 #79 item ④: MCP `logging` (setLevel + notifications/message)
    // ============================================================================

    /// logging/setLevel stores the threshold and emit_log honors it; the sink
    /// sees only frames at/above the threshold. A registered sink makes the
    /// server advertise the `logging` capability on initialize.
    #[tokio::test]
    async fn test_logging_set_level_threshold_and_emit() {
        use std::sync::Mutex as StdMutex;
        struct RecordingLogSink {
            frames: StdMutex<Vec<Value>>,
        }
        impl LogSink for RecordingLogSink {
            fn emit(&self, message: &Value) {
                self.frames.lock().unwrap().push(message.clone());
            }
        }
        let sink = Arc::new(RecordingLogSink {
            frames: StdMutex::new(Vec::new()),
        });
        let server = MCPServer::new().with_log_sink(sink.clone());

        // No threshold yet → every level passes.
        let env = server
            .emit_log(LogLevel::Trace, None, json!("boot"))
            .expect("no threshold means always pass");
        assert_eq!(env["method"], crate::NOTIFICATIONS_MESSAGE);
        assert_eq!(sink.frames.lock().unwrap().len(), 1);

        // setLevel(warning) → info filtered, error passes.
        let resp = server
            .handle_request(MCPRequest::new(
                40,
                LOGGING_SETLEVEL,
                Some(json!({"level": "warning"})),
            ))
            .await;
        assert!(!resp.is_error(), "{:?}", resp.error);
        assert!(resp.result.unwrap() == json!({}));
        assert!(
            server.emit_log(LogLevel::Info, Some("tools"), json!(1)).is_none(),
            "below threshold must be filtered"
        );
        let env2 = server
            .emit_log(LogLevel::Error, Some("tools"), json!({"detail": 2}))
            .expect("at/above threshold passes");
        assert_eq!(env2["params"]["level"], 5);
        assert_eq!(env2["params"]["logger"], "tools");
        // Exactly the two passing frames reached the sink.
        assert_eq!(sink.frames.lock().unwrap().len(), 2);

        // Advertises `logging` when a sink is registered.
        let init = server
            .handle_request(MCPRequest::new(41, "initialize", None))
            .await;
        let caps = init.result.unwrap()["capabilities"].clone();
        assert!(caps.get("logging").is_some(), "{caps}");
    }

    /// logging/setLevel on a server without a sink is method_not_found (honest
    /// boundary), emit_log is a no-op, and initialize does not advertise logging.
    #[tokio::test]
    async fn test_logging_unregistered_boundary() {
        let server = MCPServer::new(); // no sink
        let resp = server
            .handle_request(MCPRequest::new(
                42,
                LOGGING_SETLEVEL,
                Some(json!({"level": "info"})),
            ))
            .await;
        assert_eq!(resp.error.as_ref().map(|e| e.code), Some(-32601));

        assert!(
            server.emit_log(LogLevel::Error, None, json!(1)).is_none(),
            "no sink → emit is an honest no-op"
        );

        let init = server
            .handle_request(MCPRequest::new(43, "initialize", None))
            .await;
        assert!(init.result.unwrap()["capabilities"]
            .get("logging")
            .is_none());
    }

    /// A missing/invalid level param is invalid_params, not a silent accept.
    #[tokio::test]
    async fn test_logging_set_level_invalid_param() {
        struct Sink;
        impl LogSink for Sink {
            fn emit(&self, _message: &Value) {}
        }
        let server = MCPServer::new().with_log_sink(Arc::new(Sink));
        let resp = server
            .handle_request(MCPRequest::new(
                44,
                LOGGING_SETLEVEL,
                Some(json!({"level": "bogus"})),
            ))
            .await;
        assert_eq!(resp.error.as_ref().map(|e| e.code), Some(-32602));
    }

    /// A delivery sink is invoked for each interested subscriber when a watched
    /// resource changes: the envelope is physically pushed, not just returned.
    #[tokio::test]
    async fn test_notify_with_push_sink_delivers_to_each_subscriber() {
        use std::sync::Mutex as StdMutex;
        struct RecordingSink {
            deliveries: StdMutex<Vec<(String, Value)>>,
        }
        impl ResourcePushSink for RecordingSink {
            fn deliver(&self, subscriber: &str, envelope: &Value) {
                self.deliveries
                    .lock()
                    .unwrap()
                    .push((subscriber.to_string(), envelope.clone()));
            }
        }
        let sink = Arc::new(RecordingSink {
            deliveries: StdMutex::new(Vec::new()),
        });
        let subs = Arc::new(ResourceSubscriptions::new());
        let server = Arc::new(
            MCPServer::new()
                .with_subscription_registry(subs.clone())
                .with_push_sink(sink.clone()),
        );
        // Two subscribers in two sessions.
        for (id, sess) in [(40usize, "sess-1"), (41, "sess-2")] {
            let resp = server
                .handle_request_scoped(
                    Some(sess),
                    MCPRequest::new(
                        id,
                        RESOURCES_SUBSCRIBE,
                        Some(json!({"uri": "file:///a.txt"})),
                    ),
                )
                .await;
            assert!(!resp.is_error(), "{:?}", resp.error);
        }
        let envelope = server
            .notify_resource_changed("file:///a.txt")
            .expect("watched uri must emit");
        let deliveries = sink.deliveries.lock().unwrap();
        // Both subscribers received the envelope.
        assert_eq!(deliveries.len(), 2, "each subscriber must be pushed");
        let subs_seen: Vec<&str> = deliveries.iter().map(|(s, _)| s.as_str()).collect();
        assert!(subs_seen.contains(&"sess-1"), "{deliveries:?}");
        assert!(subs_seen.contains(&"sess-2"), "{deliveries:?}");
        for (_, e) in deliveries.iter() {
            assert_eq!(*e, envelope, "every subscriber gets the same envelope");
        }
    }

    /// subscriptions are scoped by session: a subscribe from one session does
    /// not make another session a subscriber.
    #[tokio::test]
    async fn test_resources_subscribe_is_session_scoped() {
        let subs = Arc::new(ResourceSubscriptions::new());
        let server = Arc::new(MCPServer::new().with_subscription_registry(subs.clone()));
        let params = json!({"uri": "file:///a.txt"});
        let resp = server
            .handle_request_scoped(
                Some("sess-1"),
                MCPRequest::new(31, RESOURCES_SUBSCRIBE, Some(params)),
            )
            .await;
        assert!(!resp.is_error());
        assert!(subs.is_subscribed("file:///a.txt", "sess-1"));
        assert!(!subs.is_subscribed("file:///a.txt", "default"));
        assert!(!subs.is_subscribed("file:///a.txt", "sess-2"));
    }

    /// resources/unsubscribe clears the interest; after the last subscriber
    /// leaves, a change no longer emits a notification.
    #[tokio::test]
    async fn test_resources_unsubscribe_stops_notifications() {
        let subs = Arc::new(ResourceSubscriptions::new());
        let server = Arc::new(MCPServer::new().with_subscription_registry(subs.clone()));
        let uri = json!({"uri": "file:///a.txt"});
        server
            .handle_request_scoped(
                Some("sess-1"),
                MCPRequest::new(32, RESOURCES_SUBSCRIBE, Some(uri.clone())),
            )
            .await;
        assert!(server.notify_resource_changed("file:///a.txt").is_some());

        let resp = server
            .handle_request_scoped(
                Some("sess-1"),
                MCPRequest::new(33, RESOURCES_UNSUBSCRIBE, Some(uri)),
            )
            .await;
        assert!(!resp.is_error(), "{:?}", resp.error);
        assert!(!subs.is_watched("file:///a.txt"));
        assert!(
            server.notify_resource_changed("file:///a.txt").is_none(),
            "unwatched resource must not emit traffic"
        );
    }

    /// resources/subscribe without a `uri` param → invalid_params.
    #[tokio::test]
    async fn test_resources_subscribe_missing_uri() {
        let server =
            MCPServer::new().with_subscription_registry(Arc::new(ResourceSubscriptions::new()));
        let resp = server
            .handle_request(MCPRequest::new(34, RESOURCES_SUBSCRIBE, None))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    /// resources/subscribe without a registry → method_not_found (honest boundary).
    #[tokio::test]
    async fn test_resources_subscribe_not_registered() {
        let server = MCPServer::new();
        let params = json!({"uri": "file:///a.txt"});
        let resp = server
            .handle_request(MCPRequest::new(35, RESOURCES_SUBSCRIBE, Some(params)))
            .await;
        assert!(resp.is_error());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    // ============================================================================
    // S10 two server→host primitives: injected mock callbacks succeed; without a callback, a clear error
    // ============================================================================

    struct MockSampling;
    #[async_trait::async_trait]
    impl SamplingHandler for MockSampling {
        async fn create_message(
            &self,
            request: &SamplingRequest,
        ) -> Result<SamplingResult, MCPError> {
            Ok(SamplingResult {
                role: SamplingRole::Assistant,
                content: SamplingContent::Text {
                    text: format!("echo: {}", request.max_tokens),
                },
                model: None,
                stop_reason: Some("endTurn".to_string()),
            })
        }
    }

    struct MockElicitation;
    #[async_trait::async_trait]
    impl ElicitationHandler for MockElicitation {
        async fn create(
            &self,
            request: &ElicitationRequest,
        ) -> Result<ElicitationResponse, MCPError> {
            Ok(ElicitationResponse {
                action: ElicitationAction::Accept,
                content: Some(json!({ "answer": request.message })),
            })
        }
    }

    fn sampling_request() -> SamplingRequest {
        SamplingRequest {
            messages: vec![SamplingMessage {
                role: SamplingRole::User,
                content: SamplingContent::Text {
                    text: "hi".to_string(),
                },
            }],
            max_tokens: 42,
            system_prompt: None,
            model_preferences: None,
            temperature: None,
            stop_sequences: None,
            include_context: None,
            metadata: None,
        }
    }

    #[tokio::test]
    async fn test_sampling_create_message_with_handler() {
        let server = MCPServer::new().with_sampling_handler(Arc::new(MockSampling));
        let result = server
            .create_message(&sampling_request())
            .await
            .expect("注入 handler 后应成功");
        assert!(matches!(result.role, SamplingRole::Assistant));
        match result.content {
            SamplingContent::Text { text } => assert_eq!(text, "echo: 42"),
            SamplingContent::Image { .. } => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn test_sampling_create_message_without_handler() {
        let server = MCPServer::new();
        let err = server
            .create_message(&sampling_request())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("sampling handler not configured"),
            "无回调应返回明确错误,实际: {err}"
        );
    }

    #[tokio::test]
    async fn test_elicitation_create_with_handler() {
        let server = MCPServer::new().with_elicitation_handler(Arc::new(MockElicitation));
        let req = ElicitationRequest {
            message: "proceed?".to_string(),
            schema: None,
        };
        let resp = server
            .create_elicitation(&req)
            .await
            .expect("注入 handler 后应成功");
        assert!(matches!(resp.action, ElicitationAction::Accept));
    }

    #[tokio::test]
    async fn test_elicitation_create_without_handler() {
        let server = MCPServer::new();
        let req = ElicitationRequest {
            message: "proceed?".to_string(),
            schema: None,
        };
        let err = server.create_elicitation(&req).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("elicitation handler not configured"),
            "无回调应返回明确错误,实际: {err}"
        );
    }

    #[test]
    fn test_server_message_notification_has_no_id() {
        // A notification (no id) should parse as id=None
        let json = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        assert!(msg.id.is_none());
        assert_eq!(msg.method, "notifications/initialized");
    }

    #[test]
    fn test_server_message_request_has_id() {
        let json = r#"{"jsonrpc":"2.0","id":42,"method":"tools/list"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id, Some(JsonRpcId::Number(42)));
    }

    #[test]
    fn test_server_message_string_id_parses() {
        // B8: string ids are valid JSON-RPC ids and must not be rejected.
        let json = r#"{"jsonrpc":"2.0","id":"req-7","method":"ping"}"#;
        let msg: ServerMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id, Some(JsonRpcId::String("req-7".into())));
    }

    /// A tool that never completes until its gate `Notify` fires (manual
    /// release). Used to prove a `notifications/cancelled` propagates into an
    /// in-flight tool run and aborts it.
    struct HangingTool {
        gate: Arc<tokio::sync::Notify>,
    }
    #[async_trait::async_trait]
    impl BaseTool for HangingTool {
        fn name(&self) -> &str {
            "hang"
        }
        fn description(&self) -> &str {
            "hangs until cancelled or released"
        }
        fn args_schema(&self) -> Option<Value> {
            Some(json!({"type":"object"}))
        }
        async fn run(&self, _input: String) -> Result<String, ToolError> {
            // Block until the test releases the gate manually; cancellation
            // (which fires the server's own Notify, not this gate) must abort
            // this run before the gate is ever released.
            self.gate.notified().await;
            Ok("done".into())
        }
    }

    /// B8 cancellation propagation: send `tools/call` for a hanging tool, then
    /// `notifications/cancelled` for its id — the in-flight run is aborted and
    /// the response carries the "cancelled by client" outcome.
    #[tokio::test]
    async fn test_cancel_propagates_into_in_flight_tool() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let server =
            Arc::new(MCPServer::new().with_tool(Arc::new(HangingTool { gate: gate.clone() })));
        let req = MCPRequest::new(
            42,
            "tools/call",
            Some(json!({"name": "hang", "arguments": {}})),
        );
        let server_in_task = server.clone();
        let in_flight = tokio::spawn(async move { server_in_task.handle_request(req).await });

        // Give the request a moment to register its Notify in the table.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        server
            .handle_notification("notifications/cancelled", Some(json!({"requestId": 42})))
            .await;

        let resp = in_flight.await.expect("in-flight task completes");
        let result = resp
            .result
            .expect("cancellation yields a result, not a raw error");
        let text = result["content"][0]["text"].as_str().unwrap_or_default();
        assert_eq!(text, "cancelled by client", "result: {result}");

        // The abandoned id must be removed from the cancellation table.
        let map = server.cancellations.lock().await;
        assert!(!map.contains_key(&(None, JsonRpcId::Number(42))), "{map:?}");
    }

    /// F1/H3: cancellation is scoped by session. Two concurrent sessions that
    /// reuse the same JSON-RPC id must not cancel each other's in-flight tool
    /// call — a `notifications/cancelled` for session B fires only B's request.
    #[tokio::test]
    async fn test_cancel_is_isolated_by_session() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let server =
            Arc::new(MCPServer::new().with_tool(Arc::new(HangingTool { gate: gate.clone() })));

        // Two sessions, same request id 7, both awaiting the hanging tool.
        let req_a = MCPRequest::new(
            7,
            "tools/call",
            Some(json!({"name": "hang", "arguments": {}})),
        );
        let req_b = MCPRequest::new(
            7,
            "tools/call",
            Some(json!({"name": "hang", "arguments": {}})),
        );
        let (sa, sb) = (server.clone(), server.clone());
        let (task_a, task_b) = (
            tokio::spawn(async move { sa.handle_request_scoped(Some("sess-a"), req_a).await }),
            tokio::spawn(async move { sb.handle_request_scoped(Some("sess-b"), req_b).await }),
        );

        // Let both register their Notify under their (session, id) keys.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            server.cancellations.lock().await.len(),
            2,
            "both registered"
        );

        // Cancelling session B's id 7 must not fire session A's request.
        server
            .handle_notification_scoped(
                Some("sess-b"),
                "notifications/cancelled",
                Some(json!({"requestId": 7})),
            )
            .await;

        let resp_b = task_b.await.expect("B task completes");
        let result_b = resp_b.result.unwrap();
        let text_b = result_b["content"][0]["text"].as_str().unwrap_or_default();
        assert_eq!(text_b, "cancelled by client", "B should be cancelled");

        // A is untouched: its key is still registered and its tool still hangs.
        let map = server.cancellations.lock().await;
        assert!(
            map.contains_key(&(Some("sess-a".to_string()), JsonRpcId::Number(7))),
            "session A's registration must survive B's cancellation: {map:?}"
        );
        assert!(
            !map.contains_key(&(Some("sess-b".to_string()), JsonRpcId::Number(7))),
            "session B's registration must be reaped: {map:?}"
        );
        drop(map);

        // Release A's gate so its tool can finish and the task resolves.
        gate.notify_waiters();
        let resp_a = task_a.await.expect("A task completes");
        let result_a = resp_a.result.unwrap();
        let text_a = result_a["content"][0]["text"].as_str().unwrap_or_default();
        assert_eq!(text_a, "done", "A must run to completion, uncancelled");
    }

    /// 0.26.0 P0-1 (BLOCK): a configured `with_tool_timeout` bounds a hung tool.
    /// `HangingTool` never completes, so the only way this `tools/call` resolves
    /// is the deadline branch — proving a lone stuck tool can no longer hold a
    /// concurrency permit forever (which previously took the whole server down).
    #[tokio::test]
    async fn test_tool_call_exceeding_timeout_returns_deadline_error() {
        // Gate never released in this test → the tool would hang forever; only
        // the 100 ms deadline can resolve the call.
        let server = MCPServer::new()
            .with_tool(Arc::new(HangingTool {
                gate: Arc::new(tokio::sync::Notify::new()),
            }))
            .with_tool_timeout(std::time::Duration::from_millis(100));

        let resp = server
            .handle_request(MCPRequest::new(
                900,
                "tools/call",
                Some(json!({"name": "hang", "arguments": {}})),
            ))
            .await;

        assert!(
            !resp.is_error(),
            "timeout surfaces as a tool-result, not a transport error"
        );
        let result = resp.result.expect("has result");
        let text = result["content"][0]["text"].as_str().unwrap_or_default();
        assert!(
            text.contains("deadline"),
            "expect the deadline observation, got: {text}"
        );
        assert!(
            result["isError"].as_bool() == Some(true),
            "timeout must be flagged isError so the caller does not treat it as success"
        );

        // The registration is reaped even when the deadline fires.
        assert!(
            server.cancellations.lock().await.is_empty(),
            "cancellation registration must be cleaned up after timeout"
        );
    }

    /// 0.26.0 P0-1: default (no `with_tool_timeout` override) should carry a bound
    /// — the security posture is timeout-by-default, never timeout-by-omission.
    #[test]
    fn test_tool_timeout_is_some_by_default() {
        assert!(
            MCPServer::new().tool_timeout.is_some(),
            "hardened default must bound hung tools"
        );
    }

    #[tokio::test]
    async fn test_handle_notification_known_and_unknown() {
        let server = server_with_echo();
        // Standard notifications should be handled (no panic)
        server
            .handle_notification("notifications/cancelled", Some(json!({"requestId": 1})))
            .await;
        server
            .handle_notification(
                "notifications/progress",
                Some(json!({"token": 1, "progress": 0.5})),
            )
            .await;
        server
            .handle_notification("notifications/roots/list_changed", None)
            .await;
        server
            .handle_notification("notifications/initialized", None)
            .await;
        // Unknown notifications should be ignored
        server.handle_notification("foo/bar", None).await;
    }

    // ------------------------------------------------------------------
    // 0.22.0 C8: serve_http hardening
    // ------------------------------------------------------------------

    use crate::auth::StaticBearerValidator;

    /// POSTs one raw HTTP request and returns (status_line, body).
    async fn post_raw(addr: &str, authorization: Option<&str>, body: &str) -> (String, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut req = format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        if let Some(auth) = authorization {
            req = req.replacen(
                "Content-Type: application/json",
                &format!("Content-Type: application/json\r\nAuthorization: {auth}"),
                1,
            );
        }
        sock.write_all(req.as_bytes()).await.unwrap();
        let mut resp = Vec::new();
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            sock.read_to_end(&mut resp),
        )
        .await;
        let text = String::from_utf8_lossy(&resp).to_string();
        let (status, rest) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        (status.to_string(), rest.to_string())
    }

    /// C8: without a token validator the server answers normally (back-compat).
    #[tokio::test]
    async fn test_serve_http_open_when_no_validator() {
        let server = Arc::new(server_with_echo());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let _url = server.serve_http(listener);
        let body = serde_json::to_string(&MCPRequest::new(1, "tools/list", None)).unwrap();
        let (status, resp_body) = post_raw(&addr, None, &body).await;
        assert!(status.contains("200 OK"), "{status}");
        assert!(resp_body.contains("echo"), "{resp_body}");
    }

    /// M-2: a public (non-loopback) bind with no token validator must be refused
    /// up front — an open MCP server would let anyone on the network invoke tools.
    #[tokio::test]
    #[should_panic(expected = "refusing to serve MCP on non-loopback bind")]
    async fn test_serve_http_refuses_public_bind_without_validator() {
        let server = Arc::new(server_with_echo());
        let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
        let _url = server.serve_http(listener);
    }

    /// C8: with a validator, missing/wrong token → 401, correct token → 200.
    #[tokio::test]
    async fn test_serve_http_enforces_token_validator() {
        let server = Arc::new(
            server_with_echo()
                .with_token_validator(Arc::new(StaticBearerValidator::new("secret-token"))),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let _url = server.serve_http(listener);
        let body = serde_json::to_string(&MCPRequest::new(1, "tools/list", None)).unwrap();

        let (status, _) = post_raw(&addr, None, &body).await;
        assert!(status.contains("401"), "missing token: {status}");

        let (status, _) = post_raw(&addr, Some("Bearer wrong"), &body).await;
        assert!(status.contains("401"), "wrong token: {status}");

        let (status, resp_body) = post_raw(&addr, Some("Bearer secret-token"), &body).await;
        assert!(status.contains("200 OK"), "valid token: {status}");
        assert!(resp_body.contains("echo"), "{resp_body}");
    }

    /// C8: an unparseable body is answered with a JSON-RPC parse error and
    /// `id: null` (JSON-RPC spec; previously `id: 0`).
    #[tokio::test]
    async fn test_serve_http_parse_error_id_null() {
        let server = Arc::new(server_with_echo());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let _url = server.serve_http(listener);
        let (status, resp_body) = post_raw(&addr, None, "this is not json").await;
        assert!(status.contains("200 OK"), "{status}");
        assert!(resp_body.contains("\"id\":null"), "{resp_body}");
        assert!(resp_body.contains("-32700"), "{resp_body}");
    }

    /// M2 (0.26.0 §1.1): a JSON-RPC *notification* (no id) over the stateless
    /// HTTP track is dispatched to `handle_notification` and ACKed 202 — it
    /// must NOT be rejected with `-32700` (the old strict `MCPRequest` parse
    /// required `id`, so a notification was misread as unparseable).
    #[tokio::test]
    async fn test_serve_http_notification_dispatched_not_parse_error() {
        let server = Arc::new(server_with_echo());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let _url = server.serve_http(listener);
        // A notification body: method + params, NO id.
        let body = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}"#;
        let (status, resp_body) = post_raw(&addr, None, body).await;
        // Per JSON-RPC, notifications get no response — ACK with 202, empty body.
        assert!(
            status.contains("202 Accepted"),
            "notification ACK expected, got: {status}"
        );
        assert!(
            resp_body.is_empty(),
            "notification must get no JSON payload, got: {resp_body}"
        );
        assert!(
            !resp_body.contains("-32700"),
            "notification must not be misread as a parse error: {resp_body}"
        );
    }

    /// C8: an oversized declared body is rejected with 413, not buffered.
    #[tokio::test]
    async fn test_serve_http_rejects_oversized_body() {
        let server = Arc::new(server_with_echo());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let _url = server.serve_http(listener);
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut sock = tokio::net::TcpStream::connect(&addr).await.unwrap();
        let header = format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\n\r\n",
            HTTP_MAX_BODY_BYTES + 1
        );
        sock.write_all(header.as_bytes()).await.unwrap();
        let mut resp = Vec::new();
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            sock.read_to_end(&mut resp),
        )
        .await;
        let text = String::from_utf8_lossy(&resp).to_string();
        assert!(
            text.starts_with("HTTP/1.1 400") || text.starts_with("HTTP/1.1 413"),
            "oversized body should be rejected, got: {text}"
        );
    }
}
