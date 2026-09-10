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
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Semaphore;

use super::auth::TokenValidator;
use super::completion::{CompletionProvider, CompletionRequest};
use super::elicitation::{ElicitationHandler, ElicitationRequest, ElicitationResponse};
use super::prompts::{ListPromptsResult, PromptProvider};
use super::protocol::{
    MCPError, MCPRequest, MCPResponse, MCP_VERSION, MCP_VERSION_STATELESS,
    SUPPORTED_PROTOCOL_VERSIONS,
};
use super::resources::{ListResourcesResult, ReadResourceResult, ResourceProvider};
use super::sampling::{SamplingHandler, SamplingRequest, SamplingResult};
use super::types::{MCPContent, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// 0.22.0 C8 hardening: maximum accepted request body (1 MiB). A larger
/// `Content-Length` is rejected with 413 and a stream exceeding it aborts —
/// without the cap a single connection could grow `buf` unboundedly (OOM).
const HTTP_MAX_BODY_BYTES: usize = 1024 * 1024;
/// 0.22.0 C8 hardening: per-request socket read deadline. A slow-loris
/// connection that never finishes its headers is reaped instead of holding
/// a task + file descriptor forever.
const HTTP_READ_TIMEOUT_SECS: u64 = 30;
/// 0.22.0 C8 hardening: maximum concurrently served connections. Accept waits
/// for a permit instead of spawning unbounded tasks.
const HTTP_MAX_CONCURRENT_CONNECTIONS: usize = 256;

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
    /// Optional sampling callback (S10, server→host direction): once injected, `create_message` can fire.
    sampling_handler: Option<Arc<dyn SamplingHandler>>,
    /// Optional elicitation callback (S10, server→host direction): once injected, `create_elicitation` can fire.
    elicitation_handler: Option<Arc<dyn ElicitationHandler>>,
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
            completion: None,
            sampling_handler: None,
            elicitation_handler: None,
        }
    }

    /// 0.22.0 C8: requires authenticated requests on [`Self::serve_http`].
    ///
    /// Every request must carry `Authorization: Bearer <token>` and pass
    /// `validator.validate(token)`; failures get HTTP 401. Recommended for
    /// any non-loopback binding (see the deployment guide).
    pub fn with_token_validator(mut self, validator: Arc<dyn TokenValidator>) -> Self {
        self.auth_validator = Some(validator);
        self
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
    pub fn with_completion_provider(mut self, provider: Arc<dyn CompletionProvider>) -> Self {
        self.completion = Some(provider);
        self
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
            "tools/list" => {
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
                    result: Some(json!({ "tools": tools_val })),
                    error: None,
                }
            }
            "tools/call" => self.handle_tools_call(req).await,
            // 2026-07-28 stateless track: on-demand capability discovery —
            // self-contained requests ask what the server can do without a
            // prior initialize handshake.
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
                MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": MCP_VERSION_STATELESS,
                        "capabilities": capabilities,
                        "serverInfo": { "name": self.server_name, "version": self.server_version }
                    })),
                    error: None,
                }
            }
            // S10 five client→server primitives: registered → correct structure, unregistered → method_not_found.
            "resources/list" => self.handle_resources_list(req).await,
            "resources/read" => self.handle_resources_read(req).await,
            "prompts/list" => self.handle_prompts_list(req).await,
            "prompts/get" => self.handle_prompts_get(req).await,
            "completion/complete" => self.handle_completion_complete(req).await,
            _ => Self::method_not_found_response(req.id),
        }
    }

    /// `resources/list`: lists the registered resources.
    async fn handle_resources_list(&self, req: MCPRequest) -> MCPResponse {
        match &self.resources {
            Some(provider) => match provider.list_resources().await {
                Ok(resources) => {
                    let result = serde_json::to_value(ListResourcesResult { resources })
                        .unwrap_or(Value::Null);
                    Self::ok_response(req.id, result)
                }
                Err(e) => Self::error_response(req.id, e),
            },
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

    /// `prompts/list`: lists the registered prompts.
    async fn handle_prompts_list(&self, req: MCPRequest) -> MCPResponse {
        match &self.prompts {
            Some(provider) => match provider.list_prompts().await {
                Ok(prompts) => {
                    let result =
                        serde_json::to_value(ListPromptsResult { prompts }).unwrap_or(Value::Null);
                    Self::ok_response(req.id, result)
                }
                Err(e) => Self::error_response(req.id, e),
            },
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
    fn ok_response(id: u64, result: Value) -> MCPResponse {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    /// Builds a response carrying a JSON-RPC error.
    fn error_response(id: u64, error: MCPError) -> MCPResponse {
        MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            result: None,
            error: Some(error),
        }
    }

    /// Builds a `method_not_found` (-32601) response: shared by unregistered capabilities / unknown methods.
    fn method_not_found_response(id: u64) -> MCPResponse {
        Self::error_response(id, MCPError::method_not_found())
    }

    /// Builds an `invalid_params` (-32602) response.
    fn invalid_params_response(id: u64, msg: impl Into<String>) -> MCPResponse {
        Self::error_response(id, MCPError::invalid_params(msg))
    }

    async fn handle_tools_call(&self, req: MCPRequest) -> MCPResponse {
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
                let input_str = serde_json::to_string(&arguments).unwrap_or_else(|_| "null".into());
                let result = t.run(input_str).await;
                let mcp_result = match result {
                    Ok(text) => MCPToolResult {
                        content: vec![MCPContent::Text { text }],
                        is_error: false,
                    },
                    Err(e) => MCPToolResult {
                        content: vec![MCPContent::Text {
                            text: e.to_string(),
                        }],
                        is_error: true,
                    },
                };
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
    /// Notifications (messages without an id, such as `notifications/initialized`) are ignored; requests
    /// (with an id) get a response.
    pub async fn serve_stdio(&self) -> Result<(), MCPError> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut stdout = tokio::io::stdout();

        loop {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
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
    ///   debugging bind `127.0.0.1:0`; for remote deployment bind
    ///   `0.0.0.0:PORT`.
    ///
    /// Returns the endpoint URL immediately after startup; the accept loop
    /// runs on a background task until the process exits.
    pub fn serve_http(self: Arc<Self>, listener: tokio::net::TcpListener) -> String {
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
                        let request =
                            match tokio::time::timeout(
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

                        // JSON-RPC over HTTP: a parse failure is answered with a
                        // JSON-RPC error envelope carrying `id: null` (spec-compliant;
                        // previously `id: 0`).
                        let resp = match serde_json::from_str::<MCPRequest>(&body) {
                            Ok(req) => server.handle_request(req).await,
                            Err(e) => MCPResponse {
                                jsonrpc: "2.0".to_string(),
                                id: None,
                                result: None,
                                error: Some(MCPError::new(-32700, format!("parse error: {e}"))),
                            },
                        };
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
        match method {
            "notifications/cancelled" => {
                // Carries requestId, pointing at the request to cancel
                log::info!("MCP received cancelled notification: {:?}", params);
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
    id: Option<u64>,
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
struct HttpRequest {
    first_line: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

/// Reads one HTTP request from the socket.
/// `Err(())` = transport closed (or an oversized header line); the caller
/// closes the connection. Body reads are bounded by [`HTTP_MAX_BODY_BYTES`].
async fn read_http_request(sock: &mut tokio::net::TcpStream) -> Result<HttpRequest, ()> {
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
    use crate::prompts::{GetPromptResult, Prompt, PromptContent, PromptMessage};
    use crate::resources::{Resource, ResourceContent};
    use crate::sampling::{SamplingContent, SamplingMessage, SamplingRole};
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
        assert_eq!(msg.id, Some(42));
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
    async fn post_raw(
        addr: &str,
        authorization: Option<&str>,
        body: &str,
    ) -> (String, String) {
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
