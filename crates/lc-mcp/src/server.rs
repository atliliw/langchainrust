//! MCP Server - exposes local `BaseTool`s as an MCP Server for other Hosts (Claude Desktop/Cursor etc.) to call
//!
//! Symmetric to `MCPClient`: the Client connects to another's Server to use tools, the Server exposes its own
//! tools to others.
//! Supports the `initialize` handshake, `tools/list`, `tools/call`, plus registration-based primitives
//! `resources/*` / `prompts/*` / `completion/complete` (still returning `method_not_found` when unregistered,
//! an honest boundary) and the server→host direction `sampling::create_message` /
//! `elicitation::create` (requires an injected callback).

use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::broadcast;

use super::completion::{CompletionProvider, CompletionRequest};
use super::elicitation::{ElicitationHandler, ElicitationRequest, ElicitationResponse};
use super::prompts::{ListPromptsResult, PromptProvider};
use super::protocol::{
    MCPError, MCPRequest, MCPResponse, MCP_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
use super::resources::{ListResourcesResult, ReadResourceResult, ResourceProvider};
use super::sampling::{SamplingHandler, SamplingRequest, SamplingResult};
use super::stream::PartialContent;
use super::types::{MCPContent, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// MCP Server - exposes a set of `BaseTool`s as MCP tools
pub struct MCPServer {
    tools: Vec<Arc<dyn BaseTool>>,
    server_name: String,
    server_version: String,
    /// Streaming tool-output broadcast (P2-9): `publish_partial` pushes incremental chunks,
    /// and transport layers such as `InMemoryTransport` subscribe and forward them to the client.
    partial_tx: broadcast::Sender<PartialContent>,
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
            partial_tx: broadcast::channel(64).0,
            resources: None,
            prompts: None,
            completion: None,
            sampling_handler: None,
            elicitation_handler: None,
        }
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

    /// Pushes one streaming tool-output incremental chunk (P2-9).
    ///
    /// Long-running tools "stream while they run": during execution they split partial results into chunks and
    /// push each via `publish_partial` to connected Hosts (transport layers such as `InMemoryTransport` subscribe
    /// and forward them to the client as `notifications/tool_partial`). With no subscribers the chunk is silently
    /// dropped — the incremental output is a preview, the final result is still carried by the `tools/call` response.
    pub fn publish_partial(&self, partial: PartialContent) {
        let _ = self.partial_tx.send(partial);
    }

    /// Subscribes to the incremental streaming chunks this server pushes (P2-9).
    ///
    /// For transport layers (such as [`InMemoryTransport`](crate::InMemoryTransport)) to turn the server-side
    /// `publish_partial` into client-visible push events.
    pub fn subscribe_partials(&self) -> broadcast::Receiver<PartialContent> {
        self.partial_tx.subscribe()
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

    /// Serves MCP over an SSE network service on an already-bound TCP listener, returning the SSE entry URL
    /// clients connect to.
    ///
    /// This is the "deployable MCP server" entry point: it exposes this server as an HTTP/SSE service that any
    /// MCP client (`MCPClient::connect(MCPConfig::sse(...))` / Cursor / Claude Desktop etc.) can connect to and
    /// use the registered tools. The SSE frame format aligns with the `MCPClient` (SseTransport) client behavior.
    ///
    /// - `listener`: a `TcpListener` already bound to an address. For local debugging bind `127.0.0.1:0`;
    ///   for remote deployment bind `0.0.0.0:PORT`.
    /// - `public_base`: the base address clients use to reach this server (e.g. `http://your-server-ip:8788`).
    ///   The POST address the server sends to clients is built from it; when deployed remotely it must be an
    ///   address clients can really reach (not `0.0.0.0`).
    ///
    /// Returns the SSE URL immediately after startup; the accept loop runs on a background task until the
    /// process exits.
    pub fn serve_sse(
        self: Arc<Self>,
        listener: tokio::net::TcpListener,
        public_base: impl Into<String>,
    ) -> String {
        crate::sse::serve(self, listener, public_base.into())
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
}
