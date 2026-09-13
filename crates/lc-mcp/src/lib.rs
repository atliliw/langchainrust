#![warn(missing_docs)]
//! MCP (Model Context Protocol) support — three client tracks (0.22.4 B1).
//!
//! MCP is the tool protocol standard introduced by Anthropic and has become
//! the de-facto industry standard.
//!
//! - [`StatelessMcpClient`] — the 2026-07-28 self-contained HTTP POST track:
//!   every request carries `_meta` + `Mcp-Method`/`Mcp-Name` headers, no
//!   handshake, no session. The framework's extension track.
//! - [`StdioMcpClient`] — the official MCP stdio track: spawns a local
//!   subprocess and speaks newline-delimited JSON-RPC with the standard
//!   `initialize` handshake, interoperating with mainstream MCP servers
//!   (Claude Desktop / official SDK servers).
//! - [`StreamableMcpClient`] — the official MCP Streamable HTTP track:
//!   JSON-RPC POSTs answered by a direct JSON body or an SSE stream, optional
//!   server-assigned `Mcp-Session-Id`, and OAuth 2.1 bearer auth
//!   ([`BearerTokenProvider`]) for remote servers.
//!
//! All three implement [`McpToolClient`], so any track drops into
//! [`MCPToolAdapter::from_client`] as agent tools.
//!
//! # Example
//! ```no_run
//! use lc_mcp::StatelessMcpClient;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = StatelessMcpClient::connect("https://host/mcp");
//! let tools = client.list_tools().await?;
//! println!("MCP 工具数量: {}", tools.len());
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod client_stateless;
pub mod client_stdio;
pub mod client_streamable;
pub mod completion;
pub mod connection_manager;
pub mod elicitation;
pub mod execution;
pub mod gateway;
pub mod health;
pub mod oauth2;
pub mod orchestrate;
pub mod prompts;
pub mod protocol;
pub mod resources;
pub mod roots;
pub mod sampling;
pub mod sandbox;
pub mod server;
pub mod tasks;
pub mod tenant;
pub mod tool_adapter;
pub mod tool_client;
pub mod tool_discovery;
pub mod tool_namespace;
pub mod tool_timeout;
pub mod transport;
pub mod types;

#[cfg(test)]
mod test_support;

/// Claim-only JWT validator with no signature check (A15): hidden; use
/// [`JwtIssValidator`] with a JWKS/introspection verifier in real deployments.
#[doc(hidden)]
pub use auth::JwtIssAssertionValidator;
pub use auth::{
    AuthScheme, Claims, JwtIssValidator, JwtSignatureVerifier, StaticBearerValidator,
    TokenValidator,
};
pub use client_stateless::{
    CannedAnswerProvider, MrtrAnswerProvider, MrtrConfig, StatelessMcpClient,
};
pub use client_stdio::StdioMcpClient;
pub use client_streamable::StreamableMcpClient;
pub use completion::{
    CompletionArgument, CompletionProvider, CompletionRef, CompletionRequest, CompletionResult,
    CompletionValue,
};
pub use connection_manager::{ConnectionManager, ServerSpec};
pub use elicitation::{
    ElicitationAction, ElicitationHandler, ElicitationRequest, ElicitationResponse,
};
pub use execution::{ToolCallApprover, ToolExecutionPolicy};
pub use gateway::{
    GatewayAuditRecord, GatewayServerSpec, MCPGateway, MethodRateLimiter, RateLimiter,
};
pub use health::{probe_health, BreakerState, CircuitBreaker, HealthStatus, ServerHealth};
pub use oauth2::{
    discover_authorization_server, discover_protected_resource, AuthorizationServerMetadata,
    BearerTokenProvider, OAuthChallenge, OAuthTokenClient, OAuthTokenResponse,
    ProtectedResourceMetadata, StaticBearerToken,
};
pub use orchestrate::{OrchestrateError, ToolCaller, ToolOrchestrator, ToolStep};
pub use prompts::{
    GetPromptParams, GetPromptResult, ListPromptsResult, Prompt, PromptArgument, PromptContent,
    PromptMessage, PromptProvider,
};
pub use protocol::{
    negotiate_protocol_version, notification_message, ClientIdentity, InputRequired, MCPError,
    MCPRequest, MCPResponse, MrtrAnswer, MrtrQuestion, ProtocolInfo, RequestMeta, VersionPolicy,
    MCP_ERROR_REQUEST_TIMEOUT, MCP_ERROR_SESSION_LOST, MCP_ERROR_UNAUTHORIZED,
    MCP_ERROR_VERSION_UNSUPPORTED, MCP_METHOD_HEADER, MCP_NAME_HEADER, MCP_VERSION,
    MCP_VERSION_STATELESS, SUPPORTED_PROTOCOL_VERSIONS,
};
pub use resources::{
    ListResourcesResult, ReadResourceParams, ReadResourceResult, Resource, ResourceContent,
    ResourceProvider,
};
pub use sampling::{
    ModelHint, ModelPreferences, SamplingContent, SamplingGuard, SamplingGuardError,
    SamplingHandler, SamplingLease, SamplingMessage, SamplingRequest, SamplingResult, SamplingRole,
};
pub use sandbox::{
    AuditRecord, EgressPolicy, ParamRule, ParamRuleError, SandboxError, ServerSandbox,
};
pub use server::MCPServer;
pub use tasks::McpTaskHandle;
pub use tenant::TenantGateway;
pub use tool_adapter::MCPToolAdapter;
pub use tool_client::McpToolClient;
pub use tool_discovery::{KeywordScorer, ToolDiscovery, ToolScorer};
pub use tool_namespace::{NamespacedTool, ToolConflict, ToolNamespace};
pub use tool_timeout::{call_tool_with_timeout, ToolSpec};
pub use transport::{
    default_meta, StatelessTransport, StdioCommand, StdioTransport, StreamableHttpTransport,
};
pub use types::{MCPContent, MCPToolDefinition, MCPToolResult};
