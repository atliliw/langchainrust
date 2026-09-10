#![warn(missing_docs)]
//! MCP (Model Context Protocol) support — 2026-07-28 stateless track only.
//!
//! MCP is the tool protocol standard introduced by Anthropic and has become the de-facto industry standard.
//! This module provides a stateless MCP Client that can connect to any stateless MCP Server to obtain tool
//! capabilities, and adapts MCP tools into `BaseTool` for use by Agents. Every request is self-contained
//! (`_meta` + `Mcp-Method`/`Mcp-Name` headers) — no handshake, no session.
//! The legacy handshake track (SSE/stdio/MCPClient) was removed in 0.22.0; see
//! `docs/internal/v0.22.0/MIGRATION.md`.
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
pub mod completion;
pub mod connection_manager;
pub mod elicitation;
pub mod gateway;
pub mod health;
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
pub mod tool_discovery;
pub mod tool_namespace;
pub mod tool_timeout;
pub mod transport;
pub mod types;

#[cfg(test)]
mod test_support;

pub use auth::{AuthScheme, Claims, JwtIssValidator, StaticBearerValidator, TokenValidator};
pub use client_stateless::{
    CannedAnswerProvider, MrtrAnswerProvider, MrtrConfig, StatelessMcpClient,
};
pub use completion::{
    CompletionArgument, CompletionProvider, CompletionRef, CompletionRequest, CompletionResult,
    CompletionValue,
};
pub use connection_manager::{ConnectionManager, ServerSpec};
pub use elicitation::{
    ElicitationAction, ElicitationHandler, ElicitationRequest, ElicitationResponse,
};
pub use gateway::{
    GatewayAuditRecord, GatewayServerSpec, MCPGateway, MethodRateLimiter, RateLimiter,
};
pub use health::{probe_health, BreakerState, CircuitBreaker, HealthStatus, ServerHealth};
pub use orchestrate::{OrchestrateError, ToolCaller, ToolOrchestrator, ToolStep};
pub use prompts::{
    GetPromptParams, GetPromptResult, ListPromptsResult, Prompt, PromptArgument, PromptContent,
    PromptMessage, PromptProvider,
};
pub use protocol::{
    ClientIdentity, InputRequired, MCPError, MCPRequest, MCPResponse, MrtrAnswer, MrtrQuestion,
    ProtocolInfo, RequestMeta, VersionPolicy, MCP_ERROR_UNAUTHORIZED, MCP_METHOD_HEADER,
    MCP_NAME_HEADER, MCP_VERSION, MCP_VERSION_STATELESS, SUPPORTED_PROTOCOL_VERSIONS,
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
pub use tool_discovery::{KeywordScorer, ToolDiscovery, ToolScorer};
pub use tool_namespace::{NamespacedTool, ToolConflict, ToolNamespace};
pub use tool_timeout::{call_tool_with_timeout, ToolSpec};
pub use transport::{default_meta, StatelessTransport};
pub use types::{MCPContent, MCPToolDefinition, MCPToolResult};
