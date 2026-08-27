#![warn(missing_docs)]
//! MCP (Model Context Protocol) support
//!
//! MCP is the tool protocol standard introduced by Anthropic and has become the de-facto industry standard.
//! This module provides an MCP Client that can connect to any MCP Server to obtain tool capabilities,
//! and adapts MCP tools into `BaseTool` for use by Agents.
//!
//! # Example
//! ```no_run
//! use lc_mcp::{MCPClient, MCPConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MCPConfig::stdio(
//!     "npx",
//!     vec!["@anthropic/mcp-server-filesystem".to_string(), "/tmp".to_string()],
//! );
//! let mut client = MCPClient::connect(config).await?;
//! let tools = client.list_tools().await?;
//! println!("MCP 工具数量: {}", tools.len());
//! # Ok(())
//! # }
//! ```

pub mod client;
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
mod sse;
pub mod stream;
pub mod tenant;
pub mod tool_adapter;
pub mod tool_discovery;
pub mod tool_namespace;
pub mod tool_timeout;
pub mod transport;
pub mod types;

#[cfg(test)]
mod test_support;

pub use client::MCPClient;
pub use completion::{
    CompletionArgument, CompletionProvider, CompletionRef, CompletionRequest, CompletionResult,
    CompletionValue,
};
pub use connection_manager::{ConnectionManager, ServerSpec};
pub use elicitation::{
    ElicitationAction, ElicitationHandler, ElicitationRequest, ElicitationResponse,
};
pub use gateway::{GatewayAuditRecord, GatewayServerSpec, MCPGateway, RateLimiter};
pub use health::{probe_health, BreakerState, CircuitBreaker, HealthStatus, ServerHealth};
pub use orchestrate::{OrchestrateError, ToolCaller, ToolOrchestrator, ToolStep};
pub use prompts::{
    GetPromptParams, GetPromptResult, ListPromptsResult, Prompt, PromptArgument, PromptContent,
    PromptMessage, PromptProvider,
};
pub use protocol::{
    MCPError, MCPRequest, MCPResponse, ProtocolInfo, VersionPolicy, MCP_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
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
pub use stream::{PartialContent, ToolStream, ToolStreamError};
pub use tenant::TenantGateway;
pub use tool_adapter::MCPToolAdapter;
pub use tool_discovery::{KeywordScorer, ToolDiscovery, ToolScorer};
pub use tool_namespace::{NamespacedTool, ToolConflict, ToolNamespace};
pub use tool_timeout::{call_tool_with_timeout, ToolSpec};
pub use transport::{InMemoryTransport, MCPEvent, MCPTransport, SseTransport, StdioTransport};
pub use types::{MCPConfig, MCPContent, MCPToolDefinition, MCPToolResult};
