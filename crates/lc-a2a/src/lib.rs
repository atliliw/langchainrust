//! A2A (Agent-to-Agent) protocol support.
//!
//! This module implements the A2A protocol for inter-agent communication.
//! It enables LangChain Rust agents to discover, invoke, and communicate
//! with other agents over HTTP using a JSON-RPC style protocol.
//!
//! # Architecture
//!
//! - **protocol**: Core data types (`AgentCard`, `A2ATask`, `A2ARequest`, etc.)
//! - **server**: `A2AServer` - handler functions to expose an agent via A2A
//! - **client**: `A2AClient` - HTTP client to connect to remote A2A agents
//! - **rate_limiter**: `RateLimiter` - concurrency + per-minute request limits
//!
//! # Quick Start
//!
//! ## Server (expose your agent)
//!
//! ```ignore
//! use lc_a2a::{A2AServer, AgentCard};
//! use lc_chains::LLMChain;
//! use std::sync::Arc;
//!
//! let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
//! let server = A2AServer::new(chain)
//!     .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));
//!
//! // In your HTTP handler (axum, actix, warp, etc.):
//! // GET /.well-known/agent-card.json -> server.get_agent_card()
//! // POST / -> server.handle_a2a_request(body).await
//! ```
//!
//! ## Client (call a remote agent)
//!
//! ```ignore
//! use lc_a2a::{A2AClient, A2AMessage};
//!
//! let client = A2AClient::new("http://localhost:8080".to_string()).unwrap();
//! let card = client.get_agent_card().await?;
//! let task = client.send_task(A2AMessage::user("hello")).await?;
//! ```

pub mod agent_adapter;
pub mod client;
pub mod discovery;
pub mod gateway;
pub mod protocol;
pub mod rate_limiter;
pub mod resilient;
pub mod router;
pub mod scale;
pub mod security;
pub mod server;
#[cfg(feature = "axum")]
pub mod server_impl;
pub mod store;

pub use agent_adapter::AgentExecutorChain;
pub use client::{A2AClient, A2AClientBuilder, A2AError};
pub use discovery::{AgentRegistry, RegistryClient, RegistryError};
pub use gateway::{CallPolicy, DataContract, FederationGateway, GatewayError};
pub use protocol::{
    metadata_keys, A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskDetails,
    A2ATaskResult, A2AWorkflow, AgentCard, AgentSkill, MessageEnvelope, TaskFilter,
    TaskPushNotification, TaskStatus, TraceContext, WorkflowStep,
};
pub use rate_limiter::{RateLimitError, RateLimiter};
pub use resilient::{ResilienceConfig, ResilientA2AClient};
pub use router::{SkillMapRouter, SkillRouter};
pub use scale::{
    AgentTier, BreakerState, CircuitBreaker, CircuitBreakerConfig, DelegationGuard,
    HierarchyPolicy, ScaleError, SkillEntry, SkillIndex, StickyRouter, TaskGraph, TaskSharder,
};
pub use security::{
    AccessRequest, SandboxConfig, SecurityError, TrustConfig, TrustRegistry, TrustRole,
    TrustVerification, TrustedAgent,
};
pub use server::A2AServer;
pub use store::{in_memory_store, InMemoryTaskStore, StoreError, StoredTask, TaskStore};
