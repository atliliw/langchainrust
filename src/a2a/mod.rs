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
//!
//! # Quick Start
//!
//! ## Server (expose your agent)
//!
//! ```ignore
//! use langchainrust::a2a::{A2AServer, AgentCard};
//! use langchainrust::LLMChain;
//! use std::sync::Arc;
//!
//! let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
//! let server = A2AServer::new(chain)
//!     .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));
//!
//! // In your HTTP handler (axum, actix, warp, etc.):
//! // GET /.well-known/agent.json -> server.get_agent_card()
//! // POST / -> server.handle_a2a_request(body).await
//! ```
//!
//! ## Client (call a remote agent)
//!
//! ```ignore
//! use langchainrust::a2a::{A2AClient, A2AMessage};
//!
//! let client = A2AClient::new("http://localhost:8080".to_string());
//! let card = client.get_agent_card().await?;
//! let task = client.send_task(A2AMessage::user("hello")).await?;
//! ```

pub mod client;
pub mod protocol;
pub mod server;

pub use client::{A2AClient, A2AError};
pub use protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskResult, AgentCard,
    TaskStatus,
};
pub use server::A2AServer;
