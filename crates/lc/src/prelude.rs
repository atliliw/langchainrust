//! The main path: five modules, one import.
//!
//! ```
//! use langchainrust::prelude::*;
//!
//! let builder = AgentBuilder::new()
//!     .system("You are a concise assistant.")
//!     .tool(Calculator::new());
//! let _ = builder;
//! ```
//!
//! The five modules covered here are `agents`, `retrieval` (RAG), `sessions`,
//! `evaluation`, and `tools`. Everything else stays reachable through its
//! module path (`langchainrust::mcp`, `langchainrust::a2a`, ...); those
//! extended surfaces are feature-gated and NOT part of the main path.

pub use crate::agents::{AgentBuilder, AgentExecutor, BaseAgent, FunctionCallingAgent};
pub use crate::core::BaseChatModel;
pub use crate::evaluation::{EvalRunner, ExactMatch, StringDistance};
pub use crate::language_models::{OpenAIChat, OpenAIConfig};
pub use crate::retrieval::{RAGPipeline, RAGPipelineBuilder};
pub use crate::schema::Message;
pub use crate::sessions::{EventSessionManager, FileCheckpoint, MemoryEventStore};
pub use crate::tools::Calculator;
pub use crate::vector_stores::InMemoryVectorStore;
