// lc-agents/src/lib.rs
//! Agent system for building autonomous LLM applications.
//!
//! Provides core abstractions and implementations for agents.
//!
//! # Core Concepts
//!
//! - **Agent**: Responsible for planning, deciding what action to execute next.
//! - **AgentExecutor**: Responsible for execution loop (plan -> act -> observe).
//! - **Tool**: Callable tools that agents can invoke.
//!
//! # Execution Flow
//!
//! ```text
//! Input question
//!     |
//! Agent.plan() -> AgentAction or AgentFinish
//!     |
//! If Action: execute tool -> get observation
//!     |
//! Add to intermediate_steps
//!     |
//! Loop until AgentFinish returned
//! ```

pub mod adapter;
pub mod adaptive_rag;
pub mod base;
pub mod builder;
pub mod crag;
pub mod deep_research;
pub mod function_calling;
pub mod handoffs;
pub mod hooks;
pub mod plan_execute;
pub mod react;
pub mod streaming;
pub mod types;

pub use adapter::AgentRunnable;
pub use adaptive_rag::{AdaptiveRAG, AdaptiveRAGError, AdaptiveRAGResult, RagDecision};
pub use base::{AgentError, AgentExecutor, BaseAgent};
pub use builder::AgentBuilder;
pub use crag::{CRAGError, CRAGResult, CorrectiveRAGAgent};
pub use deep_research::{Citation, DeepResearchAgent, ResearchError, ResearchReport};
pub use function_calling::FunctionCallingAgent;
pub use handoffs::HandoffManager;
pub use hooks::{
    AgentHook, ApprovalHook, CompletionAction, CompletionContext, CompletionResult,
    ContentFilterHook, ErrorAction, HookError, LoggingHook, StreamAction, ToolCallAction,
    ToolCallContext, ToolResultContext,
};
pub use plan_execute::{PlanExecuteAgent, PlanExecuteError};
pub use react::ReActAgent;
pub use streaming::StreamingFunctionCallingAgent;
pub use types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
