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
pub mod cache;
pub mod crag;
pub mod deep_research;
pub mod function_calling;
pub mod handoffs;
pub mod hooks;
pub mod metrics;
pub mod orchestration;
pub mod plan_execute;
pub mod policy;
pub mod react;
pub mod retry;
pub mod streaming;
mod structured;
pub mod task;
pub mod types;

pub use adapter::{AgentEventRunnable, AgentRunnable, OrchestratorRunnable};
pub use adaptive_rag::{AdaptiveRAG, AdaptiveRAGError, AdaptiveRAGResult, RagDecision};
pub use base::{AgentError, AgentExecutor, BaseAgent};
pub use builder::AgentBuilder;
pub use cache::{MemoryCache, ResponseCache};
pub use crag::{CRAGError, CRAGResult, CorrectiveRAGAgent};
pub use deep_research::{Citation, DeepResearchAgent, ResearchError, ResearchReport};
pub use function_calling::FunctionCallingAgent;
pub use handoffs::HandoffManager;
pub use hooks::{
    AgentHook, ApprovalHook, CompletionAction, CompletionContext, CompletionResult,
    ContentFilterHook, ErrorAction, HookError, LoggingHook, PromptInjectionHook, StreamAction,
    TokenBudgetHook, ToolCallAction, ToolCallContext, ToolResultContext,
};
pub use metrics::AgentMetrics;
pub use orchestration::{
    parse_review_verdict, review_envelope, task_adapter, FanOutFanIn, Orchestrator,
    ReviewOrchestrator, ReviewVerdict, RunContext, SequentialPipeline, TaskAdapter,
};
pub use plan_execute::{PlanExecuteAgent, PlanExecuteError};
pub use policy::{ToolPolicy, ToolRisk};
pub use react::ReActAgent;
pub use retry::RetryConfig;
pub use streaming::{AgentStreamEvent, StreamingFunctionCallingAgent, ToolCallState};
pub use task::AgentTask;
pub use types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
