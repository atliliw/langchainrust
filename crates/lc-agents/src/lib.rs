#![warn(missing_docs)]
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
/// 人审门(§4.2):工具执行前的异步审批闸。调用方实现 [`ApprovalHandler`] 并
/// 通过 [`AgentExecutor::with_approval`] 注入;默认关。
pub mod approval;
pub mod executor;
/// Module alias preserving the historical `lc_agents::base` path.
pub use executor as base;
pub mod builder;
pub mod cache;
pub mod crag;
pub mod deep_research;
/// Function calling based agent module.
pub mod function_calling;
pub mod handoffs;
pub mod hooks;
pub mod metrics;
pub mod orchestrator;
/// Module alias preserving the historical `lc_agents::orchestration` path.
pub use orchestrator as orchestration;
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
pub use approval::{AllowAll, ApprovalDecision, ApprovalHandler};
pub use builder::AgentBuilder;
pub use cache::{MemoryCache, ResponseCache};
pub use crag::{CRAGError, CRAGResult, CorrectiveRAGAgent};
pub use deep_research::{Citation, DeepResearchAgent, ResearchError, ResearchReport};
pub use executor::{AgentError, AgentExecutor, BaseAgent, BudgetConfig, BudgetExceeded};
pub use function_calling::FunctionCallingAgent;
pub use handoffs::HandoffManager;
pub use hooks::{
    AgentHook, ApprovalHook, CompletionAction, CompletionContext, CompletionResult,
    ContentFilterHook, ErrorAction, HookError, LoggingHook, PromptInjectionHook, StreamAction,
    TokenBudgetHook, ToolCallAction, ToolCallContext, ToolResultContext,
};
pub use metrics::AgentMetrics;
pub use orchestrator::{
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
