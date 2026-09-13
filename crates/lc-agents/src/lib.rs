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
/// Approval gate (§4.2): asynchronous approval gate before tool execution.
/// Implement [`ApprovalHandler`] and inject it via [`AgentExecutor::with_approval`];
/// off by default.
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
/// B4 (v0.22.4): LLM-backed semantic-memory extractor turning a completed turn
/// into durable [`lc_memory::MemoryItem`]s.
pub mod memory_extractor;
pub mod metrics;
pub mod orchestrator;
/// Module alias preserving the historical `lc_agents::orchestration` path.
pub use orchestrator as orchestration;
pub mod plan_execute;
pub mod policy;
pub mod react;
/// Cross-process resume (§4.2 approval/budget gate): suspend-state persistence +
/// recovery. The framework writes/clears checkpoints ([`ResumeStore`]) around
/// approvals; a new process inspects via [`AgentExecutor::pending_approval`] and
/// resumes via [`AgentExecutor::resume`]. Off by default (no [`ResumeStore`] ⇒ no
/// serialization; existing behavior unchanged).
pub mod resume;
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
pub use executor::{
    estimate_step_tokens, AgentError, AgentExecutor, BaseAgent, BudgetConfig, BudgetExceeded,
    CompactionConfig, CompactionStrategy, CompactionTrigger,
};
pub use function_calling::FunctionCallingAgent;
pub use handoffs::HandoffManager;
pub use hooks::{
    AgentHook, ApprovalHook, CompletionAction, CompletionContext, CompletionResult,
    ContentFilterHook, ErrorAction, HookError, LoggingHook, PromptInjectionHook, StreamAction,
    TokenBudgetHook, ToolCallAction, ToolCallContext, ToolResultContext,
};
pub use memory_extractor::LlmMemoryExtractor;
pub use metrics::AgentMetrics;
pub use orchestrator::{
    parse_review_verdict, review_envelope, task_adapter, FanOutFanIn, Orchestrator,
    ReviewOrchestrator, ReviewVerdict, RunContext, SequentialPipeline, TaskAdapter,
};
pub use plan_execute::{PlanExecuteAgent, PlanExecuteError};
pub use policy::{ToolPolicy, ToolRisk};
pub use react::ReActAgent;
pub use resume::{FileResumeStore, MemoryResumeStore, PendingApproval, ResumeError, ResumeStore};
pub use retry::RetryConfig;
// B8 (v0.22.4): framework-neutral SSE framing is always exported; the axum
// serving pieces come in under the `sse-server` feature.
pub use streaming::{
    agent_sse_frames, encode_sse_frame, sse_event_name, sse_event_payload, AgentEventStream,
    AgentSseRequest, AgentStreamEvent, SseFrame, SseOptions, StreamingFunctionCallingAgent,
    ToolCallState,
};
#[cfg(feature = "sse-server")]
pub use streaming::{
    agent_sse_get_handler, agent_sse_handler, agent_sse_router, agent_sse_router_with,
    serve_agent_sse, serve_agent_sse_on, AgentSseQuery, AgentSseServerConfig, AgentSseState,
    AgentStreamFactory, AgentStreamFuture, DEFAULT_SSE_HEARTBEAT,
};
pub use task::AgentTask;
pub use types::{AgentAction, AgentFinish, AgentOutput, AgentStep, ToolInput};
