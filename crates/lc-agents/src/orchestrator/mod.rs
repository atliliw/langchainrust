//! Common trait for high-level orchestrators (P1-1)
//!
//! `PlanExecuteAgent` / `DeepResearchAgent` / `CorrectiveRAGAgent` / `AdaptiveRAG`
//! each used to define its own `run()`, with incompatible signatures that could
//! not be composed or plugged into LCEL. This module unifies them:
//!
//! - [`Orchestrator`] defines `run_with_context(input, ctx)`, with errors
//!   unified to [`AgentError`].
//! - [`RunContext`] carries `trace_id` (P1-4 observability) and a cross-step
//!   shared workspace.
//! - [`crate::adapter::OrchestratorRunnable`] lets orchestrators enter LCEL pipelines.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_agents::orchestration::{Orchestrator, RunContext};
//!
//! let plan_agent = PlanExecuteAgent::new(llm, tools);
//! let ctx = RunContext::new("trace-abc");
//! let output = plan_agent.run_with_context("目标".to_string(), &ctx).await?;
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lc_core::runnables::RunnableConfig;
use serde_json::Value;

use crate::AgentError;

mod fan_out_fan_in;
mod impls;
mod review;
mod sequential;
mod task_adapter;
#[cfg(test)]
mod tests;

pub use fan_out_fan_in::FanOutFanIn;
pub use review::{parse_review_verdict, review_envelope, ReviewOrchestrator, ReviewVerdict};
pub use sequential::SequentialPipeline;
pub use task_adapter::{task_adapter, TaskAdapter};

/// Common trait for high-level orchestrators.
///
/// Associated types express each orchestrator's different input/output
/// (PlanExecute→String, AdaptiveRAG→AdaptiveRAGResult, etc.); `run_with_context`
/// unifies the signature + [`AgentError`], so orchestrators are composable and
/// can enter LCEL.
#[async_trait]
pub trait Orchestrator: Send + Sync {
    /// Input type (usually a `String` objective/question).
    type Input;
    /// Output type.
    type Output;

    /// Execution entry point carrying the run context.
    async fn run_with_context(
        &self,
        input: Self::Input,
        ctx: &RunContext,
    ) -> Result<Self::Output, AgentError>;
}

/// Orchestrator run context.
///
/// `trace_id` propagates across multi-agent / cross-step call chains (P1-4);
/// `shared_state` provides a JSON workspace shared across steps.
#[derive(Debug, Clone)]
pub struct RunContext {
    /// Trace ID: shared across the whole call chain, for log/audit/metric correlation.
    pub trace_id: String,
    /// Workspace shared across steps (optional).
    pub shared_state: Option<Arc<Mutex<Value>>>,
}

/// Generates a lightweight trace_id (hex timestamp).
pub fn generate_trace_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("trace-{:x}", nanos)
}

impl RunContext {
    /// Creates a context with the given `trace_id`.
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            shared_state: None,
        }
    }

    /// Creates a context, auto-generating `trace_id`.
    pub fn new_random() -> Self {
        Self::new(generate_trace_id())
    }

    /// Carries the shared workspace.
    pub fn with_shared_state(mut self, shared_state: Arc<Mutex<Value>>) -> Self {
        self.shared_state = Some(shared_state);
        self
    }

    /// Extracts `trace_id` from the LCEL [`RunnableConfig`] (reads
    /// `metadata["trace_id"]`), generating one if missing. Used to thread the
    /// LCEL pipeline's trace through to the orchestrator.
    pub fn from_config(config: &RunnableConfig) -> Self {
        let trace_id = config
            .metadata
            .get("trace_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(generate_trace_id);
        Self::new(trace_id)
    }
}
