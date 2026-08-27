// lc-agents/src/executor/mod.rs
//! Agent base traits and executor implementation.

use crate::types::{AgentFinish, AgentOutput, AgentStep};
use async_trait::async_trait;
use lc_core::language_models::TokenUsage;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;

/// Cache-namespace counter for each `AgentExecutor` instance.
///
/// P2-1: the same `(inputs, intermediate_steps)` may be planned into different actions
/// by different Agents across executors, so a shared cache would cross-contaminate
/// results. Giving each instance a unique namespace isolates cache keys by construction,
/// without hurting deterministic hits across multiple `invoke`s on the same instance.
static CACHE_NS: AtomicUsize = AtomicUsize::new(0);

/// Agent error types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AgentError {
    /// Output parsing error.
    #[error("Output parsing error: {0}")]
    OutputParsingError(String),

    /// Tool not found.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Tool execution error.
    #[error("Tool execution error: {0}")]
    ToolExecutionError(String),

    /// Max iterations reached.
    #[error("Max iterations reached")]
    MaxIterationsReached,

    /// Budget exceeded (approval/budget gate §4.2). Callers catch it to distinguish a
    /// "budget stop" from "the model did not converge".
    #[error("Budget exceeded: {0:?}")]
    BudgetExceeded(BudgetExceeded),

    /// Cross-process resume (§4.2): checkpoint read / write / restore failed.
    #[error("Resume error: {0}")]
    Resume(String),

    /// Other error.
    #[error("Agent error: {0}")]
    Other(String),
}

/// Base Agent trait.
///
/// Defines the core interface for agents. Agent is responsible for planning,
/// not execution. Execution is handled by AgentExecutor.
#[async_trait]
pub trait BaseAgent: Send + Sync {
    /// Plans the next action.
    ///
    /// # Arguments
    /// * `intermediate_steps` - History of executed steps.
    /// * `inputs` - User input.
    ///
    /// # Returns
    /// * `AgentOutput::Action` - Action to execute.
    /// * `AgentOutput::Finish` - Final answer.
    async fn plan(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError>;

    /// Plans the next action, streaming any model text through `on_token` as it
    /// is produced.
    ///
    /// # Arguments
    /// * `intermediate_steps` - History of executed steps.
    /// * `inputs` - User input.
    /// * `on_token` - Called with each chunk of model text as it becomes
    ///   available, taking **ownership** of the chunk (so the returned future
    ///   never borrows the token and stays `'static`). Streaming-capable agents
    ///   (e.g. ReAct, function-calling) emit free text per token; steps that
    ///   produce no free text (e.g. a function-calling step invoking a tool)
    ///   emit nothing. Non-streaming agents deliver the whole answer in one
    ///   call.
    ///
    /// # Returns
    /// * `AgentOutput::Action` - Action to execute.
    /// * `AgentOutput::Finish` - Final answer.
    ///
    /// The default implementation delegates to [`BaseAgent::plan`] and forwards
    /// the whole final-answer text as a single chunk — behaviorally identical
    /// to calling `plan()` directly. Streaming-capable agents (e.g. ReAct)
    /// override this to emit per-token chunks from the model's streaming chat
    /// API; callers must still call [`BaseAgent::plan`] for the non-streaming
    /// path (e.g. `invoke`) so that path is unaffected.
    async fn plan_stream(
        &self,
        intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
        on_token: &mut (dyn FnMut(String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send),
    ) -> Result<AgentOutput, AgentError> {
        let output = self.plan(intermediate_steps, inputs).await?;
        if let AgentOutput::Finish(finish) = &output {
            on_token(finish.output().unwrap_or("").to_string()).await;
        }
        Ok(output)
    }

    /// Returns input keys.
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }

    /// Returns allowed tools list.
    fn get_allowed_tools(&self) -> Option<Vec<&str>> {
        None
    }

    /// Returns stopped response when max iterations reached.
    fn return_stopped_response(&self, _intermediate_steps: &[AgentStep]) -> AgentFinish {
        AgentFinish::new(
            "Agent stopped due to iteration limit or time limit.".to_string(),
            String::new(),
        )
    }

    /// Returns the token usage from the most recent `plan()` call, if available.
    ///
    /// Agents that make LLM calls inside `plan()` may override this to report
    /// cost metrics to `AgentExecutor` (P1-5). Defaults to `None`.
    fn last_token_usage(&self) -> Option<TokenUsage> {
        None
    }
}

/// Minimum allowed `max_iterations`.
const MIN_MAX_ITERATIONS: usize = 1;

/// Upper bound for `max_iterations` — guards against runaway loops.
const MAX_MAX_ITERATIONS: usize = 100;

/// Default number of tools executed concurrently.
const DEFAULT_MAX_CONCURRENCY: usize = 8;

mod agent_loop;
mod budget;
mod engine;
mod hooks;
#[cfg(test)]
mod tests;
mod tools;

pub use budget::{BudgetConfig, BudgetExceeded};
pub use engine::AgentExecutor;
