// lc-agents/src/executor/mod.rs
//! Agent base traits and executor implementation.

use crate::types::{AgentFinish, AgentOutput, AgentStep};
use async_trait::async_trait;
use lc_core::language_models::TokenUsage;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;

/// 每个 `AgentExecutor` 实例的缓存命名空间计数器。
///
/// P2-1: 相同 `(inputs, intermediate_steps)` 在不同 executor 里可能被不同
/// Agent 规划出不同动作,共享缓存会串结果。给每个实例一个唯一命名空间,
/// 让缓存 key 天然隔离,又不妨碍同一实例多次 invoke 间的确定性命中。
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

    /// 预算超限(人审/预算门 §4.2)。调用方捕获后区分"预算截停"与"模型未收敛"。
    #[error("Budget exceeded: {0:?}")]
    BudgetExceeded(BudgetExceeded),

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
    ///   never borrows the token and stays `'static`). May be the whole answer
    ///   in one call for non-streaming agents, or empty for agents that never
    ///   emit free text (e.g. function-calling).
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
