// lc-agents/src/executor/mod.rs
//! Agent base traits and executor implementation.

use crate::types::{AgentFinish, AgentOutput, AgentStep};
use async_trait::async_trait;
use lc_core::language_models::TokenUsage;
use std::collections::HashMap;
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
mod engine;
mod hooks;
#[cfg(test)]
mod tests;
mod tools;

pub use engine::AgentExecutor;
