// lc-agents/src/adapter.rs
//! AgentRunnable adapter - bridges AgentExecutor to the Runnable trait.
//!
//! This allows agents to participate in LCEL pipelines via `pipe()`.

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use std::sync::Arc;

use crate::base::AgentExecutor;

/// Adapter that wraps an `AgentExecutor` as a `Runnable<String, String>`.
///
/// This enables agents to participate in LCEL pipelines:
///
/// ```rust,ignore
/// let agent_runnable = AgentRunnable::new(Arc::new(executor));
/// let pipeline = prompt.pipe(agent_runnable).pipe(parser);
/// ```
pub struct AgentRunnable {
    executor: Arc<AgentExecutor>,
}

impl AgentRunnable {
    /// Create a new adapter wrapping the given executor.
    pub fn new(executor: Arc<AgentExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Runnable<String, String> for AgentRunnable {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        self.executor.invoke(input).await.map_err(|e| LcelError::Agent(e.to_string()))
    }

    // stream, batch, transform use default implementations
    // (single-element stream, sequential batch, buffer-and-invoke)
}

/// Allow `AgentError` to convert into `LcelError`.
impl From<crate::base::AgentError> for LcelError {
    fn from(err: crate::base::AgentError) -> Self {
        LcelError::Agent(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_error_into_lcel_error() {
        let agent_err = crate::base::AgentError::MaxIterationsReached;
        let lcel_err: LcelError = agent_err.into();
        assert!(matches!(lcel_err, LcelError::Agent(_)));
        assert!(lcel_err.to_string().contains("Max iterations"));
    }
}
