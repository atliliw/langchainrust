// lc-agents/src/adapter.rs
//! AgentRunnable adapter - bridges AgentExecutor to the Runnable trait.
//!
//! This allows agents to participate in LCEL pipelines via `pipe()`.

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use std::pin::Pin;
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
        config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        // Merge config callbacks with executor's own callbacks
        self.executor
            .invoke_with_config(input, config)
            .await
            .map_err(|e| LcelError::Agent(e.to_string()))
    }

    /// Override stream() to delegate to AgentExecutor::stream(),
    /// enabling real streaming in LCEL pipelines.
    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LcelError>> + Send>>, LcelError> {
        use futures_util::StreamExt;

        let event_stream = self.executor.stream(input);

        // Map AgentStreamEvent to String, extracting FinalAnswer content
        let mapped = event_stream.filter_map(|event_result| async move {
            match event_result {
                Ok(event) => match event {
                    crate::streaming::AgentStreamEvent::FinalAnswer { content } => {
                        Some(Ok(content))
                    }
                    crate::streaming::AgentStreamEvent::Error { message } => {
                        Some(Err(LcelError::Agent(message)))
                    }
                    // Skip other event types (ToolStart, ToolEnd, PipelineStep, etc.)
                    _ => None,
                },
                Err(e) => Some(Err(LcelError::Agent(e.to_string()))),
            }
        });

        Ok(Box::pin(mapped))
    }
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
