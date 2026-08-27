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
use crate::orchestration::{Orchestrator, RunContext};
use crate::streaming::AgentStreamEvent;

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

/// Adapter that wraps an `AgentExecutor` as `Runnable<String, AgentStreamEvent>`.
///
/// Unlike [`AgentRunnable`], `stream()` preserves **all** `AgentStreamEvent`
/// variants (`Text` / `ToolStart` / `ToolEnd` / `FinalAnswer` / `Error`)
/// instead of filtering down to `FinalAnswer`. Use this in LCEL pipelines when
/// you need the fused tool-event + text-token stream (P1-8).
///
/// Non-streaming `invoke()` runs the agent and returns the final answer as a
/// single `AgentStreamEvent::FinalAnswer`.
pub struct AgentEventRunnable {
    executor: Arc<AgentExecutor>,
}

impl AgentEventRunnable {
    /// Wrap an executor, exposing its full event stream.
    pub fn new(executor: Arc<AgentExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl Runnable<String, AgentStreamEvent> for AgentEventRunnable {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        config: Option<RunnableConfig>,
    ) -> Result<AgentStreamEvent, LcelError> {
        let output = self
            .executor
            .invoke_with_config(input, config)
            .await
            .map_err(|e| LcelError::Agent(e.to_string()))?;
        Ok(AgentStreamEvent::FinalAnswer { content: output })
    }

    async fn stream(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, LcelError>> + Send>>, LcelError>
    {
        use futures_util::StreamExt;

        let event_stream = self.executor.stream(input);
        // Preserve every event; only map the error type.
        let mapped = event_stream
            .map(|event_result| event_result.map_err(|e| LcelError::Agent(e.to_string())));
        Ok(Box::pin(mapped))
    }
}

/// Adapter that wraps an [`Orchestrator`] (P1-1) as a `Runnable`.
///
/// Lets high-level orchestrators (PlanExecute / AdaptiveRAG / CorrectiveRAG /
/// DeepResearch) participate in LCEL pipelines. `config.metadata["trace_id"]`
/// flows through to [`RunContext`].
pub struct OrchestratorRunnable<O: Orchestrator> {
    orchestrator: O,
}

impl<O: Orchestrator> OrchestratorRunnable<O> {
    /// Wrap an orchestrator.
    pub fn new(orchestrator: O) -> Self {
        Self { orchestrator }
    }
}

#[async_trait]
impl<O> Runnable<O::Input, O::Output> for OrchestratorRunnable<O>
where
    O: Orchestrator,
    O::Input: Send + Sync + 'static,
    O::Output: Send + Sync + 'static,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: O::Input,
        config: Option<RunnableConfig>,
    ) -> Result<O::Output, LcelError> {
        let ctx = match &config {
            Some(cfg) => RunContext::from_config(cfg),
            None => RunContext::new_random(),
        };
        self.orchestrator
            .run_with_context(input, &ctx)
            .await
            .map_err(|e| LcelError::Agent(e.to_string()))
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
    use crate::streaming::AgentStreamEvent;
    use std::collections::HashMap;

    #[test]
    fn agent_error_into_lcel_error() {
        let agent_err = crate::base::AgentError::MaxIterationsReached;
        let lcel_err: LcelError = agent_err.into();
        assert!(matches!(lcel_err, LcelError::Agent(_)));
        assert!(lcel_err.to_string().contains("Max iterations"));
    }

    /// Mock agent that always finishes immediately.
    struct TestFinishAgent;

    #[async_trait]
    impl crate::BaseAgent for TestFinishAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[crate::types::AgentStep],
            _inputs: &HashMap<String, String>,
        ) -> Result<crate::types::AgentOutput, crate::base::AgentError> {
            Ok(crate::types::AgentOutput::Finish(
                crate::types::AgentFinish::new("answer".to_string(), String::new()),
            ))
        }
    }

    /// P1-8: AgentEventRunnable::stream preserves all events (Text + FinalAnswer),
    /// instead of filter_map'ing to a single string like AgentRunnable.
    #[tokio::test]
    async fn agent_event_runnable_preserves_all_events() {
        use futures_util::StreamExt;

        let executor = Arc::new(crate::base::AgentExecutor::new(
            Arc::new(TestFinishAgent),
            vec![],
        ));
        let runnable = AgentEventRunnable::new(executor);

        let mut stream = runnable.stream("hi".to_string(), None).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], AgentStreamEvent::Text { .. }));
        assert!(matches!(events[1], AgentStreamEvent::FinalAnswer { .. }));
    }

    /// P1-8: non-streaming invoke returns a single FinalAnswer event.
    #[tokio::test]
    async fn agent_event_runnable_invoke_returns_final_answer() {
        let executor = Arc::new(crate::base::AgentExecutor::new(
            Arc::new(TestFinishAgent),
            vec![],
        ));
        let runnable = AgentEventRunnable::new(executor);

        let event = runnable.invoke("hi".to_string(), None).await.unwrap();
        match event {
            AgentStreamEvent::FinalAnswer { content } => assert_eq!(content, "answer"),
            other => panic!("expected FinalAnswer, got {:?}", other),
        }
    }
}
