//! P1-8: adapt a stateful [`AgentExecutor`] to the stateless [`BaseChain`] facade.
//!
//! A2A models "one task = one conversation", which needs multi-turn state. A
//! `BaseChain` is stateless: every `invoke` is a fresh shot. An
//! [`AgentExecutor`] — particularly one built with `.with_memory(...)` so that
//! conversation history accumulates across turns — is the stateful counterpart.
//!
//! [`AgentExecutorChain`] bridges the two so an `A2AServer` can be backed
//! directly by an agent via [`A2AServer::from_agent`], giving each A2A task
//! genuine conversational continuity instead of a series of independent chain
//! invocations.

use std::collections::HashMap;
use std::sync::Arc;

use lc_agents::AgentExecutor;
use lc_chains::base::{BaseChain, ChainError, ChainResult};
use serde_json::Value;

/// Wraps an [`AgentExecutor`] behind the [`BaseChain`] trait.
///
/// Inputs follow the agent convention: a single `input` string (or any string
/// key the agent's planner reads). Output is produced under `output`.
pub struct AgentExecutorChain {
    executor: Arc<AgentExecutor>,
}

impl AgentExecutorChain {
    /// Create an adapter around a ready-built executor.
    ///
    /// Attach memory *before* wrapping (e.g. `.with_memory(...)`) if multi-turn
    /// state across A2A tasks is desired.
    pub fn new(executor: Arc<AgentExecutor>) -> Self {
        Self { executor }
    }

    /// The inner executor, for inspection or configuration.
    pub fn inner(&self) -> &AgentExecutor {
        &self.executor
    }
}

#[async_trait::async_trait]
impl BaseChain for AgentExecutorChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let raw = inputs
            .get("input")
            .ok_or_else(|| ChainError::MissingInput("input".to_string()))?;
        let input = raw
            .as_str()
            .ok_or_else(|| ChainError::InputError("input must be a string".to_string()))?
            .to_string();

        let output =
            self.executor.invoke(input).await.map_err(|e| {
                ChainError::ExecutionError(format!("Agent execution failed: {}", e))
            })?;

        let mut result = HashMap::new();
        result.insert("output".to_string(), Value::String(output));
        Ok(result)
    }

    fn name(&self) -> &str {
        "agent-executor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_agents::{AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent};
    use serde_json::json;

    /// A planner that echoes its `input` back verbatim.
    struct EchoAgent;

    #[async_trait::async_trait]
    impl BaseAgent for EchoAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[AgentStep],
            inputs: &HashMap<String, String>,
        ) -> Result<AgentOutput, AgentError> {
            let input = inputs.get("input").cloned().unwrap_or_default();
            Ok(AgentOutput::Finish(AgentFinish::new(
                format!("echo: {}", input),
                String::new(),
            )))
        }
    }

    /// A planner that always fails, so agent errors can be observed.
    struct FailAgent;

    #[async_trait::async_trait]
    impl BaseAgent for FailAgent {
        async fn plan(
            &self,
            _intermediate_steps: &[AgentStep],
            _inputs: &HashMap<String, String>,
        ) -> Result<AgentOutput, AgentError> {
            Err(AgentError::Other("boom".to_string()))
        }
    }

    fn echo_chain() -> AgentExecutorChain {
        let executor = AgentExecutor::new(Arc::new(EchoAgent), Vec::new());
        AgentExecutorChain::new(Arc::new(executor))
    }

    #[tokio::test]
    async fn invokes_agent_and_returns_output() {
        let chain = echo_chain();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), json!("hello"));
        let result = chain.invoke(inputs).await.unwrap();
        assert_eq!(result.get("output"), Some(&json!("echo: hello")));
    }

    #[tokio::test]
    async fn missing_input_returns_missing_input_error() {
        let chain = echo_chain();
        let result = chain.invoke(HashMap::new()).await;
        assert!(matches!(result, Err(ChainError::MissingInput(_))));
    }

    #[tokio::test]
    async fn non_string_input_returns_input_error() {
        let chain = echo_chain();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), json!(42));
        let result = chain.invoke(inputs).await;
        assert!(matches!(result, Err(ChainError::InputError(_))));
    }

    #[tokio::test]
    async fn agent_error_maps_to_execution_error() {
        let executor = AgentExecutor::new(Arc::new(FailAgent), Vec::new());
        let chain = AgentExecutorChain::new(Arc::new(executor));
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), json!("hi"));
        let result = chain.invoke(inputs).await;
        assert!(matches!(result, Err(ChainError::ExecutionError(_))));
    }

    #[test]
    fn exposes_expected_keys_and_name() {
        let chain = echo_chain();
        assert_eq!(chain.input_keys(), vec!["input"]);
        assert_eq!(chain.output_keys(), vec!["output"]);
        assert_eq!(chain.name(), "agent-executor");
    }
}
