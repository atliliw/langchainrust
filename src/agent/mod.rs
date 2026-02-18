#[allow(clippy::module_inception)]
pub mod agent;
pub mod executor;

pub use agent::ReActAgent;
pub use executor::{AgentExecutor, ExecutionResult};

use std::collections::HashMap;

use async_trait::async_trait;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn get_next_step(
        &self,
        input: &str,
        intermediate_steps: Option<&str>,
    ) -> Result<AgentAction, AgentError>;
    async fn get_next_step_with_vars(
        &self,
        input: &str,
        intermediate_steps: Option<&str>,
        _vars: &HashMap<String, String>,
    ) -> Result<AgentAction, AgentError> {
        self.get_next_step(input, intermediate_steps).await
    }
    fn add_memory(&self, _input: &str, _output: &str) {}
}

#[derive(Debug)]
pub enum AgentAction {
    ToolCall(String, HashMap<String, String>),
    FinalAnswer(String),
}

#[derive(Debug)]
pub struct AgentError(pub String);

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AgentError {}
