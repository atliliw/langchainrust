#[allow(clippy::module_inception)]
pub mod agent;
mod executor;
pub mod model_router;
use crate::tools::Tool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::llms::LLM;
use crate::memory::Memory;
use crate::prompts::ChatPromptTemplate;
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

pub struct AgentExecutor {
    agent: Box<dyn Agent>,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: usize,
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

pub struct ReActAgent {
    llm: LLM,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Mutex<Box<dyn Memory>>>,
    user_template: Option<ChatPromptTemplate>,
}
