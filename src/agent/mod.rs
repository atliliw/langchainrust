pub mod agent;
mod executor;
use crate::tools::Tool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use crate::llms::LLM;
use crate::memory::Memory;
use crate::prompts::ChatPromptTemplate;

/// Core abstraction for an agent that can decide the next action to take.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Decide the next step given the user input and optional intermediate scratchpad.
    async fn get_next_step(
        &self,
        input: &str,
        intermediate_steps: Option<&str>,
    ) -> Result<AgentAction, AgentError>;
    /// Variant of `get_next_step` that also receives runtime template variables.
    async fn get_next_step_with_vars(
        &self,
        input: &str,
        intermediate_steps: Option<&str>,
        _vars: &HashMap<String, String>,
    ) -> Result<AgentAction, AgentError> {
        self.get_next_step(input, intermediate_steps).await
    }
    /// Optionally record input/output pairs into the agent's memory implementation.
    fn add_memory(&self, _input: &str, _output: &str) {}
}

/// Executes an `Agent` in a loop, wiring it with available tools.
pub struct AgentExecutor {
    agent: Box<dyn Agent>,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: usize,
}

#[derive(Debug)]
pub enum AgentAction {
    /// Instruction to call a tool with provided name and parameters.
    ToolCall(String, HashMap<String, String>),
    /// Final answer to be returned to the user.
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


/// A ReAct-style agent that uses tools, optional memory, and chat prompts.
pub struct ReActAgent {
    llm: LLM,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Mutex<Box<dyn Memory>>>,
    user_template: Option<ChatPromptTemplate>,
    verbose: bool,
}
