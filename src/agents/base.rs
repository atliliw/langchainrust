// src/agents/base.rs
//! Agent base traits and executor implementation.

use super::types::{AgentAction, AgentFinish, AgentOutput, AgentStep};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use crate::core::tools::BaseTool;
use crate::memory::BaseMemory;
use crate::callbacks::{CallbackManager, RunTree, RunType};

/// Agent error types.
#[derive(Debug)]
pub enum AgentError {
    /// Output parsing error.
    OutputParsingError(String),
    
    /// Tool not found.
    ToolNotFound(String),
    
    /// Tool execution error.
    ToolExecutionError(String),
    
    /// Max iterations reached.
    MaxIterationsReached,
    
    /// Other error.
    Other(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::OutputParsingError(msg) => write!(f, "Output parsing error: {}", msg),
            AgentError::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            AgentError::ToolExecutionError(msg) => write!(f, "Tool execution error: {}", msg),
            AgentError::MaxIterationsReached => write!(f, "Max iterations reached"),
            AgentError::Other(msg) => write!(f, "Agent error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}

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
    fn return_stopped_response(
        &self,
        _intermediate_steps: &[AgentStep],
    ) -> AgentFinish {
        AgentFinish::new(
            "Agent stopped due to iteration limit or time limit.".to_string(),
            String::new(),
        )
    }
}

/// Agent executor.
///
/// Responsible for executing the agent's decision loop: Plan → Act → Observe.
pub struct AgentExecutor {
    /// Agent instance.
    agent: Arc<dyn BaseAgent>,
    
    /// Available tools.
    tools: Vec<Arc<dyn BaseTool>>,
    
    /// Max iterations.
    max_iterations: usize,
    
    /// Verbose output.
    verbose: bool,
    
    /// Memory (optional).
    memory: Option<Arc<tokio::sync::Mutex<dyn BaseMemory>>>,
    
    /// Callback manager (optional).
    callbacks: Option<Arc<CallbackManager>>,
}

impl AgentExecutor {
    /// Creates a new AgentExecutor.
    pub fn new(agent: Arc<dyn BaseAgent>, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        Self {
            agent,
            tools,
            max_iterations: 10,
            verbose: false,
            memory: None,
            callbacks: None,
        }
    }
    
    /// Sets max iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    
    /// Sets verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    
    /// Sets memory.
    pub fn with_memory(mut self, memory: Arc<tokio::sync::Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }
    
    /// Sets callback manager.
    pub fn with_callbacks(mut self, callbacks: Arc<CallbackManager>) -> Self {
        self.callbacks = Some(callbacks);
        self
    }
    
    /// Executes the agent.
    pub async fn invoke(&self, input: String) -> Result<String, AgentError> {
        let mut root_run = RunTree::new(
            "AgentExecutor",
            RunType::Chain,
            json!({"input": input.clone()}),
        );
        
        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler.on_chain_start(&root_run, &root_run.inputs).await;
            }
        }
        
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input.clone());
        
        if let Some(memory) = &self.memory {
            let memory_vars = memory.lock().await
                .load_memory_variables(&inputs).await
                .map_err(|e| AgentError::Other(format!("Failed to load memory: {}", e)))?;
            
            if let Some(history) = memory_vars.get("history") {
                if let Some(history_str) = history.as_str() {
                    inputs.insert("history".to_string(), history_str.to_string());
                }
            }
        }
        
        let intermediate_steps: Vec<AgentStep> = Vec::new();
        
        let result = self.run_agent_loop(inputs.clone(), intermediate_steps, &mut root_run).await;
        
        if let Some(memory) = &self.memory {
            if let Ok(ref output) = result {
                let mut outputs = HashMap::new();
                outputs.insert("output".to_string(), output.clone());
                
                memory.lock().await
                    .save_context(&inputs, &outputs).await
                    .map_err(|e| AgentError::Other(format!("Failed to save memory: {}", e)))?;
            }
        }
        
        match &result {
            Ok(output) => {
                root_run.end(json!({"output": output}));
                if let Some(ref callbacks) = self.callbacks {
                    if let Some(ref outputs) = root_run.outputs {
                        for handler in callbacks.handlers() {
                            handler.on_chain_end(&root_run, outputs).await;
                        }
                    }
                }
            }
            Err(e) => {
                root_run.end_with_error(e.to_string());
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_chain_error(&root_run, &e.to_string()).await;
                    }
                }
            }
        }
        
        result
    }
    
    /// Runs the agent loop.
    async fn run_agent_loop(
        &self,
        inputs: HashMap<String, String>,
        mut intermediate_steps: Vec<AgentStep>,
        root_run: &mut RunTree,
    ) -> Result<String, AgentError> {
        for iteration in 0..self.max_iterations {
            if self.verbose {
                println!("\n=== Iteration {} ===", iteration + 1);
            }
            
            let output = self.agent.plan(&intermediate_steps, &inputs).await?;
            
            match output {
                AgentOutput::Finish(finish) => {
                    if self.verbose {
                        println!("Final answer: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }
                
                AgentOutput::Action(action) => {
                    if self.verbose {
                        println!("Action: {}({})", action.tool, action.tool_input);
                    }
                    
                    let observation = self.execute_tool(&action, root_run).await?;
                    
                    if self.verbose {
                        println!("Observation: {}", observation);
                    }
                    
                    intermediate_steps.push(AgentStep::new(action, observation));
                }
                
                AgentOutput::Actions(actions) => {
                    if self.verbose {
                        println!("Parallel actions: {} count", actions.len());
                        for action in &actions {
                            println!("  - {}({})", action.tool, action.tool_input);
                        }
                    }
                    
                    let observations = self.execute_tools_parallel(&actions, root_run).await?;
                    
                    if self.verbose {
                        for (i, obs) in observations.iter().enumerate() {
                            println!("Observation {}: {}", i + 1, obs);
                        }
                    }
                    
                    for (action, observation) in actions.into_iter().zip(observations.into_iter()) {
                        intermediate_steps.push(AgentStep::new(action, observation));
                    }
                }
            }
        }
        
        if self.verbose {
            println!("Max iterations reached: {}", self.max_iterations);
        }
        
        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }
    
    /// Executes multiple tools in parallel.
    async fn execute_tools_parallel(
        &self,
        actions: &[super::types::AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        use futures_util::future::join_all;
        
        let futures: Vec<_> = actions.iter().map(|action| {
            self.execute_tool(action, root_run)
        }).collect();
        
        join_all(futures).await.into_iter().collect()
    }
    
    /// Executes a single tool.
    async fn execute_tool(&self, action: &AgentAction, root_run: &RunTree) -> Result<String, AgentError> {
        let tool = self.tools.iter()
            .find(|t| t.name() == action.tool)
            .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;
        
        let input_str = match &action.tool_input {
            super::types::ToolInput::String(s) => s.clone(),
            super::types::ToolInput::Object(v) => serde_json::to_string(v)
                .unwrap_or_else(|_| v.to_string()),
        };
        
        let mut tool_run = root_run.create_child(
            &action.tool,
            RunType::Tool,
            json!({"input": input_str.clone()}),
        );
        
        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler.on_tool_start(&tool_run, &action.tool, &input_str).await;
            }
        }
        
        let result = tool.run(input_str.clone()).await;
        
        match result {
            Ok(output) => {
                tool_run.end(json!({"output": output.clone()}));
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_end(&tool_run, &output).await;
                    }
                }
                Ok(output)
            }
            Err(e) => {
                tool_run.end_with_error(e.to_string());
                if let Some(ref callbacks) = self.callbacks {
                    for handler in callbacks.handlers() {
                        handler.on_tool_error(&tool_run, &e.to_string()).await;
                    }
                }
                Err(AgentError::ToolExecutionError(e.to_string()))
            }
        }
    }
}

impl std::fmt::Debug for AgentExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentExecutor")
            .field("max_iterations", &self.max_iterations)
            .field("verbose", &self.verbose)
            .field("tools_count", &self.tools.len())
            .field("has_memory", &self.memory.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ConversationBufferMemory;
    
    /// Tests AgentExecutor with memory.
    #[tokio::test]
    async fn test_agent_executor_with_memory() {
        // Create simple mock agent
        struct TestAgent;
        
        #[async_trait]
        impl BaseAgent for TestAgent {
            async fn plan(
                &self,
                _intermediate_steps: &[AgentStep],
                inputs: &HashMap<String, String>,
            ) -> Result<AgentOutput, AgentError> {
                // If history exists, check if it contains previous info
                if let Some(history) = inputs.get("history") {
                    if history.contains("Zhang San") {
                        return Ok(AgentOutput::Finish(AgentFinish::new(
                            "Your name is Zhang San".to_string(),
                            String::new(),
                        )));
                    }
                }
                
                // Otherwise return input content
                let input = inputs.get("input").unwrap();
                Ok(AgentOutput::Finish(AgentFinish::new(
                    format!("Received: {}", input),
                    String::new(),
                )))
            }
        }
        
        // Create memory
        let memory = Arc::new(tokio::sync::Mutex::new(ConversationBufferMemory::new()));
        
        // Create executor
        let executor = AgentExecutor::new(Arc::new(TestAgent), vec![])
            .with_memory(memory);
        
        // First conversation round
        let result1 = executor.invoke("My name is Zhang San".to_string()).await.unwrap();
        println!("Round 1: {}", result1);
        
        // Second conversation round - should remember the name
        let result2 = executor.invoke("What is my name?".to_string()).await.unwrap();
        println!("Round 2: {}", result2);
        
        assert!(result2.contains("Zhang San"));
    }
}