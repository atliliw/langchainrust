// lc-agents/src/base.rs
//! Agent base traits and executor implementation.

use super::types::{AgentAction, AgentFinish, AgentOutput, AgentStep};
use super::streaming::state::AgentStreamEvent;
use async_trait::async_trait;
use lc_callbacks::{CallbackManager, RunTree, RunType};
use lc_core::tools::BaseTool;
use lc_memory::BaseMemory;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use futures_util::Stream;

/// Agent error types.
#[derive(Debug, thiserror::Error)]
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
}

/// Agent executor.
///
/// Responsible for executing the agent's decision loop: Plan -> Act -> Observe.
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
            let memory_vars = memory
                .lock()
                .await
                .load_memory_variables(&inputs)
                .await
                .map_err(|e| AgentError::Other(format!("Failed to load memory: {}", e)))?;

            if let Some(history) = memory_vars.get("history") {
                if let Some(history_str) = history.as_str() {
                    inputs.insert("history".to_string(), history_str.to_string());
                }
            }
        }

        let intermediate_steps: Vec<AgentStep> = Vec::new();

        let result = self
            .run_agent_loop(inputs.clone(), intermediate_steps, &mut root_run)
            .await;

        if let Some(memory) = &self.memory {
            if let Ok(ref output) = result {
                let mut outputs = HashMap::new();
                outputs.insert("output".to_string(), output.clone());

                memory
                    .lock()
                    .await
                    .save_context(&inputs, &outputs)
                    .await
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

    /// Stream agent execution as a true async stream of events.
    ///
    /// Each step of the agent loop (tool calls, observations, final answer)
    /// is emitted as an `AgentStreamEvent` as soon as it occurs.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut stream = executor.stream("What is Rust?".to_string());
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         Ok(AgentStreamEvent::ToolStart { name, input }) => { /* show tool call */ }
    ///         Ok(AgentStreamEvent::ToolEnd { name, output }) => { /* show result */ }
    ///         Ok(AgentStreamEvent::FinalAnswer { content }) => { /* show answer */ }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn stream(&self, input: String) -> Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, AgentError>> + Send>> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let agent = self.agent.clone();
        let tools = self.tools.clone();
        let max_iterations = self.max_iterations;
        let verbose = self.verbose;

        tokio::spawn(async move {
            let mut intermediate_steps: Vec<AgentStep> = Vec::new();
            let mut inputs = HashMap::new();
            inputs.insert("input".to_string(), input);

            for iteration in 0..max_iterations {
                if verbose {
                    log::info!("=== Stream Iteration {} ===", iteration + 1);
                }

                let output = match agent.plan(&intermediate_steps, &inputs).await {
                    Ok(o) => o,
                    Err(e) => {
                        let _ = tx.send(Ok(AgentStreamEvent::Error { message: e.to_string() })).await;
                        return;
                    }
                };

                match output {
                    AgentOutput::Finish(finish) => {
                        let content = finish.output().unwrap_or("").to_string();
                        let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
                        return;
                    }

                    AgentOutput::Action(action) => {
                        let tool_name = action.tool.clone();
                        let tool_input_str = match &action.tool_input {
                            super::types::ToolInput::String { value: s } => s.clone(),
                            super::types::ToolInput::Object { value: v } => {
                                serde_json::to_string(v).unwrap_or_default()
                            }
                        };

                        let _ = tx.send(Ok(AgentStreamEvent::ToolStart {
                            name: tool_name.clone(),
                            input: tool_input_str.clone(),
                        })).await;

                        // Execute the tool
                        let observation = match execute_tool_for_stream(&tools, &action).await {
                            Ok(obs) => obs,
                            Err(e) => {
                                let _ = tx.send(Ok(AgentStreamEvent::Error { message: e.to_string() })).await;
                                return;
                            }
                        };

                        let _ = tx.send(Ok(AgentStreamEvent::ToolEnd {
                            name: tool_name,
                            output: observation.clone(),
                        })).await;

                        intermediate_steps.push(AgentStep::new(action, observation));
                    }

                    AgentOutput::Actions(actions) => {
                        for action in &actions {
                            let tool_name = action.tool.clone();
                            let tool_input_str = match &action.tool_input {
                                super::types::ToolInput::String { value: s } => s.clone(),
                                super::types::ToolInput::Object { value: v } => {
                                    serde_json::to_string(v).unwrap_or_default()
                                }
                            };

                            let _ = tx.send(Ok(AgentStreamEvent::ToolStart {
                                name: tool_name.clone(),
                                input: tool_input_str,
                            })).await;
                        }

                        let observations = execute_tools_parallel_for_stream(&tools, &actions).await;

                        for (action, observation) in actions.into_iter().zip(observations.into_iter()) {
                            let _ = tx.send(Ok(AgentStreamEvent::ToolEnd {
                                name: action.tool.clone(),
                                output: observation.clone(),
                            })).await;

                            intermediate_steps.push(AgentStep::new(action, observation));
                        }
                    }
                }
            }

            // Max iterations reached
            let finish = agent.return_stopped_response(&intermediate_steps);
            let content = finish.output().unwrap_or("").to_string();
            let _ = tx.send(Ok(AgentStreamEvent::FinalAnswer { content })).await;
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
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
                log::info!("=== Iteration {} ===", iteration + 1);
            }

            let output = self.agent.plan(&intermediate_steps, &inputs).await?;

            match output {
                AgentOutput::Finish(finish) => {
                    if self.verbose {
                        log::info!("Final answer: {:?}", finish.return_values);
                    }
                    return Ok(finish.output().unwrap_or("").to_string());
                }

                AgentOutput::Action(action) => {
                    if self.verbose {
                        log::info!("Action: {}({})", action.tool, action.tool_input);
                    }

                    let observation = self.execute_tool(&action, root_run).await?;

                    if self.verbose {
                        log::info!("Observation: {}", observation);
                    }

                    intermediate_steps.push(AgentStep::new(action, observation));
                }

                AgentOutput::Actions(actions) => {
                    if self.verbose {
                        log::info!("Parallel actions: {} count", actions.len());
                        for action in &actions {
                            log::info!("  - {}({})", action.tool, action.tool_input);
                        }
                    }

                    let observations = self.execute_tools_parallel(&actions, root_run).await?;

                    if self.verbose {
                        for (i, obs) in observations.iter().enumerate() {
                            log::info!("Observation {}: {}", i + 1, obs);
                        }
                    }

                    for (action, observation) in actions.into_iter().zip(observations.into_iter()) {
                        intermediate_steps.push(AgentStep::new(action, observation));
                    }
                }
            }
        }

        if self.verbose {
            log::info!("Max iterations reached: {}", self.max_iterations);
        }

        let finish = self.agent.return_stopped_response(&intermediate_steps);
        Ok(finish.output().unwrap_or("").to_string())
    }

    /// Executes multiple tools in parallel.
    ///
    /// Collects successful results and reports failures as error observations
    /// rather than discarding partial results when one tool fails.
    async fn execute_tools_parallel(
        &self,
        actions: &[super::types::AgentAction],
        root_run: &RunTree,
    ) -> Result<Vec<String>, AgentError> {
        use futures_util::future::join_all;

        let futures: Vec<_> = actions
            .iter()
            .map(|action| self.execute_tool(action, root_run))
            .collect();

        let results = join_all(futures).await;
        let mut observations = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(output) => observations.push(output),
                Err(e) => observations.push(format!("[Tool execution error: {}]", e)),
            }
        }
        Ok(observations)
    }

    /// Executes a single tool.
    async fn execute_tool(
        &self,
        action: &AgentAction,
        root_run: &RunTree,
    ) -> Result<String, AgentError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == action.tool)
            .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;

        let input_str = match &action.tool_input {
            super::types::ToolInput::String { value: s } => s.clone(),
            super::types::ToolInput::Object { value: v } => serde_json::to_string(v)
                .map_err(|e| AgentError::Other(format!("Failed to serialize tool input: {}", e)))?,
        };

        let mut tool_run = root_run.create_child(
            &action.tool,
            RunType::Tool,
            json!({"input": input_str.clone()}),
        );

        if let Some(ref callbacks) = self.callbacks {
            for handler in callbacks.handlers() {
                handler
                    .on_tool_start(&tool_run, &action.tool, &input_str)
                    .await;
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

/// Helper: execute a single tool for streaming (no RunTree dependency).
async fn execute_tool_for_stream(
    tools: &[Arc<dyn BaseTool>],
    action: &AgentAction,
) -> Result<String, AgentError> {
    let tool = tools
        .iter()
        .find(|t| t.name() == action.tool)
        .ok_or_else(|| AgentError::ToolNotFound(action.tool.clone()))?;

    let input_str = match &action.tool_input {
        super::types::ToolInput::String { value: s } => s.clone(),
        super::types::ToolInput::Object { value: v } => serde_json::to_string(v)
            .map_err(|e| AgentError::Other(format!("Failed to serialize tool input: {}", e)))?,
    };

    tool.run(input_str)
        .await
        .map_err(|e| AgentError::ToolExecutionError(e.to_string()))
}

/// Helper: execute multiple tools in parallel for streaming.
async fn execute_tools_parallel_for_stream(
    tools: &[Arc<dyn BaseTool>],
    actions: &[AgentAction],
) -> Vec<String> {
    use futures_util::future::join_all;

    let futures: Vec<_> = actions
        .iter()
        .map(|action| execute_tool_for_stream(tools, action))
        .collect();

    let results = join_all(futures).await;
    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|e| format!("[Tool execution error: {}]", e)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_memory::ConversationBufferMemory;

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
        let executor = AgentExecutor::new(Arc::new(TestAgent), vec![]).with_memory(memory);

        // First conversation round
        let result1 = executor
            .invoke("My name is Zhang San".to_string())
            .await
            .unwrap();
        println!("Round 1: {}", result1);

        // Second conversation round - should remember the name
        let result2 = executor
            .invoke("What is my name?".to_string())
            .await
            .unwrap();
        println!("Round 2: {}", result2);

        assert!(result2.contains("Zhang San"));
    }
}
