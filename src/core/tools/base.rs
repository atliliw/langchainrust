// src/core/tools/base.rs
//! Tool base traits.
//!
//! Python's BaseTool uses a simplified run(input: str) -> str interface.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

/// Base tool trait (object-safe version).
///
/// This is the base interface for tool registries and Agents.
/// Uses string input/output to simplify LLM calls.
///
/// All tools must implement this interface to be used by Agents.
#[async_trait]
pub trait BaseTool: Send + Sync {
    /// Returns the tool name.
    ///
    /// Name should be unique and clearly express the tool's purpose.
    fn name(&self) -> &str;
    
    /// Returns the tool description.
    ///
    /// Description should detail the tool's purpose, input format, and output format.
    fn description(&self) -> &str;
    
    /// Execute the tool (string version).
    ///
    /// This is the primary interface called by Agents.
    /// Input is typically a JSON string, output is the execution result.
    ///
    /// # Arguments
    /// * `input` - Tool input (typically JSON-formatted string).
    ///
    /// # Returns
    /// String representation of execution result.
    async fn run(&self, input: String) -> Result<String, ToolError>;
    
    /// Returns the input JSON Schema.
    ///
    /// Used to describe the tool's input format to the LLM.
    fn args_schema(&self) -> Option<Value> {
        None
    }
    
    /// Whether to return result directly to user.
    ///
    /// If true, tool output is returned directly to user, not passed to Agent.
    fn return_direct(&self) -> bool {
        false
    }
    
    /// Handle execution error.
    ///
    /// Returns a friendly error message when tool execution fails.
    async fn handle_error(&self, error: ToolError) -> String {
        format!("Tool '{}' execution failed: {}", self.name(), error)
    }
}

/// Generic tool trait (type-safe version).
///
/// For scenarios requiring type-safe input/output.
/// Tools implementing this trait can be automatically wrapped as BaseTool.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Input type (must support deserialization and JSON Schema).
    type Input: DeserializeOwned + JsonSchema + Send + Sync + 'static;
    
    /// Output type (must support serialization).
    type Output: Serialize + Send + Sync;
    
    /// Execute the tool.
    ///
    /// # Arguments
    /// * `input` - Tool input.
    ///
    /// # Returns
    /// Tool output.
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError>;
    
    /// Returns the input JSON Schema.
    fn args_schema(&self) -> Option<Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(Self::Input)).ok()
    }
}

/// Tool error type.
#[derive(Debug)]
pub enum ToolError {
    /// Input validation error.
    InvalidInput(String),
    
    /// Execution error.
    ExecutionFailed(String),
    
    /// Timeout.
    Timeout(u64),
    
    /// Tool not found.
    ToolNotFound(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ToolError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            ToolError::Timeout(seconds) => write!(f, "Timeout: {} seconds", seconds),
            ToolError::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
        }
    }
}

impl std::error::Error for ToolError {}

use super::ToolDefinition;

/// Converts BaseTool to ToolDefinition (for function calling).
///
/// # Arguments
/// * `tool` - Tool implementing BaseTool trait.
///
/// # Returns
/// ToolDefinition for bind_tools().
///
/// # Example
/// ```
/// use langchainrust::{Calculator, BaseTool, to_tool_definition};
/// use std::sync::Arc;
///
/// let calculator = Calculator::new();
/// let tool_def = to_tool_definition(&calculator);
/// ```
pub fn to_tool_definition(tool: &dyn BaseTool) -> ToolDefinition {
    ToolDefinition::new(tool.name(), tool.description())
        .with_parameters(
            tool.args_schema()
                .unwrap_or(serde_json::json!({"type": "object"}))
        )
}