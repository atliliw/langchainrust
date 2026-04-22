// src/core/tools/structured.rs
//! Structured tool wrapper.
//!
//! Wraps generic Tool as BaseTool with string interface.

use super::{BaseTool, Tool, ToolError};
use async_trait::async_trait;
use serde_json::Value;

/// Structured tool wrapper.
///
/// Wraps a Tool trait implementation as BaseTool,
/// automatically handling JSON input parsing and output serialization.
pub struct StructuredTool<T: Tool> {
    /// Inner tool instance.
    inner: T,
    /// Tool name.
    name: String,
    /// Tool description.
    description: String,
    /// JSON Schema.
    schema: Option<Value>,
}

impl<T: Tool> StructuredTool<T> {
    /// Creates a structured tool.
    ///
    /// # Arguments
    /// * `tool` - Inner tool instance.
    /// * `name` - Tool name (optional, defaults to "tool").
    /// * `description` - Tool description (optional, defaults to "A tool").
    pub fn new(tool: T, name: Option<&str>, description: Option<&str>) -> Self {
        let schema = tool.args_schema();
        Self {
            inner: tool,
            name: name.map(|s| s.to_string()).unwrap_or_else(|| "tool".to_string()),
            description: description.map(|s| s.to_string()).unwrap_or_else(|| "A tool".to_string()),
            schema,
        }
    }
    
    /// Parses JSON string to tool input type.
    fn parse_input(&self, input: String) -> Result<T::Input, ToolError> {
        // Parse as JSON
        let json: Value = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse failed: {}", e)))?;
        
        // Convert to target type
        serde_json::from_value(json)
            .map_err(|e| ToolError::InvalidInput(format!("Input format mismatch: {}", e)))
    }
    
    /// Serializes output to JSON string.
    fn serialize_output(output: T::Output) -> Result<String, ToolError> {
        serde_json::to_string(&output)
            .map_err(|e| ToolError::ExecutionFailed(format!("Output serialization failed: {}", e)))
    }
}

#[async_trait]
impl<T: Tool> BaseTool for StructuredTool<T> {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    async fn run(&self, input: String) -> Result<String, ToolError> {
        // Parse input
        let parsed_input = self.parse_input(input)?;
        
        // Execute tool
        let output = self.inner.invoke(parsed_input).await?;
        
        // Serialize output
        Self::serialize_output(output)
    }
    
    fn args_schema(&self) -> Option<Value> {
        self.schema.clone()
    }
    
    fn return_direct(&self) -> bool {
        false
    }
    
    async fn handle_error(&self, error: ToolError) -> String {
        format!("Tool '{}' execution failed: {}", self.name, error)
    }
}

impl<T: Tool> std::fmt::Debug for StructuredTool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuredTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}