use async_trait::async_trait;
use std::collections::HashMap;

#[derive(Debug, Clone)]
/// Input passed to a tool invocation.
pub struct ToolInput {
    pub tool_name: String,
    pub parameters: HashMap<String, String>,
}

/// Standardized result of a tool invocation.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub success: bool,
    pub result: String,
}
/// Trait for tools that can be executed by agents.

#[async_trait]
    /// Unique name of the tool used in prompts and calls.
pub trait Tool: Send + Sync {
    /// Short description used in tool listing and prompting.
    fn name(&self) -> &str;
    /// Execute the tool with provided parameters.
    fn description(&self) -> &str;
    /// Parameter list describing accepted keys and their meaning.
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>>;
    /// If true, tool outputs are returned directly to user without further reasoning.
    fn parameters(&self) -> Vec<(&str, &str)>;
    fn return_direct(&self) -> bool;
}
