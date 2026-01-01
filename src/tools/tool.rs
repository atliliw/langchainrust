use std::collections::HashMap;
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct ToolInput {
    pub tool_name: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub success: bool,
    pub result: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    
    fn description(&self) -> &str;
    
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, Box<dyn std::error::Error>>;
    
    fn parameters(&self) -> Vec<(&str, &str)>;
}