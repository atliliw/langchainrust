//! MCP 工具适配器 - 将 MCP Tool 转为 BaseTool

use async_trait::async_trait;
use serde_json::Value;

use super::client::MCPClient;
use super::types::MCPToolDefinition;
use crate::core::tools::ToolError;
use crate::BaseTool;

/// MCP 工具适配器 - 把 MCP Server 暴露的工具包装为 `BaseTool`
pub struct MCPToolAdapter {
    client: MCPClient,
    definition: MCPToolDefinition,
}

impl MCPToolAdapter {
    pub fn new(client: MCPClient, definition: MCPToolDefinition) -> Self {
        Self { client, definition }
    }
}

#[async_trait]
impl BaseTool for MCPToolAdapter {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(self.definition.input_schema.clone())
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let args: Value = serde_json::from_str(&input)
            .map_err(|e| ToolError::ExecutionFailed(format!("Invalid JSON input: {}", e)))?;
        let result = self
            .client
            .call_tool(&self.definition.name, args)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(result.text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::MCPConfig;
    use serde_json::json;

    fn sample_definition() -> MCPToolDefinition {
        MCPToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[tokio::test]
    #[ignore = "需要本地 MCP SSE Server"]
    async fn test_adapter_metadata() {
        let client = MCPClient::connect(MCPConfig::sse("http://localhost:3001/sse"))
            .await
            .unwrap();
        let adapter = MCPToolAdapter::new(client, sample_definition());
        assert_eq!(adapter.name(), "read_file");
        assert_eq!(adapter.description(), "Read a file");
        assert!(adapter.args_schema().is_some());
    }
}
