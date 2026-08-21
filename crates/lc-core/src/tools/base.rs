// src/core/tools/base.rs
//! Tool base traits.
//!
//! Python's BaseTool uses a simplified run(input: str) -> str interface.

use crate::runnables::{LcelError, Runnable, RunnableConfig};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::sync::Arc;

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
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Input validation error.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Execution error.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Timeout.
    #[error("Timeout: {0} seconds")]
    Timeout(u64),

    /// Tool not found.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// MCP 传输层错误(经 `MCPToolAdapter` 适配),保留 code/message/data(P1-6)。
    ///
    /// 不静默降级为 `ExecutionFailed`,上层可据 `code` 区分连接断开 / 方法不存在 /
    /// 参数错误等场景。
    #[error("MCP error [{code}]: {message}")]
    McpError {
        code: i32,
        message: String,
        data: Option<Value>,
    },
}

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
/// ```ignore
/// use langchainrust::{Calculator, BaseTool, to_tool_definition};
/// use std::sync::Arc;
///
/// let calculator = Calculator::new();
/// let tool_def = to_tool_definition(&calculator);
/// ```
pub fn to_tool_definition(tool: &dyn BaseTool) -> ToolDefinition {
    ToolDefinition::new(tool.name(), tool.description()).with_parameters(
        tool.args_schema()
            .unwrap_or(serde_json::json!({"type": "object"})),
    )
}

// Runnable 形态:让工具能进 LCEL 链,`tool.pipe(...)` 成立。
// 接收 String(通常是 JSON 输入),委托给 `run`,错误经 `From<ToolError>` 进 `LcelError::Tool`。
#[async_trait]
impl Runnable<String, String> for Arc<dyn BaseTool> {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        BaseTool::run(&**self, input).await.map_err(LcelError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// 简单回显工具:返回 `echo: {input}`。
    struct EchoTool;

    #[async_trait]
    impl BaseTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "回显输入"
        }
        async fn run(&self, input: String) -> Result<String, ToolError> {
            Ok(format!("echo: {input}"))
        }
    }

    #[tokio::test]
    async fn arc_tool_is_runnable() {
        let tool: Arc<dyn BaseTool> = Arc::new(EchoTool);
        let result = tool.invoke("hi".to_string(), None).await.unwrap();
        assert_eq!(result, "echo: hi");
    }

    #[tokio::test]
    async fn arc_tool_pipes() {
        use crate::runnables::{RunnableExt, RunnableLambda};

        let tool: Arc<dyn BaseTool> = Arc::new(EchoTool);
        let chain = tool.pipe(RunnableLambda::new_sync(|s: String| s.to_uppercase()));
        let result = chain.invoke("hi".to_string(), None).await.unwrap();
        assert_eq!(result, "ECHO: HI");
    }

    #[tokio::test]
    async fn arc_tool_error_maps_to_lcel() {
        struct FailingTool;
        #[async_trait]
        impl BaseTool for FailingTool {
            fn name(&self) -> &str {
                "fail"
            }
            fn description(&self) -> &str {
                "总是失败"
            }
            async fn run(&self, _input: String) -> Result<String, ToolError> {
                Err(ToolError::ExecutionFailed("boom".to_string()))
            }
        }

        let tool: Arc<dyn BaseTool> = Arc::new(FailingTool);
        let err = tool.invoke("x".to_string(), None).await.unwrap_err();
        assert!(matches!(err, LcelError::Tool(ref msg) if msg.contains("boom")));
    }
}
