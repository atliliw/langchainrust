// src/core/tools/structured.rs
//! 结构化工具
//!
//! 将泛型 Tool 包装为 BaseTool，提供字符串接口。

use super::{BaseTool, Tool, ToolError};
use async_trait::async_trait;
use serde_json::Value;

/// 结构化工具包装器
/// 
/// 将实现了 `Tool` trait 的工具包装为 `BaseTool`，
/// 自动处理 JSON 输入解析和输出序列化。
pub struct StructuredTool<T: Tool> {
    /// 内部工具
    inner: T,
    /// 工具名称
    name: String,
    /// 工具描述
    description: String,
    /// JSON Schema
    schema: Option<Value>,
}

impl<T: Tool> StructuredTool<T> {
    /// 创建结构化工具
    /// 
    /// # 参数
    /// * `tool` - 内部工具实例
    /// * `name` - 工具名称（可选，默认使用内部工具名称）
    /// * `description` - 工具描述（可选，默认使用内部工具描述）
    pub fn new(tool: T, name: Option<&str>, description: Option<&str>) -> Self {
        let schema = tool.args_schema();
        Self {
            inner: tool,
            name: name.map(|s| s.to_string()).unwrap_or_else(|| "tool".to_string()),
            description: description.map(|s| s.to_string()).unwrap_or_else(|| "A tool".to_string()),
            schema,
        }
    }
    
    /// 验证输入
    /// 
    /// 将 JSON 字符串转换为工具的输入类型。
    fn parse_input(&self, input: String) -> Result<T::Input, ToolError> {
        // 尝试解析为 JSON
        let json: Value = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;
        
        // 转换为目标类型
        serde_json::from_value(json)
            .map_err(|e| ToolError::InvalidInput(format!("输入格式不匹配: {}", e)))
    }
    
    /// 序列化输出
    fn serialize_output(output: T::Output) -> Result<String, ToolError> {
        serde_json::to_string(&output)
            .map_err(|e| ToolError::ExecutionFailed(format!("输出序列化失败: {}", e)))
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
        // 解析输入
        let parsed_input = self.parse_input(input)?;
        
        // 执行工具
        let output = self.inner.invoke(parsed_input).await?;
        
        // 序列化输出
        Self::serialize_output(output)
    }
    
    fn args_schema(&self) -> Option<Value> {
        self.schema.clone()
    }
    
    fn return_direct(&self) -> bool {
        false
    }
    
    async fn handle_error(&self, error: ToolError) -> String {
        format!("工具 '{}' 执行失败: {}", self.name, error)
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