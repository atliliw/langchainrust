// src/core/tools/base.rs
//! 工具基础 trait
//!
//! Python 的 BaseTool 使用 run(input: str) -> str 的简化接口。

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

/// 工具基础 trait（对象安全版本）
///
/// 这是用于工具注册表和 Agent 的基础接口。
/// 使用字符串输入/输出，简化 LLM 调用。
/// 
/// 所有工具必须实现这个接口才能被 Agent 使用。
#[async_trait]
pub trait BaseTool: Send + Sync {
    /// 获取工具名称
    /// 
    /// 名称应该是唯一的，能清晰表达工具的用途。
    fn name(&self) -> &str;
    
    /// 获取工具描述
    /// 
    /// 描述应该详细说明工具的用途、输入格式和输出格式。
    fn description(&self) -> &str;
    
    /// 执行工具（字符串版本）
    /// 
    /// 这是 Agent 调用的主要接口。
    /// 输入通常是 JSON 字符串，输出是执行结果。
    /// 
    /// # 参数
    /// * `input` - 工具输入（通常是 JSON 格式的字符串）
    /// 
    /// # 返回
    /// 执行结果的字符串表示
    async fn run(&self, input: String) -> Result<String, ToolError>;
    
    /// 获取输入的 JSON Schema
    /// 
    /// 用于向 LLM 描述工具的输入格式。
    fn args_schema(&self) -> Option<Value> {
        None
    }
    
    /// 是否直接返回结果给用户
    /// 
    /// 如果为 true，工具的输出会直接返回给用户，而不是传给 Agent。
    fn return_direct(&self) -> bool {
        false
    }
    
    /// 处理执行错误
    /// 
    /// 当工具执行失败时，可以返回一个友好的错误消息。
    async fn handle_error(&self, error: ToolError) -> String {
        format!("工具 '{}' 执行失败: {}", self.name(), error)
    }
}

/// 泛型工具 trait（类型安全版本）
///
/// 用于需要类型安全输入/输出的场景。
/// 实现这个 trait 的工具可以自动包装为 BaseTool。
#[async_trait]
pub trait Tool: Send + Sync {
    /// 输入类型（必须支持反序列化和 JSON Schema）
    type Input: DeserializeOwned + JsonSchema + Send + Sync + 'static;
    
    /// 输出类型（必须支持序列化）
    type Output: Serialize + Send + Sync;
    
    /// 执行工具
    /// 
    /// # 参数
    /// * `input` - 工具输入
    /// 
    /// # 返回
    /// 工具输出
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError>;
    
    /// 获取输入的 JSON Schema
    fn args_schema(&self) -> Option<Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(Self::Input)).ok()
    }
}

/// 工具错误类型
#[derive(Debug)]
pub enum ToolError {
    /// 输入验证错误
    InvalidInput(String),
    
    /// 执行错误
    ExecutionFailed(String),
    
    /// 超时
    Timeout(u64),
    
    /// 工具未找到
    ToolNotFound(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolError::InvalidInput(msg) => write!(f, "输入无效: {}", msg),
            ToolError::ExecutionFailed(msg) => write!(f, "执行失败: {}", msg),
            ToolError::Timeout(seconds) => write!(f, "执行超时: {}秒", seconds),
            ToolError::ToolNotFound(name) => write!(f, "工具未找到: {}", name),
        }
    }
}

impl std::error::Error for ToolError {}