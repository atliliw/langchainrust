// src/agents/types.rs
//! Agent 相关类型定义
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/agents.py

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent 动作
///
/// 表示 Agent 决定执行的一个动作（通常是调用工具）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    /// 工具名称
    pub tool: String,

    /// 工具输入（字符串或 JSON 对象）
    pub tool_input: ToolInput,

    /// 日志信息（包含完整的 LLM 输出）
    pub log: String,
}

/// 工具输入类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolInput {
    /// 字符串输入
    String(String),

    /// JSON 对象输入
    Object(serde_json::Value),
}

impl Default for ToolInput {
    fn default() -> Self {
        ToolInput::String(String::new())
    }
}

impl std::fmt::Display for ToolInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolInput::String(s) => write!(f, "{}", s),
            ToolInput::Object(v) => write!(f, "{}", serde_json::to_string(v).unwrap_or_default()),
        }
    }
}

/// Agent 完成状态
///
/// 表示 Agent 已经得出最终答案。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFinish {
    /// 返回值（键值对）
    pub return_values: HashMap<String, serde_json::Value>,

    /// 日志信息（包含完整的 LLM 输出）
    pub log: String,
}

impl AgentFinish {
    /// 创建新的 AgentFinish
    pub fn new(output: String, log: String) -> Self {
        let mut return_values = HashMap::new();
        return_values.insert("output".to_string(), serde_json::Value::String(output));
        Self { return_values, log }
    }

    /// 获取输出值
    pub fn output(&self) -> Option<&str> {
        self.return_values.get("output").and_then(|v| v.as_str())
    }
}

/// Agent 执行步骤
///
/// 表示一个已执行的动作及其观察结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    /// 执行的动作
    pub action: AgentAction,

    /// 观察结果（工具输出）
    pub observation: String,
}

impl AgentStep {
    /// 创建新的 AgentStep
    pub fn new(action: AgentAction, observation: String) -> Self {
        Self {
            action,
            observation,
        }
    }
}

/// Agent 输出
///
/// Agent 的 plan 方法可能返回动作或最终答案。
#[derive(Debug, Clone)]
pub enum AgentOutput {
    /// 执行动作
    Action(AgentAction),

    /// 完成（返回最终答案）
    Finish(AgentFinish),
}

impl AgentOutput {
    /// 是否为最终答案
    pub fn is_finish(&self) -> bool {
        matches!(self, AgentOutput::Finish(_))
    }

    /// 获取动作（如果有的话）
    pub fn action(&self) -> Option<&AgentAction> {
        match self {
            AgentOutput::Action(action) => Some(action),
            _ => None,
        }
    }

    /// 获取完成状态（如果有的话）
    pub fn finish(&self) -> Option<&AgentFinish> {
        match self {
            AgentOutput::Finish(finish) => Some(finish),
            _ => None,
        }
    }
}
