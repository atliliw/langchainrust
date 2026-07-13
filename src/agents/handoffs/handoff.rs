//! Handoff 类型定义

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Handoff 交接指令
pub struct Handoff {
    pub target_agent: String,
    pub task: String,
    pub context: Option<HandoffContext>,
}

/// 交接上下文 - 携带给目标 Agent 的信息
pub struct HandoffContext {
    pub original_request: String,
    pub current_result: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Handoff 结果
pub struct HandoffResult {
    pub agent_name: String,
    pub result: String,
    pub next_handoff: Option<Box<Handoff>>,
}

/// 交接历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    pub from_agent: String,
    pub to_agent: String,
    pub task: String,
    pub result: String,
    pub timestamp: String,
}

/// Handoff 错误
#[derive(Debug)]
pub enum HandoffError {
    AgentNotFound(String),
    ExecutionError(String),
}

impl std::fmt::Display for HandoffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandoffError::AgentNotFound(name) => {
                write!(f, "Agent 不存在: {}", name)
            }
            HandoffError::ExecutionError(msg) => write!(f, "Agent 执行错误: {}", msg),
        }
    }
}

impl std::error::Error for HandoffError {}
