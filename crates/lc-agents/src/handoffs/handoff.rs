//! Handoff 类型定义

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Handoff 交接指令
#[derive(Debug)]
pub struct Handoff {
    pub target_agent: String,
    pub task: String,
    pub context: Option<HandoffContext>,
}

/// 交接上下文 - 携带给目标 Agent 的信息
#[derive(Debug)]
pub struct HandoffContext {
    pub original_request: String,
    pub current_result: Option<String>,
    /// 当前对话摘要(P2-4):交接时把上游对话摘要带给目标 Agent,
    /// 而非裸转移控制权,目标 Agent 能延续话题而非从零开始。
    pub conversation_summary: Option<String>,
    pub metadata: HashMap<String, Value>,
}

impl HandoffContext {
    /// 创建一个交接上下文,记录原始请求。
    pub fn new(original_request: impl Into<String>) -> Self {
        Self {
            original_request: original_request.into(),
            current_result: None,
            conversation_summary: None,
            metadata: HashMap::new(),
        }
    }

    /// 携带当前对话摘要。
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.conversation_summary = Some(summary.into());
        self
    }

    /// 携带当前执行结果。
    pub fn with_result(mut self, result: impl Into<String>) -> Self {
        self.current_result = Some(result.into());
        self
    }
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
    /// 交接环检测:A 交接给 B,B 又交接给 A,无限循环(P1-7)。
    HandoffCycleDetected(String),
    /// 交接深度超过上限(P1-7)。
    MaxHandoffDepthExceeded(usize),
}

impl std::fmt::Display for HandoffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandoffError::AgentNotFound(name) => {
                write!(f, "Agent 不存在: {}", name)
            }
            HandoffError::ExecutionError(msg) => write!(f, "Agent 执行错误: {}", msg),
            HandoffError::HandoffCycleDetected(name) => {
                write!(f, "检测到交接环: {} 已在交接链中,拒绝循环交接", name)
            }
            HandoffError::MaxHandoffDepthExceeded(depth) => {
                write!(f, "交接深度超过上限: {}", depth)
            }
        }
    }
}

impl std::error::Error for HandoffError {}
