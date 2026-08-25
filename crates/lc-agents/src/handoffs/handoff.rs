//! Handoff 类型定义

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Handoff 交接指令
#[derive(Debug)]
pub struct Handoff {
    /// 目标 Agent 名称
    pub target_agent: String,
    /// 交接的任务描述
    pub task: String,
    /// 交接上下文
    pub context: Option<HandoffContext>,
}

/// 交接上下文 - 携带给目标 Agent 的信息
#[derive(Debug)]
pub struct HandoffContext {
    /// 原始请求内容
    pub original_request: String,
    /// 当前执行结果
    pub current_result: Option<String>,
    /// 当前对话摘要(P2-4):交接时把上游对话摘要带给目标 Agent,
    /// 而非裸转移控制权,目标 Agent 能延续话题而非从零开始。
    pub conversation_summary: Option<String>,
    /// 附加元数据
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
    /// 目标 Agent 名称
    pub agent_name: String,
    /// 交接执行结果
    pub result: String,
    /// 下一个交接指令(可选)
    pub next_handoff: Option<Box<Handoff>>,
}

/// 交接历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    /// 源 Agent 名称
    pub from_agent: String,
    /// 目标 Agent 名称
    pub to_agent: String,
    /// 交接的任务描述
    pub task: String,
    /// 交接结果
    pub result: String,
    /// 交接时间戳
    pub timestamp: String,
}

/// Handoff 错误
#[derive(Debug)]
#[non_exhaustive]
pub enum HandoffError {
    /// 目标 Agent 不存在
    AgentNotFound(String),
    /// Agent 执行错误
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
                write!(f, "Agent does not exist: {}", name)
            }
            HandoffError::ExecutionError(msg) => write!(f, "Agent execution error: {}", msg),
            HandoffError::HandoffCycleDetected(name) => {
                write!(f, "handoff cycle detected: {} already in the handoff chain, cyclic handoff rejected", name)
            }
            HandoffError::MaxHandoffDepthExceeded(depth) => {
                write!(f, "handoff depth exceeded the limit: {}", depth)
            }
        }
    }
}

impl std::error::Error for HandoffError {}
