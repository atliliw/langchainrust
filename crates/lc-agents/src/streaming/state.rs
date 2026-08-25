//! 流式工具调用状态与事件

use serde_json::Value;

/// 工具调用流状态
#[derive(Debug, Clone)]
pub enum ToolCallState {
    /// 工具调用开始
    Started {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
    },
    /// 参数正在流式传输
    ArgumentsStreaming {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
        /// 已流式传输的部分参数
        partial_args: String,
    },
    /// 参数完成,准备执行
    ArgumentsComplete {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
        /// 完整的工具参数
        args: Value,
    },
    /// 工具正在执行
    Executing {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
    },
    /// 执行完成
    Completed {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
        /// 工具执行结果
        result: String,
    },
    /// 执行失败
    Failed {
        /// 工具名称
        tool_name: String,
        /// 调用 ID
        call_id: String,
        /// 失败原因
        error: String,
    },
}

impl ToolCallState {
    /// 返回当前工具名称。
    pub fn tool_name(&self) -> &str {
        match self {
            ToolCallState::Started { tool_name, .. }
            | ToolCallState::ArgumentsStreaming { tool_name, .. }
            | ToolCallState::ArgumentsComplete { tool_name, .. }
            | ToolCallState::Executing { tool_name, .. }
            | ToolCallState::Completed { tool_name, .. }
            | ToolCallState::Failed { tool_name, .. } => tool_name,
        }
    }

    /// 返回当前工具调用 ID。
    pub fn call_id(&self) -> &str {
        match self {
            ToolCallState::Started { call_id, .. }
            | ToolCallState::ArgumentsStreaming { call_id, .. }
            | ToolCallState::ArgumentsComplete { call_id, .. }
            | ToolCallState::Executing { call_id, .. }
            | ToolCallState::Completed { call_id, .. }
            | ToolCallState::Failed { call_id, .. } => call_id,
        }
    }
}

/// Agent 流式事件
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    /// LLM 输出文本(token)
    Text {
        /// 输出的文本内容
        content: String,
    },

    /// 工具调用状态变化 (Function Calling 风格)
    ToolCall {
        /// 工具调用状态
        state: ToolCallState,
    },

    /// ReAct 风格工具调用开始
    ToolStart {
        /// 工具名称
        name: String,
        /// 工具输入
        input: String,
    },

    /// ReAct 风格工具调用完成
    ToolEnd {
        /// 工具名称
        name: String,
        /// 工具输出
        output: String,
    },

    /// Pipeline step event (for RAG/research agents).
    /// Indicates which stage of the pipeline is currently executing.
    PipelineStep {
        /// Step name (e.g., "retrieving", "grading", "generating", "planning", "searching", "synthesizing").
        step: String,
        /// Optional detail message.
        detail: Option<String>,
    },

    /// 最终答案
    FinalAnswer {
        /// 最终答案内容
        content: String,
    },

    /// 流式执行错误
    Error {
        /// 错误信息
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_state_accessors() {
        let s = ToolCallState::Started {
            tool_name: "calc".to_string(),
            call_id: "call_1".to_string(),
        };
        assert_eq!(s.tool_name(), "calc");
        assert_eq!(s.call_id(), "call_1");
    }

    #[test]
    fn test_completed_state() {
        let s = ToolCallState::Completed {
            tool_name: "search".to_string(),
            call_id: "call_2".to_string(),
            result: "结果".to_string(),
        };
        assert_eq!(s.tool_name(), "search");
        assert_eq!(s.call_id(), "call_2");
    }

    #[test]
    fn test_agent_stream_event_text() {
        let e = AgentStreamEvent::Text {
            content: "hello".to_string(),
        };
        assert!(matches!(e, AgentStreamEvent::Text { .. }));
    }

    #[test]
    fn test_agent_stream_event_final() {
        let e = AgentStreamEvent::FinalAnswer {
            content: "done".to_string(),
        };
        assert!(matches!(e, AgentStreamEvent::FinalAnswer { .. }));
    }

    #[test]
    fn test_agent_stream_event_error() {
        let e = AgentStreamEvent::Error {
            message: "stream failed".to_string(),
        };
        assert!(matches!(e, AgentStreamEvent::Error { .. }));
    }
}
