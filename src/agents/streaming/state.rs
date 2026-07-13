//! 流式工具调用状态与事件

use serde_json::Value;

/// 工具调用流状态
#[derive(Debug, Clone)]
pub enum ToolCallState {
    /// 工具调用开始
    Started { tool_name: String, call_id: String },
    /// 参数正在流式传输
    ArgumentsStreaming {
        tool_name: String,
        call_id: String,
        partial_args: String,
    },
    /// 参数完成,准备执行
    ArgumentsComplete {
        tool_name: String,
        call_id: String,
        args: Value,
    },
    /// 工具正在执行
    Executing { tool_name: String, call_id: String },
    /// 执行完成
    Completed {
        tool_name: String,
        call_id: String,
        result: String,
    },
    /// 执行失败
    Failed {
        tool_name: String,
        call_id: String,
        error: String,
    },
}

impl ToolCallState {
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
    Text { content: String },
    /// 工具调用状态变化
    ToolCall { state: ToolCallState },
    /// 最终答案
    FinalAnswer { content: String },
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
}
