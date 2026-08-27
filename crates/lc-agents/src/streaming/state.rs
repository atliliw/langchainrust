//! Streaming tool-call states and events

use serde_json::Value;

/// Tool-call stream state
#[derive(Debug, Clone)]
pub enum ToolCallState {
    /// Tool call started
    Started {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
    },
    /// Arguments streaming in
    ArgumentsStreaming {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
        /// Partial arguments streamed so far
        partial_args: String,
    },
    /// Arguments complete, ready to execute
    ArgumentsComplete {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
        /// Full tool arguments
        args: Value,
    },
    /// Tool executing
    Executing {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
    },
    /// Execution completed
    Completed {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
        /// Tool execution result
        result: String,
    },
    /// Execution failed
    Failed {
        /// Tool name
        tool_name: String,
        /// Call ID
        call_id: String,
        /// Failure reason
        error: String,
    },
}

impl ToolCallState {
    /// Returns the current tool name.
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

    /// Returns the current tool-call ID.
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

/// Agent streaming event
#[derive(Debug, Clone)]
pub enum AgentStreamEvent {
    /// LLM output text (token)
    Text {
        /// Output text content
        content: String,
    },

    /// Tool-call state change (Function Calling style)
    ToolCall {
        /// Tool-call state
        state: ToolCallState,
    },

    /// ReAct-style tool call start
    ToolStart {
        /// Tool name
        name: String,
        /// Tool input
        input: String,
    },

    /// ReAct-style tool call completion
    ToolEnd {
        /// Tool name
        name: String,
        /// Tool output
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

    /// Final answer
    FinalAnswer {
        /// Final answer content
        content: String,
    },

    /// Streaming execution error
    Error {
        /// Error message
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
