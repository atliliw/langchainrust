// lc-agents/src/hooks/mod.rs
//! Agent Hook/Middleware system for composable lifecycle interception.
//!
//! Hooks allow injecting custom behavior at key points in the agent execution
//! loop: before/after LLM calls, before/after tool calls, on stream tokens,
//! and on errors.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_agents::hooks::{AgentHook, ApprovalHook, ContentFilterHook};
//! use lc_agents::AgentExecutor;
//!
//! let executor = AgentExecutor::new(agent, tools)
//!     .hook(ApprovalHook::new())           // Require approval before tool calls
//!     .hook(ContentFilterHook::new(words)); // Filter sensitive words from stream
//! ```

mod approval;
mod content_filter;
mod logging;

pub use approval::ApprovalHook;
pub use content_filter::ContentFilterHook;
pub use logging::LoggingHook;

use async_trait::async_trait;
use lc_schema::Message;
use serde_json::Value;
use std::collections::HashMap;

/// Error type for hook operations.
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    /// The hook rejected the operation.
    #[error("Hook rejected: {0}")]
    Rejected(String),

    /// The hook encountered an error.
    #[error("Hook error: {0}")]
    Other(String),
}

/// Action to take for a completion (LLM call).
#[derive(Debug, Clone)]
pub enum CompletionAction {
    /// Allow the completion to proceed.
    Continue,
    /// Modify the messages before the LLM call.
    Modify { messages: Vec<Message> },
    /// Reject the LLM call entirely.
    Reject { reason: String },
}

/// Action to take for a tool call.
#[derive(Debug, Clone)]
pub enum ToolCallAction {
    /// Allow the tool call to proceed.
    Continue,
    /// Modify the tool call parameters.
    Modify { name: String, arguments: Value },
    /// Reject the tool call.
    Reject { reason: String },
    /// Skip this tool call (don't execute, don't error).
    Skip,
}

/// Action to take for a stream chunk.
#[derive(Debug, Clone)]
pub enum StreamAction {
    /// Forward the token to the stream.
    Forward(String),
    /// Filter (drop) this token.
    Filter,
    /// Replace the token with different content.
    Replace(String),
}

/// Action to take on error.
#[derive(Debug, Clone)]
pub enum ErrorAction {
    /// Propagate the error normally.
    Propagate,
    /// Retry the operation.
    Retry,
    /// Ignore the error and continue.
    Ignore,
}

/// Context for a completion (LLM call) hook.
#[derive(Debug, Clone)]
pub struct CompletionContext {
    /// The messages being sent to the LLM.
    pub messages: Vec<Message>,
    /// The model being used.
    pub model: String,
    /// Additional metadata.
    pub metadata: HashMap<String, Value>,
}

/// Result context after a completion (LLM call).
#[derive(Debug, Clone)]
pub struct CompletionResult {
    /// The response message from the LLM.
    pub message: Message,
    /// Token usage if available.
    pub tokens_used: Option<lc_core::language_models::TokenUsage>,
}

/// Context for a tool call hook.
#[derive(Debug, Clone)]
pub struct ToolCallContext {
    /// The tool name.
    pub name: String,
    /// The tool arguments.
    pub arguments: Value,
    /// The tool call ID (for function calling style).
    pub tool_id: String,
}

/// Result context after a tool call.
#[derive(Debug, Clone)]
pub struct ToolResultContext {
    /// The tool name.
    pub name: String,
    /// The tool result.
    pub result: String,
    /// The tool call ID.
    pub tool_id: String,
}

/// Trait for agent lifecycle hooks.
///
/// Implement this trait to inject custom behavior at key points in the
/// agent execution loop. All methods have default no-op implementations,
/// so you only need to override the ones you care about.
#[async_trait]
pub trait AgentHook: Send + Sync {
    /// Called before an LLM completion. Can modify messages or reject the call.
    fn on_before_completion(&self, _ctx: &mut CompletionContext) -> CompletionAction {
        CompletionAction::Continue
    }

    /// Called after an LLM completion. Can modify the response.
    fn on_after_completion(&self, _ctx: &mut CompletionResult) -> Result<(), HookError> {
        Ok(())
    }

    /// Called before a tool call. Can approve, reject, modify, or skip.
    fn on_before_tool_call(&self, _ctx: &mut ToolCallContext) -> ToolCallAction {
        ToolCallAction::Continue
    }

    /// Called after a tool call. Can modify the result.
    fn on_after_tool_call(&self, _ctx: &mut ToolResultContext) -> Result<(), HookError> {
        Ok(())
    }

    /// Called for each streaming token. Can filter, replace, or forward.
    fn on_stream_chunk(&self, chunk: &str) -> StreamAction {
        StreamAction::Forward(chunk.to_string())
    }

    /// Called when the agent starts execution.
    fn on_agent_start(&self, _input: &str) -> Result<(), HookError> {
        Ok(())
    }

    /// Called when the agent finishes execution.
    fn on_agent_end(&self, _output: &str) -> Result<(), HookError> {
        Ok(())
    }

    /// Called when an error occurs. Can retry, ignore, or propagate.
    fn on_error(&self, _error: &HookError) -> ErrorAction {
        ErrorAction::Propagate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_action_default_continue() {
        let action = CompletionAction::Continue;
        assert!(matches!(action, CompletionAction::Continue));
    }

    #[test]
    fn test_tool_call_action_variants() {
        let continue_action = ToolCallAction::Continue;
        let modify_action = ToolCallAction::Modify {
            name: "calc".to_string(),
            arguments: serde_json::json!({"x": 1}),
        };
        let reject_action = ToolCallAction::Reject {
            reason: "not allowed".to_string(),
        };
        let skip_action = ToolCallAction::Skip;

        assert!(matches!(continue_action, ToolCallAction::Continue));
        assert!(matches!(modify_action, ToolCallAction::Modify { .. }));
        assert!(matches!(reject_action, ToolCallAction::Reject { .. }));
        assert!(matches!(skip_action, ToolCallAction::Skip));
    }

    #[test]
    fn test_stream_action_variants() {
        let forward = StreamAction::Forward("hello".to_string());
        let filter = StreamAction::Filter;
        let replace = StreamAction::Replace("[REDACTED]".to_string());

        assert!(matches!(forward, StreamAction::Forward(_)));
        assert!(matches!(filter, StreamAction::Filter));
        assert!(matches!(replace, StreamAction::Replace(_)));
    }

    #[test]
    fn test_error_action_variants() {
        assert!(matches!(ErrorAction::Propagate, ErrorAction::Propagate));
        assert!(matches!(ErrorAction::Retry, ErrorAction::Retry));
        assert!(matches!(ErrorAction::Ignore, ErrorAction::Ignore));
    }

    #[test]
    fn test_hook_error_display() {
        let rejected = HookError::Rejected("not allowed".to_string());
        assert_eq!(format!("{}", rejected), "Hook rejected: not allowed");

        let other = HookError::Other("something broke".to_string());
        assert_eq!(format!("{}", other), "Hook error: something broke");
    }

    #[test]
    fn test_completion_context_default() {
        let ctx = CompletionContext {
            messages: vec![],
            model: "gpt-4".to_string(),
            metadata: HashMap::new(),
        };
        assert_eq!(ctx.model, "gpt-4");
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn test_tool_call_context() {
        let ctx = ToolCallContext {
            name: "calculator".to_string(),
            arguments: serde_json::json!({"expr": "2+2"}),
            tool_id: "call_123".to_string(),
        };
        assert_eq!(ctx.name, "calculator");
        assert_eq!(ctx.tool_id, "call_123");
    }

    #[test]
    fn test_tool_result_context() {
        let ctx = ToolResultContext {
            name: "calculator".to_string(),
            result: "4".to_string(),
            tool_id: "call_123".to_string(),
        };
        assert_eq!(ctx.result, "4");
    }

    #[test]
    fn test_completion_result() {
        let result = CompletionResult {
            message: lc_schema::Message::ai("Hello!"),
            tokens_used: None,
        };
        assert_eq!(result.message.content, "Hello!");
    }

    /// A custom hook that tracks all hook calls for testing.
    struct TrackingHook {
        before_completion_called: std::sync::atomic::AtomicBool,
        after_completion_called: std::sync::atomic::AtomicBool,
        before_tool_called: std::sync::atomic::AtomicBool,
        after_tool_called: std::sync::atomic::AtomicBool,
        agent_start_called: std::sync::atomic::AtomicBool,
        agent_end_called: std::sync::atomic::AtomicBool,
        error_called: std::sync::atomic::AtomicBool,
    }

    impl TrackingHook {
        fn new() -> Self {
            Self {
                before_completion_called: std::sync::atomic::AtomicBool::new(false),
                after_completion_called: std::sync::atomic::AtomicBool::new(false),
                before_tool_called: std::sync::atomic::AtomicBool::new(false),
                after_tool_called: std::sync::atomic::AtomicBool::new(false),
                agent_start_called: std::sync::atomic::AtomicBool::new(false),
                agent_end_called: std::sync::atomic::AtomicBool::new(false),
                error_called: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl AgentHook for TrackingHook {
        fn on_before_completion(&self, _ctx: &mut CompletionContext) -> CompletionAction {
            self.before_completion_called.store(true, std::sync::atomic::Ordering::SeqCst);
            CompletionAction::Continue
        }

        fn on_after_completion(&self, _ctx: &mut CompletionResult) -> Result<(), HookError> {
            self.after_completion_called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn on_before_tool_call(&self, _ctx: &mut ToolCallContext) -> ToolCallAction {
            self.before_tool_called.store(true, std::sync::atomic::Ordering::SeqCst);
            ToolCallAction::Continue
        }

        fn on_after_tool_call(&self, _ctx: &mut ToolResultContext) -> Result<(), HookError> {
            self.after_tool_called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn on_agent_start(&self, _input: &str) -> Result<(), HookError> {
            self.agent_start_called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn on_agent_end(&self, _output: &str) -> Result<(), HookError> {
            self.agent_end_called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn on_error(&self, _error: &HookError) -> ErrorAction {
            self.error_called.store(true, std::sync::atomic::Ordering::SeqCst);
            ErrorAction::Propagate
        }
    }

    #[test]
    fn test_custom_hook_tracking() {
        let hook = TrackingHook::new();

        // Simulate hook calls
        let mut ctx = CompletionContext {
            messages: vec![],
            model: "gpt-4".to_string(),
            metadata: HashMap::new(),
        };
        hook.on_before_completion(&mut ctx);
        assert!(hook.before_completion_called.load(std::sync::atomic::Ordering::SeqCst));

        hook.on_agent_start("test input").unwrap();
        assert!(hook.agent_start_called.load(std::sync::atomic::Ordering::SeqCst));

        hook.on_agent_end("test output").unwrap();
        assert!(hook.agent_end_called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_completion_action_reject() {
        let action = CompletionAction::Reject {
            reason: "blocked".to_string(),
        };
        if let CompletionAction::Reject { reason } = action {
            assert_eq!(reason, "blocked");
        } else {
            panic!("Expected Reject");
        }
    }

    #[test]
    fn test_completion_action_modify() {
        let action = CompletionAction::Modify {
            messages: vec![lc_schema::Message::system("test")],
        };
        if let CompletionAction::Modify { messages } = action {
            assert_eq!(messages.len(), 1);
        } else {
            panic!("Expected Modify");
        }
    }
}
