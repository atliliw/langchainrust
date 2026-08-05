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
