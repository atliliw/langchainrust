// lc-agents/src/hooks/approval.rs
//! ApprovalHook — blocks tool calls until the user approves them.
//!
//! Implements human-in-the-loop by requiring explicit approval before
//! each tool call. Useful for dangerous or expensive operations.

use async_trait::async_trait;

use super::{AgentHook, CompletionAction, CompletionContext, CompletionResult, ErrorAction, HookError,
            StreamAction, ToolCallAction, ToolCallContext, ToolResultContext};

/// A hook that requires user approval before each tool call.
///
/// When `on_before_tool_call` is invoked, it always returns `ToolCallAction::Reject`
/// with a message indicating approval is needed. In a real application, this
/// would block on user input (e.g., via a channel or callback).
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::hooks::ApprovalHook;
///
/// let hook = ApprovalHook::new();
/// let executor = AgentExecutor::new(agent, tools).hook(hook);
/// ```
pub struct ApprovalHook {
    /// If true, automatically approve all tool calls (useful for testing).
    auto_approve: bool,
}

impl ApprovalHook {
    /// Creates a new ApprovalHook that requires manual approval.
    pub fn new() -> Self {
        Self { auto_approve: false }
    }

    /// Creates an ApprovalHook that automatically approves all tool calls.
    pub fn auto_approve() -> Self {
        Self { auto_approve: true }
    }

    /// Sets the auto-approve mode.
    pub fn with_auto_approve(mut self, auto_approve: bool) -> Self {
        self.auto_approve = auto_approve;
        self
    }
}

impl Default for ApprovalHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentHook for ApprovalHook {
    fn on_before_tool_call(&self, ctx: &mut ToolCallContext) -> ToolCallAction {
        if self.auto_approve {
            ToolCallAction::Continue
        } else {
            ToolCallAction::Reject {
                reason: format!(
                    "Tool call '{}' requires manual approval. Arguments: {}",
                    ctx.name, ctx.arguments
                ),
            }
        }
    }
}
