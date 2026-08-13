// lc-agents/src/hooks/logging.rs
//! LoggingHook — logs all hook points for debugging.
//!
//! Prints a log message at each hook point, useful for debugging
//! agent execution flow.

use async_trait::async_trait;

use super::{
    AgentHook, CompletionAction, CompletionContext, CompletionResult, ErrorAction, HookError,
    StreamAction, ToolCallAction, ToolCallContext, ToolResultContext,
};

/// A hook that logs all lifecycle events for debugging.
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::hooks::LoggingHook;
///
/// let hook = LoggingHook::new();
/// let executor = AgentExecutor::new(agent, tools).hook(hook);
/// ```
pub struct LoggingHook {
    /// Whether to log stream tokens (can be very verbose).
    log_tokens: bool,
}

impl LoggingHook {
    /// Creates a new LoggingHook.
    pub fn new() -> Self {
        Self { log_tokens: false }
    }

    /// Creates a LoggingHook that also logs stream tokens.
    pub fn with_tokens() -> Self {
        Self { log_tokens: true }
    }

    /// Sets whether to log stream tokens.
    pub fn with_log_tokens(mut self, log: bool) -> Self {
        self.log_tokens = log;
        self
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentHook for LoggingHook {
    fn on_before_completion(&self, ctx: &mut CompletionContext) -> CompletionAction {
        log::info!(
            "[Hook] LLM call starting: model={}, messages={}",
            ctx.model,
            ctx.messages.len()
        );
        CompletionAction::Continue
    }

    fn on_after_completion(&self, ctx: &mut CompletionResult) -> Result<(), HookError> {
        log::info!(
            "[Hook] LLM call completed: content_len={}",
            ctx.message.content.len()
        );
        Ok(())
    }

    fn on_before_tool_call(&self, ctx: &mut ToolCallContext) -> ToolCallAction {
        log::info!("[Hook] Tool call starting: name={}", ctx.name);
        ToolCallAction::Continue
    }

    fn on_after_tool_call(&self, ctx: &mut ToolResultContext) -> Result<(), HookError> {
        log::info!(
            "[Hook] Tool call completed: name={}, result_len={}",
            ctx.name,
            ctx.result.len()
        );
        Ok(())
    }

    fn on_stream_chunk(&self, chunk: &str) -> StreamAction {
        if self.log_tokens {
            log::debug!("[Hook] Stream token: {:?}", chunk);
        }
        StreamAction::Forward(chunk.to_string())
    }

    fn on_agent_start(&self, input: &str) -> Result<(), HookError> {
        log::info!("[Hook] Agent starting: input_len={}", input.len());
        Ok(())
    }

    fn on_agent_end(&self, output: &str) -> Result<(), HookError> {
        log::info!("[Hook] Agent completed: output_len={}", output.len());
        Ok(())
    }

    fn on_error(&self, error: &HookError) -> ErrorAction {
        log::warn!("[Hook] Error: {}", error);
        ErrorAction::Propagate
    }
}
