// lc-agents/src/executor/hooks.rs
//! Completion hooks run around each LLM call (rate limiting / quota).

use super::AgentError;
use crate::hooks::{AgentHook, CompletionAction, CompletionContext, CompletionResult};
use crate::types::AgentOutput;
use lc_core::language_models::TokenUsage;
use lc_schema::Message;
use std::collections::HashMap;
use std::sync::Arc;

/// P2-9: run completion hooks before each LLM call (rate limiting / quota).
///
/// Builds a [`CompletionContext`] then calls `on_before_completion` on each hook:
/// - `Continue` → allow;
/// - `Modify` → the executor cannot rewrite the Agent's own prompt, so it logs a warn
///   and continues;
/// - `Reject { reason }` → converted into an `AgentError` that aborts this round.
pub(crate) fn run_before_completion_hooks(
    hooks: &[Arc<dyn AgentHook>],
    inputs: &HashMap<String, String>,
) -> Result<(), AgentError> {
    let messages = inputs
        .values()
        .map(|v| Message::human(v.clone()))
        .collect::<Vec<_>>();
    let mut ctx = CompletionContext {
        messages,
        model: "agent".to_string(),
        metadata: HashMap::new(),
    };
    for hook in hooks {
        match hook.on_before_completion(&mut ctx) {
            CompletionAction::Continue => {}
            CompletionAction::Modify { .. } => {
                log::warn!(
                    target: "lc_agents::security",
                    "CompletionAction::Modify ignored at AgentExecutor level (agent builds its own prompt)"
                );
            }
            CompletionAction::Reject { reason } => {
                return Err(AgentError::Other(format!(
                    "LLM call rejected by hook: {reason}"
                )));
            }
        }
    }
    Ok(())
}

/// P2-9: run completion hooks after each LLM call (accumulate token usage).
///
/// Builds a [`CompletionResult`] then calls `on_after_completion` on each hook; a hook
/// error only logs a warn and does not abort execution (same tolerance policy as
/// `on_after_tool_call`).
pub(crate) fn run_after_completion_hooks(
    hooks: &[Arc<dyn AgentHook>],
    output: &AgentOutput,
    token_usage: Option<&TokenUsage>,
) {
    let message = Message::ai(match output {
        AgentOutput::Finish(finish) => finish.output().unwrap_or("").to_string(),
        _ => String::new(),
    });
    let mut ctx = CompletionResult {
        message,
        tokens_used: token_usage.cloned(),
    };
    for hook in hooks {
        if let Err(e) = hook.on_after_completion(&mut ctx) {
            log::warn!("Hook on_after_completion error: {}", e);
        }
    }
}
