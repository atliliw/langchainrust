// lc-agents/src/hooks/rate_limit.rs
//! TokenBudgetHook — LLM token budget / quota rate limiting (P2-9).
//!
//! Checks cumulative usage in `on_before_completion` (before every LLM call): if
//! the token budget or the call-count quota is exceeded it returns `Reject`,
//! which [`crate::base::AgentExecutor`] turns into an error that aborts the run;
//! `on_after_completion` accumulates the real token usage.

use async_trait::async_trait;
use lc_core::token_counter::{CharRatioCounter, TokenCounter};
use lc_schema::Message;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{AgentHook, CompletionAction, CompletionContext, CompletionResult, HookError};

/// Pre-call rate-limiting hook based on a token budget + call quota.
///
/// - `budget`: cumulative token cap for the whole run. `on_before_completion`
///   pre-checks "confirmed usage + estimated input", returning `Reject` when
///   exceeded; `on_after_completion` accumulates the real usage.
/// - `max_calls`: optional maximum number of LLM calls.
/// - Without a custom counter, input is estimated with
///   [`CharRatioCounter::new(4)`](CharRatioCounter) (characters / 4).
///
/// # Example
///
/// ```rust,ignore
/// use lc_agents::hooks::TokenBudgetHook;
///
/// let executor = AgentExecutor::new(agent, tools)
///     .hook(TokenBudgetHook::new(10_000).with_max_calls(20));
/// ```
pub struct TokenBudgetHook {
    /// Token budget cap.
    budget: usize,
    /// Optional maximum number of LLM calls.
    max_calls: Option<usize>,
    /// Cumulative token usage (real usage from completed LLM calls).
    tokens_used: AtomicUsize,
    /// Number of LLM calls made so far.
    calls: AtomicUsize,
    /// Precise counter (optional; defaults to characters/4 estimation).
    counter: Option<Arc<dyn TokenCounter>>,
}

impl TokenBudgetHook {
    /// Creates a token-budget hook. `budget` is the cumulative token cap for the whole run.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            max_calls: None,
            tokens_used: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            counter: None,
        }
    }

    /// Sets the maximum number of LLM calls.
    pub fn with_max_calls(mut self, max_calls: usize) -> Self {
        self.max_calls = Some(max_calls);
        self
    }

    /// Uses a precise token counter instead of character-ratio estimation.
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = Some(counter);
        self
    }

    /// Token budget cap.
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Cumulative token usage (real usage from completed LLM calls).
    pub fn tokens_used(&self) -> usize {
        self.tokens_used.load(Ordering::SeqCst)
    }

    /// Number of LLM calls made so far.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Remaining token budget (floored at 0).
    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.tokens_used())
    }

    /// Estimates the token count of a message list: custom counter first, otherwise characters/4.
    fn estimate_messages(&self, messages: &[Message]) -> usize {
        match &self.counter {
            Some(c) => c.count_messages(messages) as usize,
            None => CharRatioCounter::new(4).count_messages(messages) as usize,
        }
    }
}

#[async_trait]
impl AgentHook for TokenBudgetHook {
    fn on_before_completion(&self, ctx: &mut CompletionContext) -> CompletionAction {
        // Call quota.
        if let Some(max) = self.max_calls {
            if self.calls.load(Ordering::SeqCst) >= max {
                return CompletionAction::Reject {
                    reason: format!("LLM call quota exceeded: max_calls={max}"),
                };
            }
        }

        // Token-budget pre-check: confirmed usage + estimated input.
        let used = self.tokens_used.load(Ordering::SeqCst);
        let estimate = self.estimate_messages(&ctx.messages);
        if used.saturating_add(estimate) > self.budget {
            return CompletionAction::Reject {
                reason: format!(
                    "token budget exceeded: budget={}, used={used}, estimate={estimate}",
                    self.budget
                ),
            };
        }

        self.calls.fetch_add(1, Ordering::SeqCst);
        CompletionAction::Continue
    }

    fn on_after_completion(&self, ctx: &mut CompletionResult) -> Result<(), HookError> {
        if let Some(usage) = &ctx.tokens_used {
            self.tokens_used
                .fetch_add(usage.total_tokens, Ordering::SeqCst);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;

    fn completion_ctx(text: &str) -> CompletionContext {
        CompletionContext {
            messages: vec![Message::human(text.to_string())],
            model: "mock".to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_allows_within_budget() {
        let hook = TokenBudgetHook::new(1_000);
        assert!(matches!(
            hook.on_before_completion(&mut completion_ctx("short")),
            CompletionAction::Continue
        ));
        assert_eq!(hook.calls(), 1);
    }

    #[test]
    fn test_rejects_when_estimated_over_budget() {
        // Zero budget: any input (including message overhead) exceeds it.
        let hook = TokenBudgetHook::new(0);
        assert!(matches!(
            hook.on_before_completion(&mut completion_ctx("x")),
            CompletionAction::Reject { .. }
        ));
    }

    #[test]
    fn test_accumulates_real_usage_after_completion() {
        let hook = TokenBudgetHook::new(1_000);
        let mut result = CompletionResult {
            message: Message::ai("hi"),
            tokens_used: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        hook.on_after_completion(&mut result).unwrap();
        assert_eq!(hook.tokens_used(), 30);
        assert_eq!(hook.remaining(), 970);
    }

    #[test]
    fn test_max_calls_quota() {
        let hook = TokenBudgetHook::new(1_000).with_max_calls(2);
        assert!(matches!(
            hook.on_before_completion(&mut completion_ctx("a")),
            CompletionAction::Continue
        ));
        assert!(matches!(
            hook.on_before_completion(&mut completion_ctx("b")),
            CompletionAction::Continue
        ));
        // The third call exceeds the quota.
        let action = hook.on_before_completion(&mut completion_ctx("c"));
        match action {
            CompletionAction::Reject { reason } => assert!(reason.contains("quota"), "{reason}"),
            other => panic!("expected Reject, got {:?}", other),
        }
        assert_eq!(hook.calls(), 2);
    }

    #[test]
    fn test_rejects_after_real_usage_exceeds_budget() {
        let hook = TokenBudgetHook::new(100);
        // Real usage is already 90; adding the input estimate (>10) exceeds it.
        let mut result = CompletionResult {
            message: Message::ai("hi"),
            tokens_used: Some(TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 90,
                total_tokens: 90,
            }),
        };
        hook.on_after_completion(&mut result).unwrap();
        let action = hook.on_before_completion(&mut completion_ctx("a long enough message"));
        match action {
            CompletionAction::Reject { reason } => {
                assert!(reason.contains("budget"), "{reason}")
            }
            other => panic!("expected Reject, got {:?}", other),
        }
    }
}
