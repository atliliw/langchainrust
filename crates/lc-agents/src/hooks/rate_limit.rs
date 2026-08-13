// lc-agents/src/hooks/rate_limit.rs
//! TokenBudgetHook — LLM token 预算 / 配额限流(P2-9)。
//!
//! 在 `on_before_completion`(每个 LLM 调用前)检查累计用量:超出 token 预算或
//! 超过调用次数配额 → `Reject`,由 [`crate::base::AgentExecutor`] 把 Reject
//! 转成错误中止;`on_after_completion` 累加真实 token 用量。

use async_trait::async_trait;
use lc_core::token_counter::{CharRatioCounter, TokenCounter};
use lc_schema::Message;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{AgentHook, CompletionAction, CompletionContext, CompletionResult, HookError};

/// 基于 token 预算 + 调用配额的前置限流 hook。
///
/// - `budget`:整轮执行累计 token 上限。`on_before_completion` 用"已确认用量 +
///   本次输入估算"预检,超出即 `Reject`;`on_after_completion` 累加真实用量。
/// - `max_calls`:可选的最大 LLM 调用次数配额。
/// - 无自定义计数器时用 [`CharRatioCounter::new(4)`](CharRatioCounter)(字符/4)
///   估算输入。
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
    /// token 预算上限。
    budget: usize,
    /// 可选的最大 LLM 调用次数配额。
    max_calls: Option<usize>,
    /// 累计 token 用量(来自已完成 LLM 调用的真实用量)。
    tokens_used: AtomicUsize,
    /// 已发生的 LLM 调用次数。
    calls: AtomicUsize,
    /// 精确计数器(可选;缺省用字符/4 估算)。
    counter: Option<Arc<dyn TokenCounter>>,
}

impl TokenBudgetHook {
    /// 创建 token 预算 hook。`budget` 为整轮执行累计 token 上限。
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            max_calls: None,
            tokens_used: AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            counter: None,
        }
    }

    /// 设置最大 LLM 调用次数配额。
    pub fn with_max_calls(mut self, max_calls: usize) -> Self {
        self.max_calls = Some(max_calls);
        self
    }

    /// 使用精确 token 计数器替代字符比估算。
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = Some(counter);
        self
    }

    /// token 预算上限。
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// 累计 token 用量(来自已完成 LLM 调用的真实用量)。
    pub fn tokens_used(&self) -> usize {
        self.tokens_used.load(Ordering::SeqCst)
    }

    /// 已发生的 LLM 调用次数。
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// 剩余可用 token 预算(下限 0)。
    pub fn remaining(&self) -> usize {
        self.budget.saturating_sub(self.tokens_used())
    }

    /// 估算消息列表 token 数:自定义计数器优先,否则字符/4。
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
        // 调用配额。
        if let Some(max) = self.max_calls {
            if self.calls.load(Ordering::SeqCst) >= max {
                return CompletionAction::Reject {
                    reason: format!("LLM call quota exceeded: max_calls={max}"),
                };
            }
        }

        // token 预算预检:已确认用量 + 本次输入估算。
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
        // 空预算:任何输入(含消息开销)都超。
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
        // 第三次超配额。
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
        // 真实用量已 90,再加输入估算(>10)即超。
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
