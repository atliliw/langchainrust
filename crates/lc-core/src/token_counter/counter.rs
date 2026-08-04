//! Token 计数器 trait 与用量统计

use lc_schema::Message;

/// Token 计数器 trait
pub trait TokenCounter: Send + Sync {
    /// 计算文本 token 数
    fn count_tokens(&self, text: &str) -> u32;
    /// 计算消息列表 token 数
    fn count_messages(&self, messages: &[Message]) -> u32;
}

/// Token 用量统计（计数器模块内部类型）
///
/// 注意：`language_models::TokenUsage` 是 LLM API 返回的用量（字段为 `usize`），
/// 此 `TrackerTokenUsage` 是本地追踪累计用量（字段为 `u32`），两者职责不同。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackerTokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TrackerTokenUsage {
    pub fn new() -> Self {
        Self::default()
    }

    /// 累加用量
    pub fn add(&mut self, prompt: u32, completion: u32) {
        self.prompt_tokens += prompt;
        self.completion_tokens += completion;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// Re-export as TokenUsage for backward compatibility within this module
pub use TrackerTokenUsage as TokenUsage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_add() {
        let mut u = TrackerTokenUsage::new();
        u.add(10, 20);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn test_usage_accumulate() {
        let mut u = TrackerTokenUsage::new();
        u.add(10, 20);
        u.add(5, 5);
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.total_tokens, 40);
    }

    #[test]
    fn test_usage_reset() {
        let mut u = TrackerTokenUsage::new();
        u.add(10, 20);
        u.reset();
        assert_eq!(u, TrackerTokenUsage::new());
    }
}
