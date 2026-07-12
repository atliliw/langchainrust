//! Token 计数器 trait 与用量统计

use crate::schema::Message;

/// Token 计数器 trait
pub trait TokenCounter: Send + Sync {
    /// 计算文本 token 数
    fn count_tokens(&self, text: &str) -> u32;
    /// 计算消息列表 token 数
    fn count_messages(&self, messages: &[Message]) -> u32;
}

/// Token 用量统计
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_add() {
        let mut u = TokenUsage::new();
        u.add(10, 20);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn test_usage_accumulate() {
        let mut u = TokenUsage::new();
        u.add(10, 20);
        u.add(5, 5);
        assert_eq!(u.prompt_tokens, 15);
        assert_eq!(u.total_tokens, 40);
    }

    #[test]
    fn test_usage_reset() {
        let mut u = TokenUsage::new();
        u.add(10, 20);
        u.reset();
        assert_eq!(u, TokenUsage::new());
    }
}
