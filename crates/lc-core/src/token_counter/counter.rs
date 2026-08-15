//! Token 计数器 trait 与用量统计

use lc_schema::Message;

/// Token 计数器 trait
pub trait TokenCounter: Send + Sync {
    /// 计算文本 token 数
    fn count_tokens(&self, text: &str) -> u32;
    /// 计算消息列表 token 数
    fn count_messages(&self, messages: &[Message]) -> u32;
}

/// 字符比估算计数器(零依赖快路径)
///
/// `count_tokens = len / ratio`,不依赖 tiktoken BPE 模型,离线/测试环境可用。
/// 默认 `ratio = 4` 对齐旧的 `len/4` 粗略估算;`count_messages` 沿用
/// `TiktokenCounter` 的开销结构(每条消息 4 + 名称 + 图片 1000/张,结尾 2),
/// 保证与 BPE 口径的预算语义一致。
#[derive(Debug, Clone)]
pub struct CharRatioCounter {
    ratio: u32,
}

impl CharRatioCounter {
    /// 创建字符比计数器。
    ///
    /// `ratio` 为每 token 的字符数,至少为 1。
    pub fn new(ratio: u32) -> Self {
        Self {
            ratio: ratio.max(1),
        }
    }
}

impl TokenCounter for CharRatioCounter {
    fn count_tokens(&self, text: &str) -> u32 {
        text.len() as u32 / self.ratio
    }

    fn count_messages(&self, messages: &[Message]) -> u32 {
        let mut total = 0u32;
        for msg in messages {
            total += 4; // OpenAI 消息格式开销
            total += self.count_tokens(&msg.content);
            if let Some(name) = &msg.name {
                total += self.count_tokens(name);
            }
            // 图片内容粗略计为 1000 token/图
            for _ in &msg.images {
                total += 1000;
            }
        }
        total += 2; // 对话边界标记
        total
    }
}

#[cfg(test)]
mod char_ratio_tests {
    use super::*;

    #[test]
    fn test_char_ratio_counts_bytes() {
        let counter = CharRatioCounter::new(4);
        assert_eq!(counter.count_tokens(""), 0);
        assert_eq!(counter.count_tokens("Hello"), 1); // 5 / 4 = 1
        assert_eq!(counter.count_tokens("Hello World"), 2); // 11 / 4 = 2
    }

    #[test]
    fn test_char_ratio_ratio_at_least_one() {
        let counter = CharRatioCounter::new(0);
        assert!(counter.count_tokens("x") >= 1);
    }

    #[test]
    fn test_char_ratio_count_messages_matches_structure() {
        let counter = CharRatioCounter::new(4);
        let msgs = vec![Message::system("You are helpful."), Message::human("Hi")];
        let n = counter.count_messages(&msgs);
        // 至少含 2*4 开销 + 2 边界
        assert!(n >= 10);
    }
}

/// Token 用量统计（计数器模块内部类型）
///
/// 注意：`language_models::TokenUsage` 是 LLM API 返回的用量（字段为 `usize`），
/// 此 `TrackerTokenUsage` 是本地追踪累计用量，**字段同为 `usize`**，两者可互转
/// 而无精度损失（Q6：统一底层类型，不再存在 `usize` vs `u32` 的命名冲突）。
/// 这里不使用 `TokenUsage` 别名，避免与 `language_models::TokenUsage` 混淆。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackerTokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TrackerTokenUsage {
    pub fn new() -> Self {
        Self::default()
    }

    /// 累加用量
    pub fn add(&mut self, prompt: usize, completion: usize) {
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
