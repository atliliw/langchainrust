//! Tiktoken 计数器(OpenAI tokenizer)

use tiktoken_rs::CoreBPE;

use crate::schema::Message;

use super::counter::TokenCounter;

/// Tiktoken 计数器(使用 cl100k_base,适用于 GPT-3.5 / 4 / 4o)
pub struct TiktokenCounter {
    encoder: CoreBPE,
}

impl TiktokenCounter {
    pub fn new() -> Result<Self, String> {
        let encoder =
            tiktoken_rs::cl100k_base().map_err(|e| format!("加载 tiktoken 失败: {}", e))?;
        Ok(Self { encoder })
    }
}

// H35: Removed Default impl that could panic.
// Use `TiktokenCounter::new()` instead of `TiktokenCounter::default()`.

impl TokenCounter for TiktokenCounter {
    fn count_tokens(&self, text: &str) -> u32 {
        self.encoder.encode_with_special_tokens(text).len() as u32
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
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_nonempty() {
        let counter = TiktokenCounter::new().unwrap();
        let n = counter.count_tokens("Hello, world!");
        assert!(n > 0);
    }

    #[test]
    fn test_count_tokens_empty() {
        let counter = TiktokenCounter::new().unwrap();
        assert_eq!(counter.count_tokens(""), 0);
    }

    #[test]
    fn test_count_messages_includes_overhead() {
        let counter = TiktokenCounter::new().unwrap();
        let msgs = vec![Message::system("You are helpful."), Message::human("Hi")];
        let n = counter.count_messages(&msgs);
        // 至少含 2*4 开销 + 2 边界 + 各消息 token
        assert!(n >= 10);
    }

    #[test]
    fn test_count_messages_with_image() {
        let counter = TiktokenCounter::new().unwrap();
        let msg = Message::human_with_image("看图", "https://example.com/x.png");
        let n = counter.count_messages(&[msg]);
        assert!(n >= 1000); // 图片 1000 token
    }

    #[test]
    fn test_longer_text_more_tokens() {
        let counter = TiktokenCounter::new().unwrap();
        let short = counter.count_tokens("hi");
        let long = counter.count_tokens("This is a much longer sentence with many words.");
        assert!(long > short);
    }
}
