//! Tiktoken counter (OpenAI tokenizer)

use tiktoken_rs::CoreBPE;

use lc_schema::Message;

use super::counter::TokenCounter;
use super::TokenCounterError;

/// Tiktoken counter (uses cl100k_base, for GPT-3.5 / 4 / 4o)
pub struct TiktokenCounter {
    encoder: CoreBPE,
}

impl TiktokenCounter {
    /// Creates a Tiktoken counter (loads the cl100k_base BPE encoder).
    pub fn new() -> Result<Self, TokenCounterError> {
        let encoder = tiktoken_rs::cl100k_base()
            .map_err(|e| TokenCounterError::EncoderLoad(e.to_string()))?;
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
            total += 4; // OpenAI message-format overhead
            total += self.count_tokens(&msg.content);
            if let Some(name) = &msg.name {
                total += self.count_tokens(name);
            }
            // images count roughly as 1000 tokens each
            for _ in &msg.images {
                total += 1000;
            }
        }
        total += 2; // conversation boundary marker
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
        // at least 2*4 overhead + 2 boundary + per-message tokens
        assert!(n >= 10);
    }

    #[test]
    fn test_count_messages_with_image() {
        let counter = TiktokenCounter::new().unwrap();
        let msg = Message::human_with_image("看图", "https://example.com/x.png");
        let n = counter.count_messages(&[msg]);
        assert!(n >= 1000); // image = 1000 tokens
    }

    #[test]
    fn test_longer_text_more_tokens() {
        let counter = TiktokenCounter::new().unwrap();
        let short = counter.count_tokens("hi");
        let long = counter.count_tokens("This is a much longer sentence with many words.");
        assert!(long > short);
    }
}
