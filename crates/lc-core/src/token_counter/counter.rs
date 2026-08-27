//! Token counter trait and usage statistics

use lc_schema::Message;

/// Token counter trait
pub trait TokenCounter: Send + Sync {
    /// Counts tokens in a text
    fn count_tokens(&self, text: &str) -> u32;
    /// Counts tokens in a message list
    fn count_messages(&self, messages: &[Message]) -> u32;
}

/// Char-ratio estimate counter (zero-dependency fast path)
///
/// `count_tokens = len / ratio`, no tiktoken BPE model needed, usable offline / in tests.
/// Default `ratio = 4` aligns with the old `len/4` rough estimate; `count_messages` reuses
/// `TiktokenCounter`'s overhead structure (4 per message + name + 1000/image, 2 at the end),
/// keeping the budget semantics consistent with the BPE accounting.
#[derive(Debug, Clone)]
pub struct CharRatioCounter {
    ratio: u32,
}

impl CharRatioCounter {
    /// Creates a char-ratio counter.
    ///
    /// `ratio` is the number of chars per token, at least 1.
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
        // at least 2*4 overhead + 2 boundary
        assert!(n >= 10);
    }
}

/// Token usage statistics (internal type for the counter module)
///
/// Note: `language_models::TokenUsage` is the usage returned by the LLM API (fields are `usize`);
/// this `TrackerTokenUsage` is locally tracked cumulative usage, **also `usize` fields**, so the two
/// convert without precision loss (Q6: unified base type, no more `usize` vs `u32` naming clash).
/// No `TokenUsage` alias is used here, to avoid confusion with `language_models::TokenUsage`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackerTokenUsage {
    /// Cumulative prompt token count
    pub prompt_tokens: usize,
    /// Cumulative completion token count
    pub completion_tokens: usize,
    /// Cumulative total token count (prompt + completion)
    pub total_tokens: usize,
}

impl TrackerTokenUsage {
    /// Creates empty usage statistics (all zero).
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulates usage
    pub fn add(&mut self, prompt: usize, completion: usize) {
        self.prompt_tokens += prompt;
        self.completion_tokens += completion;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    /// Resets usage statistics to all zero.
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
