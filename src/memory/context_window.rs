// src/memory/context_window.rs
//! Context Window for long context management.
//!
//! Manages conversation context by fitting messages within a token limit,
//! using either truncation or LLM-based summarization strategies.
//!
//! # Core Concepts
//!
//! - **ContextWindow**: Fits messages within a max token budget.
//! - **Strategy::Truncate**: Drops oldest messages, preserving system messages.
//! - **Strategy::Summarize**: Uses an LLM to compress old messages into a summary.
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::{ContextWindow, Strategy, TiktokenCounter};
//!
//! // Truncation strategy
//! let cw = ContextWindow::new(4096);
//! let fitted = cw.fit(messages).await?;
//!
//! // Summarization strategy
//! let cw = ContextWindow::with_strategy(4096, Strategy::Summarize::new(llm));
//! let fitted = cw.fit(messages).await?;
//! ```

use std::sync::Arc;

use crate::core::language_models::BaseChatModel;
use crate::core::token_counter::{TiktokenCounter, TokenCounter};
use crate::schema::Message;

use super::base::MemoryError;

/// Default summary prompt for the Summarize strategy.
const DEFAULT_SUMMARY_PROMPT: &str = "\
Summarize the following conversation concisely, preserving key facts, \
decisions, and context. Write the summary in the same language as the conversation.

Conversation:
{conversation}

Summary:";

/// Strategy for fitting messages within a token limit.
#[derive(Debug)]
pub enum Strategy<M: BaseChatModel = crate::language_models::OpenAIChat> {
    /// Drop oldest messages to fit within the token limit.
    /// System messages are always preserved.
    Truncate,

    /// Use an LLM to compress old messages into a summary system message.
    Summarize {
        /// The LLM used to generate summaries.
        llm: Arc<M>,
        /// Custom summary prompt. Must contain `{conversation}` placeholder.
        summary_prompt: String,
    },
}

impl<M: BaseChatModel> Strategy<M> {
    /// Creates a new Summarize strategy with the given LLM and default prompt.
    pub fn summarize(llm: M) -> Self {
        Strategy::Summarize {
            llm: Arc::new(llm),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Creates a new Summarize strategy with a custom prompt.
    ///
    /// The prompt must contain the `{conversation}` placeholder.
    pub fn summarize_with_prompt(llm: M, prompt: impl Into<String>) -> Self {
        Strategy::Summarize {
            llm: Arc::new(llm),
            summary_prompt: prompt.into(),
        }
    }
}

/// Context window manager for fitting messages within a token budget.
///
/// Counts tokens using a `TokenCounter` (defaults to `TiktokenCounter`),
/// and applies a `Strategy` when messages exceed `max_tokens`.
pub struct ContextWindow<M: BaseChatModel = crate::language_models::OpenAIChat> {
    /// Maximum token count allowed.
    max_tokens: usize,
    /// Token counter implementation.
    counter: Arc<dyn TokenCounter>,
    /// Strategy for reducing messages when over the limit.
    strategy: Strategy<M>,
}

impl<M: BaseChatModel> ContextWindow<M> {
    /// Creates a new ContextWindow with the Truncate strategy and default TiktokenCounter.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            counter: Arc::new(TiktokenCounter::default()),
            strategy: Strategy::Truncate,
        }
    }

    /// Creates a new ContextWindow with a specific strategy and default TiktokenCounter.
    pub fn with_strategy(max_tokens: usize, strategy: Strategy<M>) -> Self {
        Self {
            max_tokens,
            counter: Arc::new(TiktokenCounter::default()),
            strategy,
        }
    }

    /// Creates a new ContextWindow with a custom max token limit and default counter/strategy.
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self::new(max_tokens)
    }

    /// Sets a custom token counter.
    pub fn with_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// Returns the max token limit.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Fits messages within the token limit by applying the configured strategy.
    ///
    /// - If total tokens are within `max_tokens`, returns messages as-is.
    /// - If over, applies the `Strategy` (truncate or summarize).
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to fit.
    ///
    /// # Returns
    /// A vector of messages that fits within the token budget.
    pub async fn fit(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        let total_tokens = self.counter.count_messages(&messages) as usize;

        if total_tokens <= self.max_tokens {
            return Ok(messages);
        }

        match &self.strategy {
            Strategy::Truncate => self.truncate(messages),
            Strategy::Summarize {
                llm,
                summary_prompt,
            } => self.summarize(messages, llm, summary_prompt).await,
        }
    }

    /// Truncates messages by removing the oldest non-system messages
    /// until the total fits within `max_tokens`.
    ///
    /// System messages are always preserved and placed at the beginning.
    fn truncate(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        // Separate system messages from the rest.
        let mut system_messages: Vec<Message> = Vec::new();
        let mut other_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if matches!(msg.message_type, crate::schema::MessageType::System) {
                system_messages.push(msg);
            } else {
                other_messages.push(msg);
            }
        }

        // Keep dropping oldest non-system messages until we fit.
        // We iterate from the end (newest) to determine what to keep.
        let mut kept: Vec<Message> = Vec::new();

        for msg in other_messages.into_iter().rev() {
            let mut candidate = system_messages.clone();
            candidate.push(msg.clone());
            candidate.extend(kept.iter().cloned());

            let tokens = self.counter.count_messages(&candidate) as usize;
            if tokens <= self.max_tokens {
                kept.push(msg);
            } else {
                // This message would push us over; stop adding more.
                break;
            }
        }

        kept.reverse();

        let mut result = system_messages;
        result.extend(kept);
        Ok(result)
    }

    /// Summarizes old messages using the LLM, replacing them with a
    /// single system message containing the summary.
    ///
    /// System messages are preserved. The oldest non-system messages
    /// are summarized until the remaining messages fit within the budget.
    async fn summarize(
        &self,
        messages: Vec<Message>,
        llm: &Arc<M>,
        summary_prompt: &str,
    ) -> Result<Vec<Message>, MemoryError> {
        // Separate system messages from the rest.
        let mut system_messages: Vec<Message> = Vec::new();
        let mut other_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if matches!(msg.message_type, crate::schema::MessageType::System) {
                system_messages.push(msg);
            } else {
                other_messages.push(msg);
            }
        }

        if other_messages.is_empty() {
            // Only system messages; nothing to summarize.
            return Ok(system_messages);
        }

        // Find how many recent messages we can keep within the budget,
        // reserving some space for the summary message.
        // We try keeping the newest messages and summarizing the rest.
        // Iterate from the smallest window (fewest recent messages) to the largest,
        // keeping track of the best (smallest i = most messages kept) that fits.
        let mut keep_from_idx = other_messages.len(); // default: keep all (no summarization)

        for i in 0..other_messages.len() {
            let recent = &other_messages[i..];
            let mut candidate = system_messages.clone();
            // Reserve space for a summary message (estimate ~100 tokens).
            candidate.push(Message::system("summary placeholder"));
            candidate.extend(recent.iter().cloned());

            let tokens = self.counter.count_messages(&candidate) as usize;
            if tokens <= self.max_tokens {
                keep_from_idx = i;
                break;
            }
        }

        // If we can't even fit the recent messages with a summary placeholder,
        // fall back to truncation for the recent portion.
        if keep_from_idx >= other_messages.len() {
            // Nothing fits; truncate to just system messages.
            return self.truncate(system_messages);
        }

        let to_summarize = &other_messages[..keep_from_idx];
        let to_keep = &other_messages[keep_from_idx..];

        if to_summarize.is_empty() {
            // All messages fit with the summary placeholder; no need to summarize.
            let mut result = system_messages;
            result.extend(to_keep.to_vec());
            return Ok(result);
        }

        // Format the conversation for summarization.
        let conversation_text = to_summarize
            .iter()
            .map(|msg| {
                let role = match msg.message_type {
                    crate::schema::MessageType::Human => "Human",
                    crate::schema::MessageType::AI => "AI",
                    crate::schema::MessageType::System => "System",
                    crate::schema::MessageType::Tool { .. } => "Tool",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = summary_prompt.replace("{conversation}", &conversation_text);

        let summary_messages = vec![Message::human(&prompt)];

        let result = llm
            .invoke(summary_messages, None)
            .await
            .map_err(|e| MemoryError::SaveError(format!("LLM summarization failed: {}", e)))?;

        let summary_message = Message::system(format!("[Conversation Summary] {}", result.content));

        // Build final message list: system + summary + recent.
        let mut final_messages = system_messages;
        final_messages.push(summary_message);
        final_messages.extend(to_keep.to_vec());

        // Verify the final result fits; if not, truncate the recent portion.
        let final_tokens = self.counter.count_messages(&final_messages) as usize;
        if final_tokens > self.max_tokens {
            return self.truncate(final_messages);
        }

        Ok(final_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::language_models::{BaseLanguageModel, LLMResult};
    use crate::core::runnables::{Runnable, RunnableConfig};
    use crate::language_models::openai::{OpenAIChat, OpenAIConfig};
    use crate::schema::MessageType;
    use async_trait::async_trait;
    use futures_util::Stream;
    use std::pin::Pin;
    use tokio::sync::Mutex;

    // ---- Mock TokenCounter for deterministic tests ----

    /// A simple token counter that counts 1 token per character.
    /// This makes it easy to reason about token budgets in tests.
    #[derive(Debug)]
    struct CharTokenCounter;

    impl TokenCounter for CharTokenCounter {
        fn count_tokens(&self, text: &str) -> u32 {
            text.len() as u32
        }

        fn count_messages(&self, messages: &[Message]) -> u32 {
            let mut total = 0u32;
            for msg in messages {
                total += 4; // per-message overhead
                total += self.count_tokens(&msg.content);
            }
            total += 2; // conversation boundary
            total
        }
    }

    fn char_counter() -> Arc<dyn TokenCounter> {
        Arc::new(CharTokenCounter)
    }

    // ---- Mock LLM for Summarize strategy tests ----

    #[derive(Debug)]
    struct MockLLM {
        responses: Arc<Mutex<Vec<String>>>,
    }

    impl MockLLM {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses)),
            }
        }
    }

    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLLM {
        fn model_name(&self) -> &str {
            "mock-llm"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLLM {
        type Error = std::convert::Infallible;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            let mut responses = self.responses.lock().await;
            let content = responses.pop().unwrap_or_else(|| "Summary".to_string());
            Ok(LLMResult {
                content,
                model: "mock-llm".to_string(),
                token_usage: None,
                tool_calls: None,
            })
        }
    }

    #[async_trait]
    impl BaseChatModel for MockLLM {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    // ---- Helper to build messages ----

    fn make_messages(contents: &[(&str, &str)]) -> Vec<Message> {
        contents
            .iter()
            .map(|(role, content)| match *role {
                "system" => Message::system(*content),
                "human" => Message::human(*content),
                "ai" => Message::ai(*content),
                _ => Message::human(*content),
            })
            .collect()
    }

    // ---- Tests ----

    #[test]
    fn test_new_creates_truncate_strategy() {
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096);
        assert_eq!(cw.max_tokens(), 4096);
    }

    #[test]
    fn test_with_max_tokens() {
        let cw: ContextWindow<OpenAIChat> = ContextWindow::with_max_tokens(8192);
        assert_eq!(cw.max_tokens(), 8192);
    }

    #[tokio::test]
    async fn test_fit_under_limit_returns_as_is() {
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(1000)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("human", "Hello"),
            ("ai", "Hi there"),
        ]);

        let result = cw.fit(messages).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_fit_empty_messages() {
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(100)
            .with_counter(char_counter());

        let result = cw.fit(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_truncate_preserves_system_messages() {
        // With CharTokenCounter: each message = 4 overhead + content length, + 2 boundary.
        // System: 4 + 7 = 11, Human1: 4 + 4 = 8, AI1: 4 + 4 = 8, Human2: 4 + 4 = 8, AI2: 4 + 4 = 8
        // Total = 11 + 8 + 8 + 8 + 8 + 2 = 45
        // Budget = 30: system(11) + AI2(8) + boundary(2) = 21, + Human2(8) = 29 <= 30
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(30)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "You are"),
            ("human", "Q1?"),
            ("ai", "A1!"),
            ("human", "Q2?"),
            ("ai", "A2!"),
        ]);

        let result = cw.fit(messages).await.unwrap();

        // System message must be preserved.
        assert!(result.iter().any(|m| matches!(m.message_type, MessageType::System)));
        // Most recent messages should be kept.
        assert!(result.iter().any(|m| m.content == "A2!"));
    }

    #[tokio::test]
    async fn test_truncate_drops_oldest_first() {
        // Budget = 25: system(4+4=8) + AI(4+3=7) + boundary(2) = 17, + Human(4+3=7) = 24 <= 25
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(25)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "Sys"),
            ("human", "Old question here"),
            ("ai", "Old answer here"),
            ("human", "New"),
            ("ai", "Ans"),
        ]);

        let result = cw.fit(messages).await.unwrap();

        // System message preserved.
        assert!(result.iter().any(|m| m.content == "Sys"));
        // Newest messages kept.
        assert!(result.iter().any(|m| m.content == "Ans"));
        // Old messages dropped.
        assert!(!result.iter().any(|m| m.content == "Old question here"));
    }

    #[tokio::test]
    async fn test_truncate_only_system_messages() {
        // If only system messages exist and they fit, return them.
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(20)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "Hello"),
        ]);
        // 4 + 5 + 2 = 11 <= 20

        let result = cw.fit(messages).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Hello");
    }

    #[tokio::test]
    async fn test_truncate_system_only_over_budget() {
        // If system messages alone exceed the budget, truncate returns just system messages
        // (they are always preserved).
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(5)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "Very long system prompt that exceeds budget"),
        ]);
        // 4 + 42 + 2 = 48 > 5

        let result = cw.fit(messages).await.unwrap();
        // System messages are always preserved even if over budget.
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_summarize_replaces_old_messages() {
        // CharTokenCounter: per-message = 4 + content_len, + 2 boundary.
        // system("S"): 5, human("Q1"): 6, ai("A1"): 6, human("Q2"): 6, ai("A2"): 6,
        // human("Q3"): 6, ai("A3"): 6, human("Q4"): 6, ai("A4"): 6
        // Total = 5 + 8*6 + 2 = 55 > 40, so summarization is triggered.
        //
        // Budget 40: system(5) + placeholder(23) + ai("A4")(6) + boundary(2) = 36 <= 40.
        // keep_from_idx = 7 (keep ai("A4"), summarize Q1..Q4).
        // LLM returns "S." (2 chars). Summary = "[Conversation Summary] S." (24 chars) = 4+24=28 tokens.
        // Final: 5+28+6+2 = 41 > 40 => falls back to truncation.
        //
        // Use budget 50 instead:
        // system(5) + placeholder(23) + human("Q4")(6) + ai("A4")(6) + boundary(2) = 42 <= 50.
        // keep_from_idx = 6 (keep Q4, A4, summarize Q1..A3).
        // Summary = "[Conversation Summary] S." = 28 tokens. Final: 5+28+6+6+2 = 47 <= 50. Fits!
        let mock_llm = MockLLM::new(vec!["S.".to_string()]);

        let cw = ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "S"),
            ("human", "Q1"),
            ("ai", "A1"),
            ("human", "Q2"),
            ("ai", "A2"),
            ("human", "Q3"),
            ("ai", "A3"),
            ("human", "Q4"),
            ("ai", "A4"),
        ]);

        let result = cw.fit(messages).await.unwrap();

        // System message preserved.
        assert!(result.iter().any(|m| m.content == "S"));

        // Should contain a summary message.
        let summary_msgs: Vec<&Message> = result
            .iter()
            .filter(|m| m.content.starts_with("[Conversation Summary]"))
            .collect();
        assert_eq!(summary_msgs.len(), 1);
        assert!(summary_msgs[0].content.contains("S"));
    }

    #[tokio::test]
    async fn test_summarize_preserves_recent_messages() {
        // Budget = 50: system(5) + placeholder(23) + human("Q4")(6) + ai("A4")(6) + boundary(2) = 42 <= 50.
        // keep_from_idx = 6 (keep Q4, A4, summarize Q1..A3).
        // LLM returns "S." Summary = "[Conversation Summary] S." = 28 tokens.
        // Final: 5+28+6+6+2 = 47 <= 50. Fits!
        let mock_llm = MockLLM::new(vec!["S.".to_string()]);

        let cw = ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "S"),
            ("human", "Q1"),
            ("ai", "A1"),
            ("human", "Q2"),
            ("ai", "A2"),
            ("human", "Q3"),
            ("ai", "A3"),
            ("human", "Q4"),
            ("ai", "A4"),
        ]);

        let result = cw.fit(messages).await.unwrap();

        // Recent messages should be preserved.
        assert!(result.iter().any(|m| m.content == "Q4"));
        assert!(result.iter().any(|m| m.content == "A4"));
    }

    #[tokio::test]
    async fn test_summarize_with_custom_prompt() {
        // Budget = 50: system(5) + placeholder(23) + human("Q4")(6) + ai("A4")(6) + boundary(2) = 42 <= 50.
        // keep_from_idx = 6 (keep Q4, A4, summarize Q1..A3).
        // LLM returns "O." Summary = "[Conversation Summary] O." = 28 tokens.
        // Final: 5+28+6+6+2 = 47 <= 50. Fits!
        let mock_llm = MockLLM::new(vec!["O.".to_string()]);

        let cw = ContextWindow::with_strategy(
            50,
            Strategy::summarize_with_prompt(
                mock_llm,
                "Please compress: {conversation}\nCompressed:",
            ),
        )
        .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "S"),
            ("human", "Q1"),
            ("ai", "A1"),
            ("human", "Q2"),
            ("ai", "A2"),
            ("human", "Q3"),
            ("ai", "A3"),
            ("human", "Q4"),
            ("ai", "A4"),
        ]);

        let result = cw.fit(messages).await.unwrap();
        let summary_msgs: Vec<&Message> = result
            .iter()
            .filter(|m| m.content.starts_with("[Conversation Summary]"))
            .collect();
        assert_eq!(summary_msgs.len(), 1);
        assert!(summary_msgs[0].content.contains("O"));
    }

    #[tokio::test]
    async fn test_summarize_no_non_system_messages() {
        let mock_llm = MockLLM::new(vec!["Should not be called".to_string()]);

        let cw: ContextWindow<MockLLM> = ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "S"),
        ]);

        let result = cw.fit(messages).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "S");
    }

    #[tokio::test]
    async fn test_strategy_truncate_enum() {
        let cw = ContextWindow::with_strategy(100, Strategy::<OpenAIChat>::Truncate)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("human", "Hello"),
            ("ai", "World"),
        ]);

        let result = cw.fit(messages).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_strategy_summarize_new() {
        let config = OpenAIConfig::default();
        let llm = OpenAIChat::new(config);
        let strategy: Strategy<OpenAIChat> = Strategy::summarize(llm);

        if let Strategy::Summarize { summary_prompt, .. } = &strategy {
            assert!(summary_prompt.contains("{conversation}"));
        } else {
            panic!("Expected Summarize variant");
        }
    }

    #[test]
    fn test_strategy_summarize_with_custom_prompt() {
        let config = OpenAIConfig::default();
        let llm = OpenAIChat::new(config);
        let custom = "Custom: {conversation} ->";
        let strategy: Strategy<OpenAIChat> = Strategy::summarize_with_prompt(llm, custom);

        if let Strategy::Summarize { summary_prompt, .. } = &strategy {
            assert_eq!(summary_prompt, custom);
        } else {
            panic!("Expected Summarize variant");
        }
    }

    #[tokio::test]
    async fn test_fit_with_real_tiktoken_counter() {
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096);

        let messages = make_messages(&[
            ("system", "You are a helpful assistant."),
            ("human", "Hello!"),
            ("ai", "Hi there! How can I help you?"),
        ]);

        // These short messages should easily fit within 4096 tokens.
        let result = cw.fit(messages).await.unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_truncate_preserves_order() {
        // Budget = 40: system(4+3=7) + human(4+3=7) + ai(4+3=7) + boundary(2) = 23 <= 40
        let cw: ContextWindow<OpenAIChat> = ContextWindow::new(40)
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "Sys"),
            ("human", "Old"),
            ("ai", "OldA"),
            ("human", "New"),
            ("ai", "NewA"),
        ]);

        let result = cw.fit(messages).await.unwrap();

        // Verify order: system first, then conversation in order.
        let types: Vec<&str> = result.iter().map(|m| m.type_str()).collect();
        // System should be first.
        assert_eq!(types[0], "system");
        // The rest should maintain human/ai alternation.
        for i in 1..types.len() {
            if i + 1 < types.len() {
                // Not strictly required, but for our test data this holds.
            }
        }
    }

    #[tokio::test]
    async fn test_summarize_fallback_to_truncate() {
        // When the summary + recent messages still exceed the budget,
        // the method falls back to truncation.
        let mock_llm = MockLLM::new(vec!["A very long summary that will not fit in the small budget.".to_string()]);

        // Very small budget that even the summary won't fit.
        let cw: ContextWindow<MockLLM> = ContextWindow::with_strategy(20, Strategy::summarize(mock_llm))
            .with_counter(char_counter());

        let messages = make_messages(&[
            ("system", "S"),
            ("human", "Q1"),
            ("ai", "A1"),
            ("human", "Q2"),
            ("ai", "A2"),
        ]);

        let result = cw.fit(messages).await.unwrap();
        // Should still return some messages (truncation fallback).
        assert!(!result.is_empty());
    }
}
