// lc-memory/src/context_window/tests.rs
//! Tests for context_window module.

use super::*;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::token_counter::TokenCounter;
use lc_providers::{OpenAIChat, OpenAIConfig};
use lc_schema::{Message, MessageType};
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
            thinking_content: None,
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
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
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096).unwrap();
    assert_eq!(cw.max_tokens(), 4096);
}

#[test]
fn test_with_max_tokens() {
    let cw: ContextWindow<OpenAIChat> = ContextWindow::with_max_tokens(8192).unwrap();
    assert_eq!(cw.max_tokens(), 8192);
}

#[tokio::test]
async fn test_fit_under_limit_returns_as_is() {
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(1000)
        .unwrap()
        .with_counter(char_counter());

    let messages = make_messages(&[("human", "Hello"), ("ai", "Hi there")]);

    let result = cw.fit(messages).await.unwrap();
    assert_eq!(result.len(), 2);
}

#[tokio::test]
async fn test_fit_empty_messages() {
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(100)
        .unwrap()
        .with_counter(char_counter());

    let result = cw.fit(vec![]).await.unwrap();
    assert!(result.is_empty());
}

#[tokio::test]
async fn test_truncate_preserves_system_messages() {
    let cw: ContextWindow<OpenAIChat> =
        ContextWindow::new(30).unwrap().with_counter(char_counter());

    let messages = make_messages(&[
        ("system", "You are"),
        ("human", "Q1?"),
        ("ai", "A1!"),
        ("human", "Q2?"),
        ("ai", "A2!"),
    ]);

    let result = cw.fit(messages).await.unwrap();

    // System message must be preserved.
    assert!(result
        .iter()
        .any(|m| matches!(m.message_type, MessageType::System)));
    // Most recent messages should be kept.
    assert!(result.iter().any(|m| m.content == "A2!"));
}

#[tokio::test]
async fn test_truncate_drops_oldest_first() {
    let cw: ContextWindow<OpenAIChat> =
        ContextWindow::new(25).unwrap().with_counter(char_counter());

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
    let cw: ContextWindow<OpenAIChat> =
        ContextWindow::new(20).unwrap().with_counter(char_counter());

    let messages = make_messages(&[("system", "Hello")]);

    let result = cw.fit(messages).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].content, "Hello");
}

#[tokio::test]
async fn test_truncate_system_only_over_budget() {
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(5).unwrap().with_counter(char_counter());

    let messages = make_messages(&[("system", "Very long system prompt that exceeds budget")]);

    let result = cw.fit(messages).await.unwrap();
    // System messages are always preserved even if over budget.
    assert_eq!(result.len(), 1);
}

#[tokio::test]
async fn test_summarize_replaces_old_messages() {
    let mock_llm = MockLLM::new(vec!["S.".to_string()]);

    let cw = ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
        .unwrap()
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
    let mock_llm = MockLLM::new(vec!["S.".to_string()]);

    let cw = ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
        .unwrap()
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
    let mock_llm = MockLLM::new(vec!["O.".to_string()]);

    let cw = ContextWindow::with_strategy(
        50,
        Strategy::summarize_with_prompt(mock_llm, "Please compress: {conversation}\nCompressed:"),
    )
    .unwrap()
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

    let cw: ContextWindow<MockLLM> =
        ContextWindow::with_strategy(50, Strategy::summarize(mock_llm))
            .unwrap()
            .with_counter(char_counter());

    let messages = make_messages(&[("system", "S")]);

    let result = cw.fit(messages).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].content, "S");
}

#[tokio::test]
async fn test_strategy_truncate_enum() {
    let cw = ContextWindow::with_strategy(100, Strategy::<OpenAIChat>::Truncate)
        .unwrap()
        .with_counter(char_counter());

    let messages = make_messages(&[("human", "Hello"), ("ai", "World")]);

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
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(4096).unwrap();

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
    let cw: ContextWindow<OpenAIChat> =
        ContextWindow::new(40).unwrap().with_counter(char_counter());

    let messages = make_messages(&[
        ("system", "Sys"),
        ("human", "Old"),
        ("ai", "OldA"),
        ("human", "New"),
        ("ai", "NewA"),
    ]);

    let result = cw.fit(messages).await.unwrap();

    // Verify order: system first, then conversation in order.
    let types: Vec<String> = result.iter().map(|m| m.type_str()).collect();
    // System should be first.
    assert_eq!(types[0], "system");
}

#[tokio::test]
async fn test_truncate_system_does_not_consume_budget() {
    // M7: system messages 不占预算。若计入预算(base = count(system)),
    // 预算 10 连 system 本身(4+15+2=21)都放不下 → 历史被清空;
    // 不计入后,最新一条对话(成本 5)应能留下。
    let cw: ContextWindow<OpenAIChat> =
        ContextWindow::new(10).unwrap().with_counter(char_counter());

    let messages = make_messages(&[("system", "LongSystemPrompt"), ("human", "Q")]);

    let result = cw.fit(messages).await.unwrap();
    assert!(
        result.iter().any(|m| m.content == "Q"),
        "M7: system 不占预算,最新一条对话应保留"
    );
}

#[tokio::test]
async fn test_truncate_keeps_newest_when_budget_tiny() {
    // H7: 预算小到一条对话都放不下时,至少保留最新一条,绝不静默丢光历史。
    let cw: ContextWindow<OpenAIChat> = ContextWindow::new(2).unwrap().with_counter(char_counter());

    let messages = make_messages(&[
        ("system", "S"),
        ("human", "q1"),
        ("ai", "a1"),
        ("human", "q2"),
    ]);

    let result = cw.fit(messages).await.unwrap();
    assert!(!result.is_empty());
    assert!(
        result.iter().any(|m| m.content == "q2"),
        "H7: 预算再小也应保留最新一条对话"
    );
}

#[tokio::test]
async fn test_summarize_tiny_budget_keeps_recent() {
    // H7: Summarize 无分区可容纳时退化为截断——截断保证保留最新消息,
    // 而不是 `truncate(system_messages)` 那样静默清空全部历史。
    let mock_llm = MockLLM::new(vec!["summary".to_string()]);
    let cw: ContextWindow<MockLLM> = ContextWindow::with_strategy(2, Strategy::summarize(mock_llm))
        .unwrap()
        .with_counter(char_counter());

    let messages = make_messages(&[("system", "S"), ("human", "q1"), ("ai", "a1")]);

    let result = cw.fit(messages).await.unwrap();
    assert!(
        result.iter().any(|m| m.content == "a1"),
        "H7: Summarize 兜底截断应保留最新对话,而非只留 system"
    );
}

#[tokio::test]
async fn test_summarize_fallback_to_truncate() {
    // When the summary + recent messages still exceed the budget,
    // the method falls back to truncation.
    let mock_llm = MockLLM::new(vec![
        "A very long summary that will not fit in the small budget.".to_string(),
    ]);

    // Very small budget that even the summary won't fit.
    let cw: ContextWindow<MockLLM> =
        ContextWindow::with_strategy(20, Strategy::summarize(mock_llm))
            .unwrap()
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
