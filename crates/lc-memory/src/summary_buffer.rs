// lc-memory/src/summary_buffer.rs
//! Conversation Summary Buffer Memory
//!
//! Combines summary and full conversation, balancing token consumption and conversation quality.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, ChatMessageHistory, MemoryError};
use lc_core::language_models::BaseChatModel;
use lc_core::language_models::LLMResult;
use lc_core::runnables::Runnable;
use lc_prompts::PromptTemplate;
use lc_schema::Message;

const DEFAULT_SUMMARY_PROMPT: &str =
    "Progressively summarize the conversation, adding new content to the previous summary.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:";

/// Conversation Summary Buffer Memory
///
/// Combines summary and full conversation:
/// - Keeps the last k rounds of full conversation (ensuring fluency)
/// - Summarizes older conversations (saving tokens)
///
/// # Example
/// ```ignore
/// use lc_memory::ConversationSummaryBufferMemory;
/// use lc_providers::OpenAIChat;
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationSummaryBufferMemory::new(llm, 5); // Keep last 5 rounds
///
/// // After 20 rounds:
/// // - First 15 rounds -> summary
/// // - Last 5 rounds -> full conversation
/// ```
pub struct ConversationSummaryBufferMemory<M: BaseChatModel> {
    llm: M,

    /// M67: Removed `Mutex<String>` - &mut self already guarantees exclusive access
    buffer: String,
    chat_memory: ChatMessageHistory,

    max_token_limit: usize,

    input_key: String,
    output_key: String,
    memory_key: String,

    summary_prompt: String,
    return_messages: bool,
}

impl<M: BaseChatModel> ConversationSummaryBufferMemory<M> {
    pub fn new(llm: M, max_token_limit: usize) -> Self {
        Self {
            llm,
            buffer: String::new(),
            chat_memory: ChatMessageHistory::new(),
            max_token_limit,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            return_messages: false,
        }
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }

    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }

    pub fn chat_memory_mut(&mut self) -> &mut ChatMessageHistory {
        &mut self.chat_memory
    }

    pub fn max_token_limit(&self) -> usize {
        self.max_token_limit
    }

    pub async fn buffer(&self) -> String {
        self.buffer.clone()
    }

    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }

    fn prune_messages(&self, messages: &[Message]) -> Vec<Message> {
        let total_tokens = messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum::<usize>();

        if total_tokens <= self.max_token_limit {
            return messages.to_vec();
        }

        let mut kept_messages = Vec::new();
        let mut current_tokens = 0;

        for msg in messages.iter().rev() {
            let msg_tokens = Self::estimate_tokens(&msg.content);
            if current_tokens + msg_tokens <= self.max_token_limit {
                kept_messages.push(msg.clone());
                current_tokens += msg_tokens;
            } else {
                break;
            }
        }

        kept_messages.reverse();
        kept_messages
    }

    async fn predict_new_summary(&self, new_lines: &str) -> Result<String, MemoryError> {
        let buffer = self.buffer.clone();

        let prompt = {
            let template = PromptTemplate::new(&self.summary_prompt);
            let mut vars: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
            vars.insert("summary", buffer.as_str());
            vars.insert("new_lines", new_lines);
            template
                .format(&vars)
                .unwrap_or_else(|_| self.summary_prompt.clone())
        };

        let messages = vec![Message::human(&prompt)];

        let result =
            self.llm.invoke(messages, None).await.map_err(|e| {
                MemoryError::SaveError(format!("LLM summary generation failed: {}", e))
            })?;

        Ok(result.content)
    }
}

#[async_trait]
impl<M: BaseChatModel + Send + Sync + 'static> BaseMemory for ConversationSummaryBufferMemory<M>
where
    <M as Runnable<Vec<Message>, LLMResult>>::Error: std::fmt::Display,
{
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();

        let buffer = self.buffer.clone();
        let messages = self.chat_memory.messages();
        let pruned = self.prune_messages(messages);

        if self.return_messages {
            let mut all_messages = Vec::new();

            if !buffer.is_empty() {
                all_messages.push(Message::system(&buffer));
            }

            all_messages.extend(pruned);

            let messages_value: Vec<Value> = all_messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect();

            result.insert(self.memory_key.clone(), Value::Array(messages_value));
        } else {
            let mut history = String::new();

            if !buffer.is_empty() {
                history.push_str(&format!("Summary: {}\n\n", buffer));
            }

            for msg in &pruned {
                let role = match msg.message_type {
                    lc_schema::MessageType::Human => "Human",
                    lc_schema::MessageType::AI => "AI",
                    lc_schema::MessageType::System => "System",
                    lc_schema::MessageType::Tool { .. } => "Tool",
                };
                history.push_str(&format!("{}: {}\n", role, msg.content));
            }

            result.insert(self.memory_key.clone(), Value::String(history));
        }

        Ok(result)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let empty = String::new();
        let input = inputs.get(&self.input_key).unwrap_or(&empty);
        let output = outputs.get(&self.output_key).unwrap_or(&empty);

        self.chat_memory.add_user_message(input);
        self.chat_memory.add_ai_message(output);

        let messages = self.chat_memory.messages();
        let total_tokens = messages
            .iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum::<usize>();

        if total_tokens > self.max_token_limit {
            let pruned = self.prune_messages(messages);

            let pruned_count = pruned.len();

            if messages.len() > pruned_count {
                let messages_to_summarize: Vec<&Message> = messages
                    .iter()
                    .take(messages.len() - pruned_count)
                    .collect();

                if !messages_to_summarize.is_empty() {
                    let new_lines: String = messages_to_summarize
                        .iter()
                        .map(|m| {
                            let role = match m.message_type {
                                lc_schema::MessageType::Human => "Human",
                                lc_schema::MessageType::AI => "AI",
                                lc_schema::MessageType::System => "System",
                                lc_schema::MessageType::Tool { .. } => "Tool",
                            };
                            format!("{}: {}", role, m.content)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    let new_summary = self.predict_new_summary(&new_lines).await?;

                    self.buffer = new_summary;
                }

                self.chat_memory.clear();
                for msg in pruned {
                    if matches!(msg.message_type, lc_schema::MessageType::Human) {
                        self.chat_memory.add_user_message(&msg.content);
                    } else if matches!(msg.message_type, lc_schema::MessageType::AI) {
                        self.chat_memory.add_ai_message(&msg.content);
                    } else if matches!(msg.message_type, lc_schema::MessageType::System) {
                        // H28: Preserve System messages during pruning
                        self.chat_memory.add_system_message(&msg.content);
                    }
                }
            }
        }

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.buffer = String::new();
        self.chat_memory.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_providers::{OpenAIChat, OpenAIConfig};

    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig::default()
    }

    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        assert_eq!(memory.memory_variables(), vec!["history"]);
        assert_eq!(memory.max_token_limit(), 1000);
    }

    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 500)
                .with_input_key("question")
                .with_output_key("answer")
                .with_memory_key("context")
                .with_return_messages(true);

        assert_eq!(memory.input_key, "question");
        assert_eq!(memory.output_key, "answer");
        assert_eq!(memory.memory_key, "context");
        assert!(memory.return_messages);
    }

    #[test]
    fn test_estimate_tokens() {
        let text1 = "Hello";
        let text2 = "Hello World";
        let text3 = "This is some Chinese text";

        assert!(ConversationSummaryBufferMemory::<OpenAIChat>::estimate_tokens(text1) > 0);
        assert!(
            ConversationSummaryBufferMemory::<OpenAIChat>::estimate_tokens(text2)
                > ConversationSummaryBufferMemory::<OpenAIChat>::estimate_tokens(text1)
        );
        assert!(ConversationSummaryBufferMemory::<OpenAIChat>::estimate_tokens(text3) > 0);
    }

    #[test]
    fn test_prune_messages_within_limit() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let messages = vec![
            Message::human("Short message 1"),
            Message::ai("Short reply 1"),
        ];

        let pruned = memory.prune_messages(&messages);

        assert_eq!(pruned.len(), 2);
    }

    #[tokio::test]
    async fn test_buffer_initial_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let buffer = memory.buffer().await;
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn test_load_memory_variables_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();

        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_clear() {
        let llm = OpenAIChat::new(create_test_config());
        let mut memory: ConversationSummaryBufferMemory<OpenAIChat> =
            ConversationSummaryBufferMemory::new(llm, 1000);

        memory.chat_memory.add_user_message("test");
        memory.chat_memory.add_ai_message("reply");

        memory.buffer = "Test summary".to_string();

        memory.clear().await.unwrap();

        assert!(memory.buffer().await.is_empty());
        assert_eq!(memory.chat_memory().len(), 0);
    }
}
