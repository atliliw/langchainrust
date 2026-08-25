// lc-memory/src/window.rs
//! Conversation Buffer Window Memory
//!
//! Conversation memory with window, keeping only the last k rounds.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseChatMemory, BaseMemory, ChatMessageHistory, MemoryError};
use lc_schema::Message;

/// Conversation Buffer Window Memory
///
/// Keeps only the last k rounds of conversation to avoid overly long context.
///
/// # Example
/// ```ignore
/// use lc_memory::ConversationBufferWindowMemory;
///
/// // Keep only the last 2 rounds
/// let mut memory = ConversationBufferWindowMemory::new(2);
/// ```
#[derive(Debug)]
pub struct ConversationBufferWindowMemory {
    /// Chat history
    chat_memory: ChatMessageHistory,

    /// Window size (keep last k rounds, default 5)
    k: usize,

    /// Input key name
    input_key: String,

    /// Output key name
    output_key: String,

    /// Memory variable name
    memory_key: String,

    /// Whether to return message objects
    return_messages: bool,
}

impl ConversationBufferWindowMemory {
    /// Create new window memory
    ///
    /// # Arguments
    /// * `k` - Keep last k rounds (each round includes user message and AI message)
    pub fn new(k: usize) -> Self {
        Self {
            chat_memory: ChatMessageHistory::new(),
            k,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            return_messages: false,
        }
    }

    /// Set input key name
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    /// Set output key name
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set memory variable name
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set whether to return message objects
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// Get chat history
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }

    /// Get window size
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get messages within the window
    ///
    /// Only keeps the last k rounds (2*k messages)
    fn get_window_messages(&self) -> Vec<Message> {
        let messages = self.chat_memory.messages();
        let total = messages.len();

        // Each round includes 2 messages (user + AI)
        let max_messages = self.k * 2;

        if total <= max_messages {
            messages.to_vec()
        } else {
            messages[total - max_messages..].to_vec()
        }
    }

    /// Convert to string
    fn buffer_as_string(&self) -> String {
        self.get_window_messages()
            .iter()
            .map(|msg| {
                let role = match msg.message_type {
                    lc_schema::MessageType::Human => "Human",
                    lc_schema::MessageType::AI => "AI",
                    lc_schema::MessageType::System => "System",
                    lc_schema::MessageType::Tool { .. } => "Tool",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for ConversationBufferWindowMemory {
    fn default() -> Self {
        Self::new(5)
    }
}

/// P0-1: `ConversationBufferWindowMemory` 实现 `BaseChatMemory`。
impl BaseChatMemory for ConversationBufferWindowMemory {
    fn messages(&self) -> &[Message] {
        self.chat_memory.messages()
    }

    fn add_message(&mut self, message: Message) {
        self.chat_memory.add_message(message);
    }
}

#[async_trait]
impl BaseMemory for ConversationBufferWindowMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();

        if self.return_messages {
            let messages: Vec<Value> = self
                .get_window_messages()
                .into_iter()
                .map(|msg| serde_json::to_value(&msg).unwrap_or(Value::Null))
                .collect();
            result.insert(self.memory_key.clone(), Value::Array(messages));
        } else {
            result.insert(
                self.memory_key.clone(),
                Value::String(self.buffer_as_string()),
            );
        }

        Ok(result)
    }

    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        // M76: Return error when required keys are missing instead of silently skipping
        let input = inputs.get(&self.input_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing input key '{}'", self.input_key))
        })?;
        self.chat_memory.add_user_message(input);

        let output = outputs.get(&self.output_key).ok_or_else(|| {
            MemoryError::SaveError(format!("Missing output key '{}'", self.output_key))
        })?;
        self.chat_memory.add_ai_message(output);

        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.chat_memory.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_window_memory() {
        let mut memory = ConversationBufferWindowMemory::new(2);

        // Add 3 rounds (6 messages total)
        for i in 1..=3 {
            let inputs = HashMap::from([("input".to_string(), format!("Question{}", i))]);
            let outputs = HashMap::from([("output".to_string(), format!("Answer{}", i))]);
            memory.save_context(&inputs, &outputs).await.unwrap();
        }

        // Full history has 6 messages
        assert_eq!(memory.chat_memory().len(), 6);

        // But only returns last 2 rounds (4 messages)
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();

        // Should contain Question2, Answer2, Question3, Answer3
        assert!(!history.contains("Question1"));
        assert!(!history.contains("Answer1"));
        assert!(history.contains("Question2"));
        assert!(history.contains("Answer3"));
    }

    #[tokio::test]
    async fn test_window_memory_smaller_than_k() {
        let mut memory = ConversationBufferWindowMemory::new(5);

        // Only add 2 rounds
        for i in 1..=2 {
            let inputs = HashMap::from([("input".to_string(), format!("Question{}", i))]);
            let outputs = HashMap::from([("output".to_string(), format!("Answer{}", i))]);
            memory.save_context(&inputs, &outputs).await.unwrap();
        }

        // Should return all 4 messages
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();

        assert!(history.contains("Question1"));
        assert!(history.contains("Question2"));
    }

    #[tokio::test]
    async fn test_window_memory_clear() {
        let mut memory = ConversationBufferWindowMemory::new(2);

        let inputs = HashMap::from([("input".to_string(), "test".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "received".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();

        assert_eq!(memory.chat_memory().len(), 2);

        memory.clear().await.unwrap();
        assert_eq!(memory.chat_memory().len(), 0);
    }
}
