// lc-memory/src/buffer.rs
//! Conversation Buffer Memory
//!
//! Simple conversation buffer memory that saves all conversation history.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::base::{BaseMemory, ChatMessageHistory, MemoryError};
use lc_schema::Message;

/// Conversation Buffer Memory
///
/// Saves all conversation history in memory.
///
/// # Example
/// ```ignore
/// use lc_memory::ConversationBufferMemory;
///
/// let mut memory = ConversationBufferMemory::new();
///
/// // Save conversation
/// let inputs = HashMap::from([("input".to_string(), "Hello".to_string())]);
/// let outputs = HashMap::from([("output".to_string(), "Hi!".to_string())]);
/// memory.save_context(&inputs, &outputs).await?;
///
/// // Load memory
/// let memory_vars = memory.load_memory_variables(&HashMap::new()).await?;
/// println!("{:?}", memory_vars.get("history"));
/// ```
#[derive(Debug)]
pub struct ConversationBufferMemory {
    /// Chat history
    chat_memory: ChatMessageHistory,

    /// Input key name (default: "input")
    input_key: String,

    /// Output key name (default: "output")
    output_key: String,

    /// Memory variable name (default: "history")
    memory_key: String,

    /// Whether to return message objects
    return_messages: bool,
}

impl ConversationBufferMemory {
    /// Create new conversation buffer memory
    pub fn new() -> Self {
        Self {
            chat_memory: ChatMessageHistory::new(),
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            return_messages: false,
        }
    }

    /// Set input key name
    pub fn with_input_key(mut self, key: String) -> Self {
        self.input_key = key;
        self
    }

    /// Set output key name
    pub fn with_output_key(mut self, key: String) -> Self {
        self.output_key = key;
        self
    }

    /// Set memory variable name
    pub fn with_memory_key(mut self, key: String) -> Self {
        self.memory_key = key;
        self
    }

    /// Set whether to return message objects
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// Create from existing history
    pub fn from_chat_memory(chat_memory: ChatMessageHistory) -> Self {
        Self {
            chat_memory,
            ..Self::new()
        }
    }

    /// Get chat history
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }

    /// Get mutable chat history
    pub fn chat_memory_mut(&mut self) -> &mut ChatMessageHistory {
        &mut self.chat_memory
    }

    /// Convert history to string
    fn buffer_as_string(&self) -> String {
        self.chat_memory.to_string()
    }

    /// Convert history to message list
    fn buffer_as_messages(&self) -> Vec<Message> {
        self.chat_memory.messages().to_vec()
    }
}

impl Default for ConversationBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseMemory for ConversationBufferMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }

    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();

        if self.return_messages {
            // Return message list
            let messages: Vec<Value> = self
                .buffer_as_messages()
                .into_iter()
                .map(|msg| serde_json::to_value(&msg).unwrap_or(Value::Null))
                .collect();
            result.insert(self.memory_key.clone(), Value::Array(messages));
        } else {
            // Return string
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
    async fn test_conversation_buffer_memory() {
        let mut memory = ConversationBufferMemory::new();

        // Save conversation
        let inputs = HashMap::from([("input".to_string(), "Hello".to_string())]);
        let outputs = HashMap::from([(
            "output".to_string(),
            "Hello! How can I help you?".to_string(),
        )]);

        memory.save_context(&inputs, &outputs).await.unwrap();

        // Load memory
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();

        assert!(memory_vars.contains_key("history"));
        let history = memory_vars.get("history").unwrap();
        assert!(history.as_str().unwrap().contains("Human: Hello"));
        assert!(history.as_str().unwrap().contains("AI: Hello"));
    }

    #[tokio::test]
    async fn test_conversation_buffer_memory_multiple() {
        let mut memory = ConversationBufferMemory::new();

        // First round
        let inputs1 = HashMap::from([("input".to_string(), "My name is Zhang San".to_string())]);
        let outputs1 = HashMap::from([("output".to_string(), "Hello Zhang San!".to_string())]);
        memory.save_context(&inputs1, &outputs1).await.unwrap();

        // Second round
        let inputs2 = HashMap::from([("input".to_string(), "What is my name?".to_string())]);
        let outputs2 =
            HashMap::from([("output".to_string(), "Your name is Zhang San".to_string())]);
        memory.save_context(&inputs2, &outputs2).await.unwrap();

        // Check history
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();

        assert!(history.contains("Zhang San"));
        assert!(memory.chat_memory().len() == 4); // 2 rounds * 2 messages
    }

    #[tokio::test]
    async fn test_conversation_buffer_memory_clear() {
        let mut memory = ConversationBufferMemory::new();

        // Save conversation
        let inputs = HashMap::from([("input".to_string(), "test".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "received".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();

        assert_eq!(memory.chat_memory().len(), 2);

        // Clear
        memory.clear().await.unwrap();
        assert_eq!(memory.chat_memory().len(), 0);
    }

    #[tokio::test]
    async fn test_conversation_buffer_memory_return_messages() {
        let mut memory = ConversationBufferMemory::new().with_return_messages(true);

        let inputs = HashMap::from([("input".to_string(), "Hello".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "Hello!".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();

        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap();

        // Should return message array
        assert!(history.is_array());
        let messages = history.as_array().unwrap();
        assert_eq!(messages.len(), 2);
    }
}
