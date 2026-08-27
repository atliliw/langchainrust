// lc-memory/src/base.rs
//! Memory base trait

use async_trait::async_trait;
use lc_schema::Message;
use std::collections::HashMap;

/// Memory error type
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
    /// Load error
    #[error("Failed to load memory: {0}")]
    LoadError(String),

    /// Save error
    #[error("Failed to save memory: {0}")]
    SaveError(String),

    /// Clear error
    #[error("Failed to clear memory: {0}")]
    ClearError(String),

    /// Other error
    #[error("Memory error: {0}")]
    Other(String),
}

/// Base Memory trait
///
/// The base interface for all Memory types.
#[async_trait]
pub trait BaseMemory: Send + Sync {
    /// Get memory variable names
    ///
    /// Returns all variable keys stored in memory.
    fn memory_variables(&self) -> Vec<&str>;

    /// Load memory variables
    ///
    /// # Arguments
    /// * `inputs` - Current input
    ///
    /// # Returns
    /// Memory variable dictionary
    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError>;

    /// Save context
    ///
    /// # Arguments
    /// * `inputs` - User input
    /// * `outputs` - System output
    ///
    /// # Contract
    /// Missing `input` / `output` keys return [`MemoryError::SaveError`].
    /// All built-in implementations (Buffer / Window / Summary / SummaryBuffer) behave
    /// consistently, with no silent empty-string fallback.
    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError>;

    /// Clear memory
    async fn clear(&mut self) -> Result<(), MemoryError>;
}

/// Base Chat Memory trait
///
/// Memory specifically for chat scenarios.
///
/// P0-1: lets memory types holding a `ChatMessageHistory` implement this directly
/// (internally still using the concrete history — only adds an impl, no storage change),
/// enabling generic memory code like `fn answer_with<T: BaseChatMemory>(m: &mut T)`.
pub trait BaseChatMemory: BaseMemory {
    /// Get chat message list
    fn messages(&self) -> &[Message];

    /// Add message
    fn add_message(&mut self, message: Message);

    /// Add user message
    fn add_user_message(&mut self, content: &str) {
        self.add_message(Message::human(content));
    }

    /// Add AI message
    fn add_ai_message(&mut self, content: &str) {
        self.add_message(Message::ai(content));
    }
}

/// Chat message buffer
///
/// Simple message storage for ConversationBufferMemory.
#[derive(Debug, Clone)]
pub struct ChatMessageHistory {
    /// Message list
    messages: Vec<Message>,
}

impl ChatMessageHistory {
    /// Create empty history
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Create from existing messages
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Add message
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Add user message
    pub fn add_user_message(&mut self, content: &str) {
        self.add_message(Message::human(content));
    }

    /// Add AI message
    pub fn add_ai_message(&mut self, content: &str) {
        self.add_message(Message::ai(content));
    }

    /// Add system message
    pub fn add_system_message(&mut self, content: &str) {
        self.add_message(Message::system(content));
    }

    /// Get all messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Message count
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

impl std::fmt::Display for ChatMessageHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted: String = self
            .messages
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
            .join("\n");
        write!(f, "{}", formatted)
    }
}

impl Default for ChatMessageHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert `load_memory_variables` output into a message list for LLM consumption.
///
/// Memory components produce two shapes depending on the variable:
/// - `Value::Array` (`return_messages = true`): array elements are serialized [`Message`]s,
///   deserialized one by one; a non-`Message` string element is wrapped as a `System` message;
/// - `Value::String` (`return_messages = false` / summary / vectorstore): the whole history
///   text, wrapped as a `System` message.
///
/// Reused by the `lc-sessions` bridge, `lc-chains`, and others to feed memory variables into the
/// LLM context.
pub fn memory_variables_to_messages(vars: &HashMap<String, serde_json::Value>) -> Vec<Message> {
    let mut messages = Vec::new();
    for value in vars.values() {
        match value {
            serde_json::Value::Array(items) => {
                for item in items {
                    if let Ok(msg) = serde_json::from_value::<Message>(item.clone()) {
                        messages.push(msg);
                    } else if let Some(s) = item.as_str() {
                        messages.push(Message::system(s));
                    }
                }
            }
            serde_json::Value::String(s) => messages.push(Message::system(s)),
            _ => {}
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_history() {
        let mut history = ChatMessageHistory::new();

        history.add_user_message("hello");
        history.add_ai_message("Hello! How can I help you?");
        history.add_user_message("introduce yourself");

        assert_eq!(history.len(), 3);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_chat_message_history_to_string() {
        let mut history = ChatMessageHistory::new();

        history.add_user_message("hello");
        history.add_ai_message("Hello!");

        let str = history.to_string();
        assert!(str.contains("Human: hello"));
        assert!(str.contains("AI: Hello!"));
    }

    #[test]
    fn test_chat_message_history_clear() {
        let mut history = ChatMessageHistory::new();

        history.add_user_message("test");
        assert_eq!(history.len(), 1);

        history.clear();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
    }

    /// P2-1: `memory_variables_to_messages` converts both memory-variable shapes.
    #[test]
    fn test_memory_variables_to_messages() {
        // shape one: Value::Array (return_messages = true) -> deserialize into a Message
        let msg = Message::ai("你好");
        let mut vars = HashMap::new();
        vars.insert(
            "history".to_string(),
            serde_json::json!([serde_json::to_value(&msg).unwrap()]),
        );
        let messages = memory_variables_to_messages(&vars);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "你好");

        // shape two: Value::String (return_messages = false / summary) -> wrapped as System
        let mut vars = HashMap::new();
        vars.insert(
            "history".to_string(),
            serde_json::Value::String("Human: 在吗\nAI: 在".to_string()),
        );
        let messages = memory_variables_to_messages(&vars);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_type, lc_schema::MessageType::System);
    }
}
