use serde::{Deserialize, Serialize};

/// A message within an A2A task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AMessage {
    /// Role of the message sender (e.g. "user", "agent").
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

impl A2AMessage {
    /// Create a new message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an agent message.
    pub fn agent(content: impl Into<String>) -> Self {
        Self::new("agent", content)
    }
}
