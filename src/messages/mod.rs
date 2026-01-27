use serde::{Deserialize, Serialize};

/// A single chat message with a specific role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    Human(HumanMessage),
    #[serde(rename = "assistant")]
    AIMessage(AIMessage),
}

/// System-level message, usually used for instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
}

/// Human/user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanMessage {
    pub content: String,
}

/// Assistant/AI message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMessage {
    pub content: String,
}

impl Message {
    /// Create a system message from string-like content.
    pub fn system(content: impl Into<String>) -> Self {
        Message::System(SystemMessage {
            content: content.into(),
        })
    }

    /// Create a human/user message from string-like content.
    pub fn human(content: impl Into<String>) -> Self {
        Message::Human(HumanMessage {
            content: content.into(),
        })
    }

    /// Create an assistant/AI message from string-like content.
    pub fn ai(content: impl Into<String>) -> Self {
        Message::AIMessage(AIMessage {
            content: content.into(),
        })
    }

    /// Return the message content as a string slice.
    pub fn content(&self) -> &str {
        match self {
            Message::System(msg) => &msg.content,
            Message::Human(msg) => &msg.content,
            Message::AIMessage(msg) => &msg.content,
        }
    }

    /// Return the OpenAI-style role string for this message.
    pub fn role(&self) -> &str {
        match self {
            Message::System(_) => "system",
            Message::Human(_) => "user",
            Message::AIMessage(_) => "assistant",
        }
    }
}

impl From<SystemMessage> for Message {
    fn from(msg: SystemMessage) -> Self {
        Message::System(msg)
    }
}

impl From<HumanMessage> for Message {
    fn from(msg: HumanMessage) -> Self {
        Message::Human(msg)
    }
}

impl From<AIMessage> for Message {
    fn from(msg: AIMessage) -> Self {
        Message::AIMessage(msg)
    }
}

pub trait IntoApiMessage {
    fn role(&self) -> String;
    fn content(&self) -> String;
}
