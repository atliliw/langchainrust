use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    #[serde(rename = "system")]
    System(SystemMessage),
    #[serde(rename = "user")]
    Human(HumanMessage),
    #[serde(rename = "assistant")]
    AIMessage(AIMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanMessage {
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIMessage {
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Message::System(SystemMessage {
            content: content.into(),
        })
    }

    pub fn human(content: impl Into<String>) -> Self {
        Message::Human(HumanMessage {
            content: content.into(),
        })
    }

    pub fn ai(content: impl Into<String>) -> Self {
        Message::AIMessage(AIMessage {
            content: content.into(),
        })
    }

    pub fn content(&self) -> &str {
        match self {
            Message::System(msg) => &msg.content,
            Message::Human(msg) => &msg.content,
            Message::AIMessage(msg) => &msg.content,
        }
    }

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
