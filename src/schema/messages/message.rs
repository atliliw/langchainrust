// src/schema/messages/message.rs

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::core::tools::ToolCall;

/// 消息类型分类
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    System,
    Human,
    AI,
    Tool { tool_call_id: String },
}

/// 完整的消息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub content: String,

    #[serde(rename = "type")]
    pub message_type: MessageType,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub additional_kwargs: HashMap<String, Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::System,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    pub fn human(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    pub fn ai(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::AI,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    pub fn ai_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::AI,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: Some(tool_calls),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            message_type: MessageType::Tool {
                tool_call_id: tool_call_id.into(),
            },
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_additional_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.additional_kwargs.insert(key.into(), value);
        self
    }

    pub fn type_str(&self) -> &str {
        match &self.message_type {
            MessageType::System => "system",
            MessageType::Human => "human",
            MessageType::AI => "ai",
            MessageType::Tool { .. } => "tool",
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.is_some() && !self.tool_calls.as_ref().unwrap().is_empty()
    }

    pub fn get_tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}
