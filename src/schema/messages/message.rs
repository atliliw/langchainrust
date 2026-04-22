// src/schema/messages/message.rs
//! Message data structures for chat models.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::core::tools::ToolCall;

/// Message type classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    System,
    Human,
    AI,
    Tool { tool_call_id: String },
}

/// Complete message structure for chat interactions.
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
    /// Creates a system message.
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

    /// Creates a human (user) message.
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

    /// Creates an AI (assistant) message.
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

    /// Creates an AI message with tool calls.
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

    /// Creates a tool result message.
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

    /// Sets the message name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the message ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Adds an additional keyword argument.
    pub fn with_additional_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.additional_kwargs.insert(key.into(), value);
        self
    }

    /// Returns the message type as a string.
    pub fn type_str(&self) -> &str {
        match &self.message_type {
            MessageType::System => "system",
            MessageType::Human => "human",
            MessageType::AI => "ai",
            MessageType::Tool { .. } => "tool",
        }
    }

    /// Returns whether the message has tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.is_some() && !self.tool_calls.as_ref().unwrap().is_empty()
    }

    /// Returns the tool calls if present.
    pub fn get_tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}
