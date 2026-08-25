//! Message data structures for chat models.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use lc_shared::tools::ToolCall;

use super::audio::AudioContent;
use super::file::FileContent;
use super::image::ImageContent;

/// Message type classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    /// 系统消息
    System,
    /// 人类（用户）消息
    Human,
    /// AI（助手）消息
    AI,
    /// 工具结果消息，携带对应的 tool_call_id
    Tool {
        /// 关联的工具调用 ID
        tool_call_id: String,
    },
}

/// Complete message structure for chat interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// 消息文本内容
    pub content: String,

    /// 图片内容(多模态 vision)
    #[serde(default)]
    pub images: Vec<ImageContent>,

    /// 音频内容(多模态 audio)
    #[serde(default)]
    pub audio: Vec<AudioContent>,

    /// 文件内容(多模态 document)
    #[serde(default)]
    pub files: Vec<FileContent>,

    /// 消息类型
    #[serde(rename = "type")]
    pub message_type: MessageType,

    /// 消息名称（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// 附加关键字参数
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_kwargs: HashMap<String, Value>,

    /// 消息 ID（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// 工具调用列表（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            files: Vec::new(),
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
            images: Vec::new(),
            audio: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with an image (vision).
    pub fn human_with_image(content: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: vec![ImageContent::from_url(image_url)],
            audio: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with multiple images.
    pub fn human_with_images(content: impl Into<String>, images: Vec<ImageContent>) -> Self {
        Self {
            content: content.into(),
            images,
            audio: Vec::new(),
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with audio content.
    pub fn human_with_audio(content: impl Into<String>, audio: AudioContent) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: vec![audio],
            files: Vec::new(),
            message_type: MessageType::Human,
            name: None,
            additional_kwargs: HashMap::new(),
            id: None,
            tool_calls: None,
        }
    }

    /// Creates a human message with file content.
    pub fn human_with_file(content: impl Into<String>, file: FileContent) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            audio: Vec::new(),
            files: vec![file],
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
            images: Vec::new(),
            audio: Vec::new(),
            files: Vec::new(),
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
            images: Vec::new(),
            audio: Vec::new(),
            files: Vec::new(),
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
            images: Vec::new(),
            audio: Vec::new(),
            files: Vec::new(),
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

    /// Adds an image to the message (vision).
    pub fn with_image(mut self, image: ImageContent) -> Self {
        self.images.push(image);
        self
    }

    /// Adds audio content to the message.
    pub fn with_audio(mut self, audio: AudioContent) -> Self {
        self.audio.push(audio);
        self
    }

    /// Adds file content to the message.
    pub fn with_file(mut self, file: FileContent) -> Self {
        self.files.push(file);
        self
    }

    /// Returns whether the message has images.
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }

    /// Returns whether the message has audio content.
    pub fn has_audio(&self) -> bool {
        !self.audio.is_empty()
    }

    /// Returns whether the message has file content.
    pub fn has_files(&self) -> bool {
        !self.files.is_empty()
    }

    /// Returns whether the message has any multimodal content (images, audio, or files).
    pub fn is_multimodal(&self) -> bool {
        self.has_images() || self.has_audio() || self.has_files()
    }

    /// Returns the message type as a string.
    ///
    /// Tool messages include their `tool_call_id` (e.g. `"tool:call_123"`) so
    /// the type string is unambiguous about which tool result the message holds.
    pub fn type_str(&self) -> String {
        match &self.message_type {
            MessageType::System => "system".to_string(),
            MessageType::Human => "human".to_string(),
            MessageType::AI => "ai".to_string(),
            MessageType::Tool { tool_call_id } => format!("tool:{tool_call_id}"),
        }
    }

    /// Returns whether the message has tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_deref().is_some_and(|t| !t.is_empty())
    }

    /// Returns the tool calls if present.
    pub fn get_tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_with_image() {
        let msg = Message::human_with_image("描述这张图", "https://example.com/img.jpg");
        assert_eq!(msg.content, "描述这张图");
        assert_eq!(msg.images.len(), 1);
        assert_eq!(msg.images[0].url, "https://example.com/img.jpg");
        assert!(msg.has_images());
    }

    #[test]
    fn test_human_no_images_by_default() {
        let msg = Message::human("纯文本");
        assert!(msg.images.is_empty());
        assert!(!msg.has_images());
    }

    #[test]
    fn test_with_image_builder() {
        let msg = Message::human("看图")
            .with_image(ImageContent::from_url("https://example.com/a.png"))
            .with_image(ImageContent::from_base64("abc"));
        assert_eq!(msg.images.len(), 2);
    }

    #[test]
    fn test_message_deserialize_without_images_field() {
        // 旧格式(无 images 字段)应能反序列化(#[serde(default)])
        let json = r#"{"content":"hi","type":"human"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content, "hi");
        assert!(msg.images.is_empty());
    }

    #[test]
    fn test_human_with_images_multiple() {
        let msg = Message::human_with_images(
            "多图",
            vec![
                ImageContent::from_url("https://example.com/1.jpg"),
                ImageContent::from_url("https://example.com/2.jpg"),
            ],
        );
        assert_eq!(msg.images.len(), 2);
    }

    #[test]
    fn test_system_ai_no_images() {
        assert!(Message::system("s").images.is_empty());
        assert!(Message::ai("a").images.is_empty());
        assert!(Message::tool("id", "c").images.is_empty());
    }

    #[test]
    fn test_type_str_includes_tool_call_id() {
        assert_eq!(Message::system("s").type_str(), "system");
        assert_eq!(Message::human("h").type_str(), "human");
        assert_eq!(Message::ai("a").type_str(), "ai");
        assert_eq!(
            Message::tool("call_123", "result").type_str(),
            "tool:call_123"
        );
    }

    #[test]
    fn test_has_tool_calls_empty_and_present() {
        let with_calls = Message::ai_with_tool_calls(
            "call tool",
            vec![ToolCall::builder("call_1")
                .name("weather")
                .arguments(r#"{"city":"beijing"}"#)
                .build()],
        );
        assert!(with_calls.has_tool_calls());
        assert_eq!(with_calls.get_tool_calls().unwrap().len(), 1);

        // No panic on None or on an empty vec
        assert!(!Message::ai("plain").has_tool_calls());
        let empty = Message::ai_with_tool_calls("no calls", vec![]);
        assert!(!empty.has_tool_calls());
    }
}
