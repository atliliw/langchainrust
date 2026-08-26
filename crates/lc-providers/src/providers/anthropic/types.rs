// src/language_models/providers/anthropic/types.rs
//! Public and private types for the Anthropic API request/response format.

use serde::{Deserialize, Serialize};

/// A token emitted during streaming, distinguishing between thinking and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicStreamToken {
    /// A text content token (the final answer).
    Text(String),
    /// A thinking content token (extended reasoning).
    Thinking(String),
    /// Token usage reported at the end of the stream (`message_delta` event).
    /// Carried separately so the streaming path can observe usage without a
    /// separate non-streaming `chat` call.
    Usage(AnthropicUsage),
}

/// Content for an Anthropic message, supporting both simple text and structured content arrays.
#[derive(Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content array with multiple blocks.
    Blocks(Vec<AnthropicContentBlock>),
}

/// A single content block in an Anthropic message.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Text content block.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Image content block (base64-encoded).
    #[serde(rename = "image")]
    Image {
        /// The image source.
        source: AnthropicImageSource,
    },
    /// Tool use content block (from assistant).
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Tool call ID.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments as JSON.
        input: serde_json::Value,
    },
    /// Tool result content block (from user, responding to tool use).
    #[serde(rename = "tool_result")]
    ToolResult {
        /// The tool call ID this result responds to.
        tool_use_id: String,
        /// The result content.
        content: String,
    },
}

/// Image source for Anthropic's image content block.
///
/// Anthropic only supports base64-encoded images (no URL-based images).
#[derive(Serialize, Clone, Debug)]
pub struct AnthropicImageSource {
    /// Source type (e.g. "base64").
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type of the image (e.g. "image/png").
    pub media_type: String,
    /// Base64-encoded image data.
    pub data: String,
}

// --- Private API types ---

#[derive(Serialize, Clone)]
pub(crate) struct AnthropicMessage {
    pub(crate) role: String,
    pub(crate) content: AnthropicMessageContent,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub(crate) struct AnthropicResponse {
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) content: Vec<AnthropicContent>,
    pub(crate) usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub(crate) struct AnthropicContent {
    #[serde(rename = "type")]
    pub(crate) content_type: String,
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) thinking: String,
    /// Tool use ID (present when content_type == "tool_use")
    #[serde(default)]
    pub(crate) id: Option<String>,
    /// Tool name (present when content_type == "tool_use")
    #[serde(default)]
    pub(crate) name: Option<String>,
    /// Tool input (present when content_type == "tool_use")
    #[serde(default)]
    pub(crate) input: Option<serde_json::Value>,
}

/// Token usage reported in the Anthropic `message_delta` stream event.
///
/// Exposed (with crate-private fields) because it appears as the payload of
/// the public [`AnthropicStreamToken::Usage`] variant.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    pub(crate) type_field: String,
    pub(crate) delta: Option<AnthropicDelta>,
    /// Token usage from the `message_delta` event (stream end).
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicDelta {
    #[serde(rename = "type")]
    pub(crate) type_field: String,
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) thinking: String,
}
