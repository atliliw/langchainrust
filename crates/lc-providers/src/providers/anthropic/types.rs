// src/language_models/providers/anthropic/types.rs
//! Public and private types for the Anthropic API request/response format.

use lc_core::tools::ToolCall;
use serde::{Deserialize, Serialize};

/// A token emitted during streaming, distinguishing between thinking and text.
#[derive(Debug, Clone, PartialEq)]
pub enum AnthropicStreamToken {
    /// A text content token (the final answer).
    Text(String),
    /// A thinking content token (extended reasoning).
    Thinking(String),
    /// A complete tool call, emitted when its `content_block_stop` arrives
    /// (0.22.0 C2 fix: tool calls are no longer silently dropped on the
    /// streaming path — `input_json_delta` fragments are accumulated per
    /// content-block index and flushed on stop).
    ToolCall(ToolCall),
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
///
/// B1: we also capture the prompt-cache breakdown Anthropic reports separately —
/// `cache_creation_input_tokens` (tokens written into the cache by this call) and
/// `cache_read_input_tokens` (tokens served from cache). Both are `#[serde(default)]`
/// so older payloads (and providers/proxies that omit them) still deserialize.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
    /// Tokens written into the prompt cache by this call (B1 statistic).
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: usize,
    /// Prompt tokens served from cache instead of re-processed (B1 statistic).
    #[serde(default)]
    pub(crate) cache_read_input_tokens: usize,
}

impl AnthropicUsage {
    /// Tokens written into the cache by this call, if reported (else 0).
    pub fn cache_creation_tokens(&self) -> usize {
        self.cache_creation_input_tokens
    }

    /// Prompt tokens served from cache (cache hits), if reported (else 0).
    pub fn cache_read_tokens(&self) -> usize {
        self.cache_read_input_tokens
    }

    /// Prompt tokens that actually had to be recomputed: input minus cache reads.
    pub fn cache_miss_tokens(&self) -> usize {
        self.input_tokens.saturating_sub(self.cache_read_input_tokens)
    }

    /// `(creation, read)` cache breakdown for observability.
    pub fn cache_breakdown(&self) -> (usize, usize) {
        (self.cache_creation_input_tokens, self.cache_read_input_tokens)
    }
}

#[derive(Deserialize)]
pub(crate) struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    pub(crate) type_field: String,
    /// Content-block index carried by `content_block_*` events (C2: keys
    /// `input_json_delta` fragments to the right tool call).
    #[serde(default)]
    pub(crate) index: Option<usize>,
    /// The starting content block of a `content_block_start` event.
    pub(crate) content_block: Option<AnthropicStreamBlock>,
    pub(crate) delta: Option<AnthropicDelta>,
    /// Token usage from the `message_delta` event (stream end).
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
}

/// The `content_block` payload of a `content_block_start` stream event.
#[derive(Deserialize)]
pub(crate) struct AnthropicStreamBlock {
    #[serde(rename = "type")]
    pub(crate) type_field: String,
    /// Tool call ID (present when type == "tool_use").
    #[serde(default)]
    pub(crate) id: String,
    /// Tool name (present when type == "tool_use").
    #[serde(default)]
    pub(crate) name: String,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicDelta {
    #[serde(rename = "type")]
    pub(crate) type_field: String,
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) thinking: String,
    /// Incremental tool-call JSON from `input_json_delta` fragments (C2).
    #[serde(default)]
    pub(crate) partial_json: String,
}
