// src/language_models/providers/anthropic/mod.rs
//! Anthropic Claude API implementation (native API format).
//!
//! Supports extended thinking via the `ThinkingConfig` / `with_thinking()` API.
//! When thinking is enabled, the request includes a `thinking` parameter and
//! the response may contain both "thinking" and "text" content blocks.

pub mod chat;
pub mod config;
pub mod error;
pub mod impls;
#[cfg(test)]
pub mod tests;
pub mod types;

pub use chat::{AnthropicChat, AnthropicStructuredOutputMethod};
pub use config::{
    AnthropicConfig, ThinkingConfig, ThinkingType, ANTHROPIC_BASE_URL, CLAUDE_MODELS,
};
pub use error::AnthropicError;
pub use types::{AnthropicContentBlock, AnthropicMessageContent, AnthropicStreamToken};
