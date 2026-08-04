// src/core/structured_output/mod.rs
//! Structured output utilities for extracting typed data from LLM responses.
//!
//! This module provides a provider-agnostic `with_structured_output` function
//! that works with any `BaseChatModel` implementation. It injects a JSON schema
//! into the prompt, calls the LLM, and parses the JSON response into the
//! target type `T`.
//!
//! It also provides streaming support via `stream_structured_output`, which
//! returns a stream of partial `T` values as the model generates tokens,
//! using `PartialJsonParser` to incrementally parse incomplete JSON.
//!
//! # Strategy
//!
//! The default (generic) strategy uses **prompt injection**: the JSON schema
//! and format instructions are embedded in the system prompt, and the
//! `JsonOutputParser` is used to extract JSON from the response.
//!
//! Provider-specific implementations (OpenAI function calling, Ollama JSON mode)
//! are available on the concrete types directly (e.g., `OpenAIChat::with_structured_output`).
//!
//! # Example
//!
//! ```ignore
//! use serde::{Deserialize, Serialize};
//! use langchainrust::core::structured_output::{with_structured_output, StructuredOutputError};
//! use langchainrust::{OpenAIChat, OpenAIConfig, Message};
//!
//! #[derive(Debug, Deserialize, Serialize)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! let llm = OpenAIChat::new(OpenAIConfig::default());
//! let schema = serde_json::json!({
//!     "type": "object",
//!     "properties": {
//!         "name": {"type": "string"},
//!         "age": {"type": "integer"}
//!     },
//!     "required": ["name", "age"]
//! });
//!
//! let person: Person = with_structured_output(&llm, schema, "Tell me about Alice who is 30").await?;
//! ```

pub mod extract;
pub mod parser;
pub mod streaming;

pub use extract::{with_structured_output, StructuredOutputError, StructuredOutputExt};
pub use parser::{PartialJsonError, PartialJsonParser};
pub use streaming::{stream_structured_output, StreamingStructuredOutputExt};

#[cfg(test)]
mod tests;
