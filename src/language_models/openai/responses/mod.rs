// src/language_models/openai/responses/mod.rs
//! OpenAI Responses API model implementation.
//!
//! This module implements the `/v1/responses` endpoint, which provides
//! access to OpenAI's built-in tools (web search, file search, code
//! interpreter, computer use) alongside standard chat capabilities.

pub mod model;
#[cfg(test)]
mod tests;
pub mod types;

// Re-export all public items so that external `use` paths remain unchanged.
pub use model::ResponsesModel;
pub use types::{BuiltinTool, ResponsesConfig, ResponsesError};
