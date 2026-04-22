// src/core/language_models/mod.rs
//! Language model base traits.

mod base;
mod chat;

pub use base::BaseLanguageModel;
pub use chat::{BaseChatModel, LLMResult, TokenUsage};