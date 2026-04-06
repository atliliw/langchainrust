// src/language_models/openai/mod.rs
//! OpenAI 语言模型实现
//!
//! 参考 Python 版本: langchain/libs/partners/openai/langchain_openai/

mod config;
mod chat;
mod sse;

pub use config::OpenAIConfig;
pub use chat::OpenAIChat;
pub use sse::{SSEParser, SSEEvent};