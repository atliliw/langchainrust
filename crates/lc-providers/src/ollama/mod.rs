// src/language_models/ollama/mod.rs
//! Ollama local model support
//!
//! Ollama is a local large-language-model runner that supports many open-source models.
//! Ollama exposes an OpenAI-compatible API, so it can use this framework's chat interface directly.
//!
//! # Supported Models
//! - llama3.2
//! - mistral
//! - codellama
//! - qwen2
//! - gemma
//! - and more
//!
//! # Usage Example
//! ```rust,ignore
//! use langchainrust::{OllamaChat, OllamaConfig, BaseChatModel};
//! use langchainrust::schema::Message;
//!
//! let llm = OllamaChat::new("llama3.2");
//! let messages = vec![
//!     Message::system("你是一个助手"),
//!     Message::human("你好"),
//! ];
//! let response = llm.chat(messages, None).await?;
//! ```

pub mod chat;
pub mod config;

pub use chat::OllamaChat;
pub use chat::OllamaError;
pub use config::OllamaConfig;
