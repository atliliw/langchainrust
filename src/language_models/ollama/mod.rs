// src/language_models/ollama/mod.rs
//! Ollama 本地模型支持
//!
//! Ollama 是一个本地大语言模型运行工具，支持多种开源模型。
//! Ollama 提供 OpenAI 兼容的 API，可以直接使用本框架的聊天接口。
//!
//! # 支持的模型
//! - llama3.2
//! - mistral
//! - codellama
//! - qwen2
//! - gemma
//! - 等等
//!
//! # 使用示例
//! ```rust
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

pub mod config;
pub mod chat;

pub use config::OllamaConfig;
pub use chat::OllamaChat;