// src/language_models/mod.rs
//! 语言模型实现

pub mod openai;
pub mod ollama;

pub use openai::{OpenAIChat, OpenAIConfig};
pub use ollama::{OllamaChat, OllamaConfig};