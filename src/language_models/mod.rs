// src/language_models/mod.rs
//! LLM integrations for various providers.
//!
//! This module provides unified chat model interfaces for:
//! - OpenAI (GPT-4, GPT-3.5)
//! - Ollama (local LLMs like Llama, Mistral)
//! - DeepSeek (cost-effective Chinese LLM)
//! - Moonshot (long-context Kimi)
//! - Qwen (Alibaba Cloud)
//! - Zhipu (ChatGLM)
//! - Anthropic (Claude)

/// OpenAI API integration.
pub mod openai;
/// Ollama local LLM integration.
pub mod ollama;
/// Third-party provider integrations.
pub mod providers;

pub use openai::{OpenAIChat, OpenAIConfig};
pub use ollama::{OllamaChat, OllamaConfig};
pub use providers::{
    DeepSeekChat, DeepSeekConfig,
    MoonshotChat, MoonshotConfig,
    ZhipuChat, ZhipuConfig,
    QwenChat, QwenConfig,
    AnthropicChat, AnthropicConfig, AnthropicError,
};