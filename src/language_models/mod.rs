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

/// Ollama local LLM integration.
pub mod ollama;
/// OpenAI API integration.
pub mod openai;
/// Third-party provider integrations.
pub mod providers;

pub use ollama::{OllamaChat, OllamaConfig};
pub use openai::{
    AssistantError, BuiltinTool, OpenAIAssistant, OpenAIChat, OpenAIConfig, ResponsesConfig,
    ResponsesError, ResponsesModel,
};
pub use providers::{
    AnthropicChat, AnthropicConfig, AnthropicError, AnthropicStreamToken, DeepSeekChat,
    DeepSeekConfig, GeminiChat, GeminiConfig, GeminiError, MoonshotChat, MoonshotConfig, QwenChat,
    QwenConfig, ThinkingConfig, ThinkingType, ZhipuChat, ZhipuConfig,
};
