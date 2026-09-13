#![warn(missing_docs)]
// lc-providers/src/lib.rs
//! LLM provider integrations for langchainrust.
//!
//! This crate provides unified chat model interfaces for:
//! - OpenAI (GPT-5 / GPT-4o / o-series)
//! - Generic OpenAI-compatible endpoints ([`openai_compatible`]): Groq,
//!   OpenRouter, xAI (Grok), and arbitrary self-hosted/private base URLs
//! - Ollama (local LLMs like Llama, Mistral)
//! - DeepSeek (cost-effective Chinese LLM)
//! - Moonshot (long-context Kimi)
//! - Qwen (Alibaba Cloud)
//! - Zhipu (ChatGLM)
//! - Anthropic (Claude)
//! - Gemini (Google)

mod error;
mod media;
mod retry;
mod sampling;
mod wrapper;

/// LLMClient — zero-config unified entry point.
pub mod client;
/// Ollama local LLM integration.
pub mod ollama;
/// OpenAI API integration.
pub mod openai;
/// B5: generic OpenAI-compatible endpoint support (Groq, OpenRouter, xAI,
/// vLLM, LM Studio, private gateways) — one transport, named presets.
pub mod openai_compatible;
/// Third-party provider integrations.
pub mod providers;

pub use client::LLMClient;
pub use error::ProviderError;
pub use ollama::{OllamaChat, OllamaConfig};
pub use openai::{
    AssistantError, BuiltinTool, OpenAIAssistant, OpenAIChat, OpenAIConfig, ResponsesConfig,
    ResponsesError, ResponsesModel, OPENAI_MODELS,
};
pub use openai_compatible::{
    GroqChat, OpenAICompatibleChat, OpenAICompatibleConfig, OpenRouterChat, XaiChat,
    DEFAULT_GROQ_MODEL, DEFAULT_XAI_MODEL, GROQ_BASE_URL, GROQ_MODELS, OPENROUTER_BASE_URL,
    XAI_BASE_URL, XAI_MODELS,
};
pub use providers::{
    AnthropicChat, AnthropicConfig, AnthropicError, AnthropicStreamToken,
    AnthropicStructuredOutputMethod, AnthropicUsage, AzureOpenAIChat, AzureOpenAIConfig,
    AzureOpenAIError, CohereChat, CohereConfig, CohereError, DeepSeekChat, DeepSeekConfig,
    GeminiChat, GeminiConfig, GeminiError, GeminiStructuredOutputMethod, MistralChat,
    MistralConfig, MoonshotChat, MoonshotConfig, QwenChat, QwenConfig, ThinkingConfig,
    ThinkingType, ZhipuChat, ZhipuConfig,
};
pub use wrapper::{wrap_chat_model, ChatModelWrapper};

/// Global mutex for serializing env-var tests across the crate.
/// Env vars are process-global, so parallel tests that set/remove them
/// can race. All env-var tests should `lock()` this before touching
/// `std::env::set_var` / `std::env::remove_var`.
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
