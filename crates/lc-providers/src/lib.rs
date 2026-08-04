// lc-providers/src/lib.rs
//! LLM provider integrations for langchainrust.
//!
//! This crate provides unified chat model interfaces for:
//! - OpenAI (GPT-4, GPT-3.5)
//! - Ollama (local LLMs like Llama, Mistral)
//! - DeepSeek (cost-effective Chinese LLM)
//! - Moonshot (long-context Kimi)
//! - Qwen (Alibaba Cloud)
//! - Zhipu (ChatGLM)
//! - Anthropic (Claude)
//! - Gemini (Google)

mod error;
mod wrapper;

/// LLMClient — zero-config unified entry point.
pub mod client;
/// Ollama local LLM integration.
pub mod ollama;
/// OpenAI API integration.
pub mod openai;
/// Third-party provider integrations.
pub mod providers;

pub use client::LLMClient;
pub use error::ProviderError;
pub use ollama::{OllamaChat, OllamaConfig};
pub use openai::{
    AssistantError, BuiltinTool, OpenAIAssistant, OpenAIChat, OpenAIConfig, ResponsesConfig,
    ResponsesError, ResponsesModel,
};
pub use providers::{
    AnthropicChat, AnthropicConfig, AnthropicError, AnthropicStreamToken,
    AnthropicStructuredOutputMethod, DeepSeekChat, DeepSeekConfig, GeminiChat, GeminiConfig,
    GeminiError, GeminiStructuredOutputMethod, MoonshotChat, MoonshotConfig, QwenChat, QwenConfig,
    ThinkingConfig, ThinkingType, ZhipuChat, ZhipuConfig,
};
pub use wrapper::{wrap_chat_model, ChatModelWrapper};

/// Global mutex for serializing env-var tests across the crate.
/// Env vars are process-global, so parallel tests that set/remove them
/// can race. All env-var tests should `lock()` this before touching
/// `std::env::set_var` / `std::env::remove_var`.
#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
