// src/language_models/providers/mod.rs
//! Third-party LLM provider integrations.
//!
//! This module provides unified API wrappers for various LLM providers:
//! - DeepSeek: Cost-effective Chinese LLM provider
//! - Moonshot (Kimi): Long-context Chinese LLM
//! - Qwen: Alibaba Cloud's Qwen series
//! - Zhipu (ChatGLM): Chinese enterprise LLM
//! - Anthropic (Claude): Safety-focused Western LLM

pub mod anthropic;
pub mod azure;
pub mod cohere;
pub mod deepseek;
pub mod gemini;
pub mod mistral;
pub mod moonshot;
pub mod qwen;
pub mod zhipu;

pub use anthropic::{
    AnthropicChat, AnthropicConfig, AnthropicError, AnthropicStreamToken,
    AnthropicStructuredOutputMethod, ThinkingConfig, ThinkingType, ANTHROPIC_BASE_URL,
    CLAUDE_MODELS,
};
pub use azure::{AzureOpenAIChat, AzureOpenAIConfig, AzureOpenAIError, AZURE_DEFAULT_API_VERSION};
pub use cohere::{CohereChat, CohereConfig, CohereError, COHERE_BASE_URL, COHERE_MODELS};
pub use deepseek::{DeepSeekChat, DeepSeekConfig, DEEPSEEK_BASE_URL, DEEPSEEK_MODELS};
pub use gemini::{
    GeminiChat, GeminiConfig, GeminiError, GeminiStructuredOutputMethod, GEMINI_BASE_URL,
    GEMINI_MODELS,
};
pub use mistral::{MistralChat, MistralConfig, MISTRAL_BASE_URL, MISTRAL_MODELS};
pub use moonshot::{MoonshotChat, MoonshotConfig, MOONSHOT_BASE_URL, MOONSHOT_MODELS};
pub use qwen::{QwenChat, QwenConfig, QWEN_BASE_URL, QWEN_MODELS};
pub use zhipu::{ZhipuChat, ZhipuConfig, ZHIPU_BASE_URL, ZHIPU_MODELS};
