// src/language_models/providers/mod.rs
//! Third-party LLM provider integrations.
//!
//! This module provides unified API wrappers for various LLM providers:
//! - DeepSeek: Cost-effective Chinese LLM provider
//! - Moonshot (Kimi): Long-context Chinese LLM
//! - Qwen: Alibaba Cloud's Qwen series
//! - Zhipu (ChatGLM): Chinese enterprise LLM
//! - Anthropic (Claude): Safety-focused Western LLM

pub mod deepseek;
pub mod moonshot;
pub mod zhipu;
pub mod qwen;
pub mod anthropic;

pub use deepseek::{DeepSeekChat, DeepSeekConfig, DEEPSEEK_BASE_URL, DEEPSEEK_MODELS};
pub use moonshot::{MoonshotChat, MoonshotConfig, MOONSHOT_BASE_URL, MOONSHOT_MODELS};
pub use zhipu::{ZhipuChat, ZhipuConfig, ZHIPU_BASE_URL, ZHIPU_MODELS};
pub use qwen::{QwenChat, QwenConfig, QWEN_BASE_URL, QWEN_MODELS};
pub use anthropic::{AnthropicChat, AnthropicConfig, AnthropicError, ANTHROPIC_BASE_URL, CLAUDE_MODELS};