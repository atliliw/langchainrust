// src/core/language_models/mod.rs
//! 语言模型基础模块
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/language_models/base.py

mod base;
mod chat;

pub use base::BaseLanguageModel;
pub use chat::{BaseChatModel, LLMResult, TokenUsage};