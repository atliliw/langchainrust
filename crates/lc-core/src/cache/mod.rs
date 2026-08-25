// src/core/cache/mod.rs
//! LLM 调用缓存
//!
//! 提供内存缓存功能，减少重复 LLM 调用。
//! 支持 TTL（过期时间）和最大条目数限制。

pub mod llm_cache;

pub use llm_cache::{CacheConfig, CacheError, CachedLLMResult, LLMCache};
