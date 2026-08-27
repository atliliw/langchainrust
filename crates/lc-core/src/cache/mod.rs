// src/core/cache/mod.rs
//! LLM call cache
//!
//! In-memory caching to reduce repeated LLM calls.
//! Supports TTL (expiry) and a maximum-entry limit.

pub mod llm_cache;

pub use llm_cache::{CacheConfig, CacheError, CachedLLMResult, LLMCache};
