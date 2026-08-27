// src/core/cache/llm_cache.rs
//! LLM call cache implementation.
//!
//! In-memory LRU cache that memoizes repeated LLM call results.
//! Supports optional TTL expiry and a maximum-entry limit.

use crate::language_models::LLMResult;
use lc_schema::Message;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Error type for the LLM cache.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    /// Failed to serialize the cache key.
    #[error("cache key serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Cached LLM result, with its expiry time
#[derive(Debug, Clone)]
pub struct CachedLLMResult {
    /// The LLM result
    pub result: LLMResult,
    /// When it was cached
    pub cached_at: Instant,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache entries (0 = unlimited)
    pub max_entries: usize,
    /// TTL expiry (None = never expires)
    pub ttl: Option<Duration>,
    /// Whether the cache is enabled
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Some(Duration::from_secs(3600)), // expires after 1 hour by default
            enabled: true,
        }
    }
}

impl CacheConfig {
    /// Creates a cache config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables TTL (never expires)
    pub fn no_ttl(mut self) -> Self {
        self.ttl = None;
        self
    }

    /// Sets the TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Sets the maximum entry count
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Disables the cache
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// LLM call cache
///
/// Caches LLM call inputs/outputs, avoiding duplicate API calls for identical requests.
///
/// # Example
/// ```ignore
/// use langchainrust::core::cache::LLMCache;
///
/// let cache = LLMCache::new();
/// cache.put("key", llm_result).await;
///
/// if let Some(cached) = cache.get("key").await {
///     println!("缓存命中: {}", cached.result.content);
/// }
/// ```
pub struct LLMCache {
    config: CacheConfig,
    store: RwLock<HashMap<String, CachedLLMResult>>,
}

impl LLMCache {
    /// Creates a cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Creates a cache with the given configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            config,
            store: RwLock::new(HashMap::new()),
        }
    }

    /// Builds a cache key from a message list.
    ///
    /// Serializes the message list to a JSON string as the key.
    /// Includes the model name so calls to different models do not affect each other.
    /// On serialization failure, returns an error instead of falling back to an empty string (M34).
    pub fn build_key(messages: &[Message], model: &str) -> Result<String, CacheError> {
        let serialized = serde_json::to_string(messages).map_err(CacheError::Serialization)?;
        Ok(format!("{}:{}", model, serialized))
    }

    /// Fetches a cached result.
    ///
    /// An expired entry is removed immediately (H36). A hit refreshes `cached_at`,
    /// keeping the cache in true LRU semantics (Q7: a hit makes the entry the most-recently used).
    pub async fn get(&self, key: &str) -> Option<CachedLLMResult> {
        if !self.config.enabled {
            return None;
        }

        let store = self.store.read().await;
        let entry = store.get(key)?;

        // check TTL
        if let Some(ttl) = self.config.ttl {
            if entry.cached_at.elapsed() > ttl {
                // H36: expired entries must be removed; drop the read lock first, then take the write lock
                drop(store);
                let mut store = self.store.write().await;
                // Double-check after acquiring write lock
                if let Some(entry) = store.get(key) {
                    if entry.cached_at.elapsed() > ttl {
                        store.remove(key);
                    }
                }
                return None;
            }
        }

        let result = entry.clone();

        // Q7: refresh the LRU timestamp on a hit. Drop the read lock, then take the write lock to update.
        drop(store);
        let mut store = self.store.write().await;
        if let Some(entry) = store.get_mut(key) {
            entry.cached_at = Instant::now();
        }

        Some(result)
    }

    /// Stores a cached result
    pub async fn put(&self, key: impl Into<String>, result: LLMResult) {
        if !self.config.enabled {
            return;
        }

        let mut store = self.store.write().await;

        // check whether eviction is needed
        if self.config.max_entries > 0 && store.len() >= self.config.max_entries {
            // evict the oldest entry
            if let Some(oldest_key) = store
                .iter()
                .min_by_key(|(_, v)| v.cached_at)
                .map(|(k, _)| k.clone())
            {
                store.remove(&oldest_key);
            }
        }

        store.insert(
            key.into(),
            CachedLLMResult {
                result,
                cached_at: Instant::now(),
            },
        );
    }

    /// Clears the cache
    pub async fn clear(&self) {
        let mut store = self.store.write().await;
        store.clear();
    }

    /// Returns the cache size
    pub async fn len(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }

    /// Whether the cache is empty
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Removes expired entries
    pub async fn evict_expired(&self) -> usize {
        if let Some(ttl) = self.config.ttl {
            let mut store = self.store.write().await;
            let before = store.len();
            store.retain(|_, v| v.cached_at.elapsed() <= ttl);
            before - store.len()
        } else {
            0
        }
    }
}

impl Default for LLMCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::TokenUsage;

    fn make_result(content: &str) -> LLMResult {
        LLMResult {
            content: content.to_string(),
            model: "test-model".to_string(),
            token_usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
            tool_calls: None,
            thinking_content: None,
        }
    }

    #[tokio::test]
    async fn test_cache_put_and_get() {
        let cache = LLMCache::new();
        let key = "test-key";
        let result = make_result("Hello, world!");

        cache.put(key.to_string(), result.clone()).await;
        let cached = cache.get(key).await;

        assert!(cached.is_some());
        assert_eq!(cached.unwrap().result.content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = LLMCache::new();
        let cached = cache.get("non-existent").await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = LLMCache::new();
        cache.put("k1".to_string(), make_result("r1")).await;
        cache.put("k2".to_string(), make_result("r2")).await;
        assert_eq!(cache.len().await, 2);

        cache.clear().await;
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        let config = CacheConfig::new().disabled();
        let cache = LLMCache::with_config(config);

        cache.put("key".to_string(), make_result("test")).await;
        let cached = cache.get("key").await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let config = CacheConfig::new().with_ttl(Duration::from_millis(10));
        let cache = LLMCache::with_config(config);

        cache.put("key".to_string(), make_result("test")).await;
        assert!(cache.get("key").await.is_some());

        // wait for expiry
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(cache.get("key").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_max_entries() {
        let config = CacheConfig::new().with_max_entries(3).no_ttl();
        let cache = LLMCache::with_config(config);

        cache.put("a".to_string(), make_result("1")).await;
        cache.put("b".to_string(), make_result("2")).await;
        cache.put("c".to_string(), make_result("3")).await;
        assert_eq!(cache.len().await, 3);

        // past the cap: evict the oldest entry
        cache.put("d".to_string(), make_result("4")).await;
        assert_eq!(cache.len().await, 3);
        // "a" should have been evicted
        assert!(cache.get("a").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_get_refreshes_lru_order() {
        // max_entries = 2: hit on "a" must make "a" the most-recent entry, so
        // inserting "c" evicts "b" (the old LRU) instead of "a".
        let config = CacheConfig::new().with_max_entries(2).no_ttl();
        let cache = LLMCache::with_config(config);

        cache.put("a".to_string(), make_result("1")).await;
        cache.put("b".to_string(), make_result("2")).await;

        // Hit "a" → refreshes its cached_at.
        assert!(cache.get("a").await.is_some());

        // Push past the cap: the LRU evicts "b", keeping "a".
        cache.put("c".to_string(), make_result("3")).await;
        assert!(cache.get("a").await.is_some());
        assert!(cache.get("b").await.is_none());
        assert!(cache.get("c").await.is_some());
    }

    #[tokio::test]
    async fn test_cache_no_ttl() {
        let config = CacheConfig::new().no_ttl();
        let cache = LLMCache::with_config(config);

        cache.put("key".to_string(), make_result("persist")).await;

        // should not expire even after a long wait
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(cache.get("key").await.is_some());
    }

    #[tokio::test]
    async fn test_cache_evict_expired() {
        // 0 TTL forces immediate expiry
        let config = CacheConfig::new().with_ttl(Duration::from_millis(0));
        let cache = LLMCache::with_config(config);

        cache.put("key".to_string(), make_result("test")).await;
        tokio::time::sleep(Duration::from_millis(1)).await;

        let evicted = cache.evict_expired().await;
        assert_eq!(evicted, 1);
        assert!(cache.is_empty().await);
    }

    #[tokio::test]
    async fn test_cache_build_key() {
        let messages = vec![Message::human("Hello"), Message::ai("Hi!")];
        let key = LLMCache::build_key(&messages, "gpt-4").unwrap();
        assert!(key.contains("gpt-4"));
        assert!(key.contains("Hello"));
    }

    #[tokio::test]
    async fn test_cache_is_empty() {
        let cache = LLMCache::new();
        assert!(cache.is_empty().await);

        cache.put("key".to_string(), make_result("test")).await;
        assert!(!cache.is_empty().await);
    }
}
