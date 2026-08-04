// src/core/cache/llm_cache.rs
//! LLM 调用缓存实现
//!
//! 基于内存的 LRU 缓存，缓存重复的 LLM 调用结果。
//! 支持可选的 TTL 过期和最大条目限制。

use crate::language_models::LLMResult;
use lc_schema::Message;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// 缓存的 LLM 结果，包含过期时间
#[derive(Debug, Clone)]
pub struct CachedLLMResult {
    /// LLM 返回结果
    pub result: LLMResult,
    /// 缓存时间戳
    pub cached_at: Instant,
}

/// 缓存配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 最大缓存条目数（0 表示不限制）
    pub max_entries: usize,
    /// TTL 过期时间（None 表示永不过期）
    pub ttl: Option<Duration>,
    /// 是否启用
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Some(Duration::from_secs(3600)), // 默认 1 小时过期
            enabled: true,
        }
    }
}

impl CacheConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// 禁用 TTL（永不过期）
    pub fn no_ttl(mut self) -> Self {
        self.ttl = None;
        self
    }

    /// 设置 TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// 设置最大条目数
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// 禁用缓存
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

/// LLM 调用缓存
///
/// 缓存 LLM 调用的输入输出，避免相同请求重复调用 API。
///
/// # 示例
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
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            config,
            store: RwLock::new(HashMap::new()),
        }
    }

    /// 从消息列表生成缓存键
    ///
    /// 将消息列表序列化为 JSON 字符串作为键。
    /// 包含 model 名称以确保不同模型的调用不互相影响。
    /// 如果序列化失败，返回错误而非回退到空字符串（M34）。
    pub fn build_key(messages: &[Message], model: &str) -> Result<String, String> {
        let serialized = serde_json::to_string(messages)
            .map_err(|e| format!("cache key serialization failed: {}", e))?;
        Ok(format!("{}:{}", model, serialized))
    }

    /// 获取缓存结果
    ///
    /// 如果发现过期条目，会立即删除（H36）。
    pub async fn get(&self, key: &str) -> Option<CachedLLMResult> {
        if !self.config.enabled {
            return None;
        }

        let store = self.store.read().await;
        if let Some(entry) = store.get(key) {
            // 检查 TTL
            if let Some(ttl) = self.config.ttl {
                if entry.cached_at.elapsed() > ttl {
                    // H36: 过期条目需要删除，先释放读锁再获取写锁
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
            Some(entry.clone())
        } else {
            None
        }
    }

    /// 存入缓存结果
    pub async fn put(&self, key: String, result: LLMResult) {
        if !self.config.enabled {
            return;
        }

        let mut store = self.store.write().await;

        // 检查是否需要淘汰
        if self.config.max_entries > 0 && store.len() >= self.config.max_entries {
            // 移除最早的一条
            if let Some(oldest_key) = store
                .iter()
                .min_by_key(|(_, v)| v.cached_at)
                .map(|(k, _)| k.clone())
            {
                store.remove(&oldest_key);
            }
        }

        store.insert(
            key,
            CachedLLMResult {
                result,
                cached_at: Instant::now(),
            },
        );
    }

    /// 清除缓存
    pub async fn clear(&self) {
        let mut store = self.store.write().await;
        store.clear();
    }

    /// 获取缓存大小
    pub async fn len(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }

    /// 缓存是否为空
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// 移除过期条目
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

        // 等待过期
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

        // 超过限制，淘汰最早的一条
        cache.put("d".to_string(), make_result("4")).await;
        assert_eq!(cache.len().await, 3);
        // a 应该被淘汰
        assert!(cache.get("a").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_no_ttl() {
        let config = CacheConfig::new().no_ttl();
        let cache = LLMCache::with_config(config);

        cache.put("key".to_string(), make_result("persist")).await;

        // 即使等待许久也不应过期
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(cache.get("key").await.is_some());
    }

    #[tokio::test]
    async fn test_cache_evict_expired() {
        // 用 0 TTL 确保立即过期
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
