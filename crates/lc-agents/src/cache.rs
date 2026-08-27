// lc-agents/src/cache.rs
//! LLM result cache (P2-1)
//!
//! `plan()`'s LLM calls in the agent loop are keyed by `(input +
//! intermediate steps + executor namespace)`; under deterministic prompts the
//! same input reuses the previous `AgentOutput`, skipping the LLM round trip.
//! Tool execution results (observations) enter the key; tools themselves are
//! not cached.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// LLM result cache abstraction.
///
/// Values are carried as strings (internally a serialized `AgentOutput`); the
/// implementation can be swapped for disk / Redis / cross-process sharing as
/// long as `get`/`put` semantics stay consistent.
pub trait ResponseCache: Send + Sync {
    /// Returns the serialized result for `key`, if cached.
    fn get(&self, key: &str) -> Option<String>;
    /// Writes a cache entry.
    fn put(&self, key: String, value: String);
    /// Clears the cache.
    fn clear(&self);
}

/// Bounded in-memory cache.
///
/// When entries exceed `max_entries`, the oldest are evicted FIFO to keep the
/// deterministic cache from growing without bound.
#[derive(Default)]
pub struct MemoryCache {
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    map: HashMap<String, String>,
    order: VecDeque<String>,
    max_entries: usize,
}

impl Default for CacheInner {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_entries: 256,
        }
    }
}

impl MemoryCache {
    /// Creates an in-memory cache with default capacity (256 entries).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum entry count (at least 1).
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                max_entries: max_entries.max(1),
                ..Default::default()
            }),
        }
    }
}

impl ResponseCache for MemoryCache {
    fn get(&self, key: &str) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|inner| inner.map.get(key).cloned())
    }

    fn put(&self, key: String, value: String) {
        if let Ok(mut inner) = self.inner.lock() {
            if inner.map.insert(key.clone(), value).is_none() {
                inner.order.push_back(key);
            }
            while inner.order.len() > inner.max_entries {
                if let Some(oldest) = inner.order.pop_front() {
                    inner.map.remove(&oldest);
                }
            }
        }
    }

    fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.map.clear();
            inner.order.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let cache = MemoryCache::new();
        assert!(cache.get("k").is_none());
        cache.put("k".to_string(), "v".to_string());
        assert_eq!(cache.get("k").as_deref(), Some("v"));
    }

    #[test]
    fn test_cache_evicts_oldest() {
        let cache = MemoryCache::with_capacity(2);
        cache.put("a".to_string(), "1".to_string());
        cache.put("b".to_string(), "2".to_string());
        cache.put("c".to_string(), "3".to_string());
        assert!(cache.get("a").is_none(), "最旧条目应被淘汰");
        assert_eq!(cache.get("b").as_deref(), Some("2"));
        assert_eq!(cache.get("c").as_deref(), Some("3"));
    }

    #[test]
    fn test_cache_clear() {
        let cache = MemoryCache::new();
        cache.put("a".to_string(), "1".to_string());
        cache.clear();
        assert!(cache.get("a").is_none());
    }
}
