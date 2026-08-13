// lc-agents/src/cache.rs
//! LLM 结果缓存(P2-1)
//!
//! Agent 循环里 `plan()` 的 LLM 调用按 `(输入 + 中间步骤 + 执行器命名空间)`
//! 哈希命中缓存,确定性 prompt 场景下相同输入直接复用上次的 `AgentOutput`,
//! 跳过 LLM 往返。工具执行结果(observation)进入 key,不缓存工具本身。

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// LLM 结果缓存抽象。
///
/// 值以字符串承载(内部存序列化后的 `AgentOutput`),实现可换成磁盘 / Redis /
/// 进程间共享,只要 `get`/`put` 语义一致。
pub trait ResponseCache: Send + Sync {
    /// 按 key 命中缓存,返回序列化结果。
    fn get(&self, key: &str) -> Option<String>;
    /// 写入缓存。
    fn put(&self, key: String, value: String);
    /// 清空缓存。
    fn clear(&self);
}

/// 有界内存缓存。
///
/// 条目超过 `max_entries` 时按 FIFO 淘汰最旧条目,防止确定性缓存无限增长。
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
    /// 创建默认容量(256 条)的内存缓存。
    pub fn new() -> Self {
        Self::default()
    }

    /// 指定最大条目数(至少 1)。
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
