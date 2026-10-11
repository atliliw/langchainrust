// lc-embeddings/src/cached.rs
//! `CachingEmbeddings`: transparent embedding cache + per-batch dedup (L153).
//!
//! Wraps any [`Embeddings`] and adds two layers:
//!
//! - **Cross-request cache** — the same text embedded once is served from an
//!   in-memory `text → vector` map on later queries, so repeat calls hit the
//!   provider/CPU once and are answered with a clone after that.
//! - **Batch dedup** — `embed_documents` embeds each distinct text in the batch
//!   exactly once, then maps the results back in the caller's order.
//!
//! v0.27.0 T3 (zero-copy): cached vectors are stored as `Arc<Vec<f32>>`, so a
//! cache miss *moves* the delegate's vector into the cache (no payload copy)
//! and a hit is a refcount bump. `embed_documents` therefore copies once per
//! returned row — the minimum the `Vec<Vec<f32>>` return type allows — instead
//! of once into the cache plus once per row. `embed_query` still returns an
//! owned vector, i.e. exactly the one copy it always made.
//!
//! The cache is **bounded**: at most [`CachingEmbeddings::capacity`] keys survive.
//! Eviction is deliberately coarse (clear the whole map once the cap is reached)
//! so the memory footprint stays bounded with no bookkeeping cost — a hot, small
//! working set of texts is what this optimizes for, and for RAG that is the
//! typical profile. Identical-output guarantees are inherited from the delegate:
//! only successful results are cached, and `dimension` / `model_name` pass through.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::{EmbeddingError, Embeddings};

/// Default cache capacity when none is given.
const DEFAULT_CAPACITY: usize = 4096;

/// A caching, deduplicating wrapper around any [`Embeddings`].
pub struct CachingEmbeddings<M> {
    inner: M,
    /// Cached vectors are shared behind an [`Arc`] (v0.27.0 T3, zero-copy batch
    /// path): a hit costs a refcount bump, and [`Embeddings::embed_documents`]
    /// can materialize one copy per *output row* instead of one copy into the
    /// cache plus one copy per row. The inner `Vec` is never cloned into the
    /// cache — the delegate's vector is moved into the `Arc`.
    cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    capacity: usize,
}

impl<M: Embeddings> CachingEmbeddings<M> {
    /// Wraps `inner` with a cache that keeps at most `capacity` distinct texts.
    ///
    /// A `capacity` of `0` is interpreted as `DEFAULT_CAPACITY` (the user asking
    /// for "no cap" would be unbounded memory; an explicit bound is the point).
    pub fn new(inner: M, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
            capacity: if capacity == 0 {
                DEFAULT_CAPACITY
            } else {
                capacity
            },
        }
    }

    /// Wraps `inner` with the default cache capacity.
    pub fn wrap(inner: M) -> Self {
        Self::new(inner, DEFAULT_CAPACITY)
    }

    /// Accesses the delegate.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// The maximum number of distinct texts held in the cache.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of distinct texts currently cached.
    pub fn cached_len(&self) -> usize {
        self.cache
            .lock()
            .map(|c| c.len())
            .unwrap_or_else(|e| e.into_inner().len())
    }

    /// Evicts the whole cache. Useful when the underlying data may have changed
    /// (e.g. the delegate is a retriever DB whose embeddings were re-indexed).
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Inserts one result, evicting the whole cache once the capacity is reached.
    /// Kept as a short, non-awaiting critical section (no `await` while locked).
    fn cache_insert(&self, key: String, vector: Arc<Vec<f32>>) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= self.capacity && !cache.contains_key(&key) {
            // Coarse bounded eviction: when full, drop everything. Documented
            // trade-off (see module docs) — cheap, and enough for a small hot set.
            cache.clear();
        }
        cache.insert(key, vector);
    }

    /// Returns the (shared) vector for `text`, computing and caching it on a
    /// miss.
    ///
    /// On a miss the delegate's `Vec<f32>` is **moved** into the cache entry —
    /// no copy of the payload — so the batch path below pays one copy per
    /// returned row instead of one per row plus one into the cache.
    async fn cached_vector(&self, text: &str) -> Result<Arc<Vec<f32>>, EmbeddingError> {
        // Lock held only for the lookup (no await while locked).
        if let Some(v) = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(text)
        {
            return Ok(Arc::clone(v));
        }
        let vector = Arc::new(self.inner.embed_query(text).await?);
        self.cache_insert(text.to_string(), Arc::clone(&vector));
        Ok(vector)
    }
}

#[async_trait]
impl<M: Embeddings> Embeddings for CachingEmbeddings<M> {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // A single caller needs an owned vector, so exactly one copy is made —
        // the same one copy the pre-Arc implementation made.
        Ok(self.cached_vector(text).await?.as_ref().clone())
    }

    /// Override: embeds each distinct text exactly once and maps results back in
    /// the caller's order, so a batch with repeated texts (common in RAG re-query)
    /// issues only the unique queries. The emptiness contract is preserved.
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // One pass builds both the unique-text list and a position → unique-index
        // map. Reassembling through indices (instead of the previous text-keyed
        // `HashMap<&str, Vec<f32>>`) drops a second hash of every input text and
        // the per-text map node, and removes a lookup that could only ever miss.
        let mut unique: Vec<&str> = Vec::with_capacity(texts.len());
        let mut index_of: HashMap<&str, usize> = HashMap::with_capacity(texts.len());
        let mut positions: Vec<usize> = Vec::with_capacity(texts.len());
        for t in texts {
            let index = *index_of.entry(t).or_insert_with(|| {
                unique.push(t);
                unique.len() - 1
            });
            positions.push(index);
        }

        let mut vectors: Vec<Arc<Vec<f32>>> = Vec::with_capacity(unique.len());
        for t in &unique {
            vectors.push(self.cached_vector(t).await?);
        }

        // Every position maps to a computed vector, so the reassembly cannot
        // miss (the previous `.unwrap_or_default()` could silently emit a zero
        // vector).
        Ok(positions
            .into_iter()
            .map(|i| vectors[i].as_ref().clone())
            .collect())
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockEmbeddings;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Delegate that records how many times it was actually asked to embed —
    /// the observable side of the dedup + cache contract.
    struct CountingEmbeddings {
        inner: MockEmbeddings,
        calls: AtomicUsize,
    }

    impl CountingEmbeddings {
        fn new(dim: usize) -> Self {
            Self {
                inner: MockEmbeddings::new(dim),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl Embeddings for CountingEmbeddings {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.embed_query(text).await
        }

        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn model_name(&self) -> &str {
            self.inner.model_name()
        }
    }

    /// v0.27.0 T3: the batch path computes each distinct text exactly once and
    /// reassembles in caller order (the index-mapped replacement for the old
    /// text-keyed map).
    #[tokio::test]
    async fn batch_computes_each_unique_text_once_in_order() {
        let cached = CachingEmbeddings::wrap(CountingEmbeddings::new(8));
        let results = cached
            .embed_documents(&["a", "b", "a", "c", "b", "a"])
            .await
            .unwrap();

        assert_eq!(results.len(), 6, "one row per input, order preserved");
        assert_eq!(results[0], results[2]);
        assert_eq!(results[0], results[5]);
        assert_eq!(results[1], results[4]);
        assert_ne!(results[0], results[1]);
        assert_eq!(
            cached.inner().calls(),
            3,
            "only the three distinct texts were embedded"
        );
        assert_eq!(cached.cached_len(), 3);
    }

    /// The batch path populates the same cache `embed_query` reads, and a later
    /// query is served from it without a second delegate call.
    #[tokio::test]
    async fn batch_populates_cache_for_later_queries() {
        let cached = CachingEmbeddings::wrap(CountingEmbeddings::new(4));
        let batch = cached.embed_documents(&["x", "y"]).await.unwrap();
        assert_eq!(cached.inner().calls(), 2);

        let single = cached.embed_query("x").await.unwrap();
        assert_eq!(single, batch[0], "cache hit returns the same vector");
        assert_eq!(cached.inner().calls(), 2, "no extra delegate call");
    }

    /// A batch that is entirely cache hits costs no delegate calls at all.
    #[tokio::test]
    async fn warm_batch_hits_cache_entirely() {
        let cached = CachingEmbeddings::wrap(CountingEmbeddings::new(4));
        cached.embed_documents(&["a", "b"]).await.unwrap();
        let again = cached.embed_documents(&["a", "b", "a"]).await.unwrap();
        assert_eq!(cached.inner().calls(), 2, "warm batch recomputed nothing");
        assert_eq!(again[0], again[2]);
        assert_ne!(again[0], again[1]);
    }

    /// `embed_query` returns the identical vector on a cache hit without a second
    /// delegate call (L153: one compute, many answers).
    #[tokio::test]
    async fn caches_query_by_text() {
        let inner = MockEmbeddings::new(8);
        let cached = CachingEmbeddings::wrap(inner);

        let first = cached.embed_query("same text").await.unwrap();
        let second = cached.embed_query("same text").await.unwrap();
        assert_eq!(first, second);
        assert_eq!(cached.cached_len(), 1, "one distinct text cached");
        assert_eq!(cached.embed_query("another").await.unwrap().len(), 8);
        assert_eq!(cached.cached_len(), 2);
    }

    /// `embed_documents` embeds each distinct text once and returns results in
    /// the caller's original order (L153 dedup).
    #[tokio::test]
    async fn dedups_batch_keeping_order() {
        let cached = CachingEmbeddings::wrap(MockEmbeddings::new(4));
        let results = cached
            .embed_documents(&["a", "b", "a", "c", "b"])
            .await
            .unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], results[2], "a repeats map to same vector");
        assert_eq!(results[1], results[4], "b repeats map to same vector");
        assert_ne!(results[0], results[1]);
        assert_eq!(cached.cached_len(), 3, "three distinct texts computed");
    }

    /// Emptiness / empty-batch contract is preserved by the wrapper.
    #[tokio::test]
    async fn preserves_empty_contract() {
        let cached = CachingEmbeddings::wrap(MockEmbeddings::new(4));
        assert!(cached.embed_documents(&[""]).await.is_err());
        assert!(cached.embed_documents(&["ok", "  "]).await.is_err());
        assert_eq!(cached.embed_documents(&[]).await.unwrap().len(), 0);
    }

    /// The bounded cache evicts once the capacity is exceeded but stays correct.
    #[tokio::test]
    async fn capacity_bounds_cache() {
        let cached = CachingEmbeddings::new(MockEmbeddings::new(2), 3);
        for i in 0..3 {
            cached.embed_query(&format!("t{i}")).await.unwrap();
        }
        assert_eq!(cached.cached_len(), 3);
        // A fourth distinct text clears the cache (coarse eviction).
        cached.embed_query("t3").await.unwrap();
        assert!(cached.cached_len() <= 2);
        cached.clear_cache();
        assert_eq!(cached.cached_len(), 0);
    }

    /// Delegates dimension and model name through to the underlying provider.
    #[tokio::test]
    async fn delegates_metadata() {
        let cached = CachingEmbeddings::wrap(MockEmbeddings::new(128));
        assert_eq!(cached.dimension(), 128);
        assert_eq!(cached.model_name(), "mock-embeddings");
        assert_eq!(
            CachingEmbeddings::new(MockEmbeddings::new(1), 0).capacity(),
            4096
        );
    }
}
