// lc-rag/src/semantic_cache.rs
//! Semantic cache for retrieval results (0.21.0 S4.2).
//!
//! A semantic cache answers repeated (or paraphrased) queries from cached
//! retrieval results instead of re-running embedding + retrieval:
//! - **lexical hit**: the query text is byte-identical to a cached query —
//!   always preferred (product codes, IDs: lexical is more reliable);
//! - **semantic hit**: the query embedding's cosine similarity against a
//!   cached query's embedding reaches the threshold.
//!
//! The cache is a [`RetrieverTrait`] decorator ([`CachedRetriever`]); it is
//! off-by-default (wrap your retriever explicitly) and eviction/TTL keep the
//! footprint bounded. `k` is part of the cached result set: a hit requires the
//! same `k` (a different `k` recomputes rather than slicing padded results).
//!
//! The inner bookkeeping ([`SemanticCacheCore`]) is a pure, embedder-free
//! unit so the threshold/FIFO/TTL/exact-priority semantics are testable with
//! synthetic vectors.

use crate::retriever::{RetrieverError, RetrieverTrait};
use lc_embeddings::Embeddings;
use lc_vector_stores::{Document, SearchResult};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Semantic cache configuration.
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    /// Minimum cosine similarity for a semantic hit. Conservative default:
    /// a too-low threshold serves semantically-nearbut-different queries.
    pub threshold: f32,
    /// Max entries; the oldest are evicted FIFO (same policy as lc-agents'
    /// `MemoryCache`).
    pub max_entries: usize,
    /// Optional TTL: entries older than this are treated as misses (corpus
    /// updates invalidate results; pair with `invalidate()` for explicit
    /// invalidation).
    pub ttl: Option<Duration>,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            threshold: 0.95,
            max_entries: 256,
            ttl: None,
        }
    }
}

impl SemanticCacheConfig {
    /// Creates a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the semantic-hit threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the max entry count (FIFO eviction).
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries.max(1);
        self
    }

    /// Sets an optional TTL.
    pub fn with_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }
}

/// One cache entry: query text + its embedding + the result set for `k`.
#[derive(Debug, Clone)]
struct CacheEntry {
    query: String,
    query_vector: Vec<f32>,
    k: usize,
    results: Vec<SearchResult>,
    inserted_at: Instant,
}

/// Hit kind returned by [`SemanticCacheCore::lookup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHitKind {
    /// Byte-identical query text.
    Lexical,
    /// Similarity reached the threshold.
    Semantic,
}

/// Pure cache bookkeeping (no embedder, no retriever — fully unit-testable).
#[derive(Debug)]
pub struct SemanticCacheCore {
    config: SemanticCacheConfig,
    entries: Mutex<CacheInner>,
}

#[derive(Debug, Default)]
struct CacheInner {
    map: Vec<CacheEntry>,
    order: VecDeque<String>,
}

impl SemanticCacheCore {
    /// Creates a core with the given config.
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            config,
            entries: Mutex::new(CacheInner::default()),
        }
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map
            .len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Looks up `query` (optionally with its embedding for semantic matching)
    /// for result sets of size `k`. Returns the cached results and the hit kind.
    ///
    /// `query_vector` should be the freshly embedded query; pass `None` to
    /// restrict to lexical hits only.
    pub fn lookup(
        &self,
        query: &str,
        query_vector: Option<&[f32]>,
        k: usize,
        now: Instant,
    ) -> Option<(Vec<SearchResult>, CacheHitKind)> {
        let inner = self.entries.lock().unwrap_or_else(|e| e.into_inner());

        // 1. Lexical exact match wins.
        for entry in &inner.map {
            if entry.query == query && entry.k == k && !self.is_expired(&entry.inserted_at, now) {
                return Some((entry.results.clone(), CacheHitKind::Lexical));
            }
        }

        // 2. Semantic match above threshold.
        let query_vector = query_vector?;
        for entry in &inner.map {
            if entry.k != k || self.is_expired(&entry.inserted_at, now) {
                continue;
            }
            if lc_embeddings::cosine_similarity(query_vector, &entry.query_vector).unwrap_or(0.0)
                >= self.config.threshold
            {
                return Some((entry.results.clone(), CacheHitKind::Semantic));
            }
        }
        None
    }

    /// Inserts a result set for `(query, k)`.
    pub fn insert(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        k: usize,
        results: Vec<SearchResult>,
        now: Instant,
    ) {
        let mut inner = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        // Replace existing entry for the same (query, k).
        if let Some(pos) = inner.map.iter().position(|e| e.query == query && e.k == k) {
            inner.map.remove(pos);
            if let Some(p) = inner.order.iter().position(|q| q == query) {
                inner.order.remove(p);
            }
        }
        // FIFO eviction.
        while inner.map.len() >= self.config.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                if let Some(pos) = inner.map.iter().position(|e| e.query == oldest) {
                    inner.map.remove(pos);
                }
            } else {
                break;
            }
        }
        inner.order.push_back(query.to_string());
        inner.map.push(CacheEntry {
            query: query.to_string(),
            query_vector,
            k,
            results,
            inserted_at: now,
        });
    }

    /// Clears all entries (corpus update invalidation).
    pub fn invalidate(&self) {
        let mut inner = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        inner.map.clear();
        inner.order.clear();
    }

    fn is_expired(&self, inserted_at: &Instant, now: Instant) -> bool {
        match self.config.ttl {
            Some(ttl) => now.duration_since(*inserted_at) > ttl,
            None => false,
        }
    }
}

/// A [`RetrieverTrait`] decorator backed by a [`SemanticCacheCore`].
///
/// Query flow: embed once → lexical hit → semantic hit → miss (call inner,
/// insert). On a hit no retrieval call (and only the embedding call) happens;
/// the embedding itself could be skipped only for lexical hits — which skip
/// embedding too. Failures of the inner retriever propagate (never cached).
pub struct CachedRetriever {
    inner: Arc<dyn RetrieverTrait>,
    embeddings: Arc<dyn Embeddings>,
    cache: Arc<SemanticCacheCore>,
}

impl std::fmt::Debug for CachedRetriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedRetriever")
            .field("entries", &self.cache.len())
            .finish()
    }
}

impl CachedRetriever {
    /// Wraps `inner` with a semantic cache using `config`.
    pub fn new(
        inner: Arc<dyn RetrieverTrait>,
        embeddings: Arc<dyn Embeddings>,
        config: SemanticCacheConfig,
    ) -> Self {
        Self {
            inner,
            embeddings,
            cache: Arc::new(SemanticCacheCore::new(config)),
        }
    }

    /// The underlying cache core (for inspection / invalidation).
    pub fn cache(&self) -> &Arc<SemanticCacheCore> {
        &self.cache
    }

    async fn lookup_or_retrieve(
        &self,
        query: &str,
        k: usize,
    ) -> Result<(Vec<SearchResult>, Option<CacheHitKind>), RetrieverError> {
        let now = Instant::now();
        // Lexical hits need no embedding at all.
        if let Some((results, kind)) = self.cache.lookup(query, None, k, now) {
            return Ok((results, Some(kind)));
        }
        let qvec = self
            .embeddings
            .embed_query(query)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        if let Some((results, kind)) = self.cache.lookup(query, Some(&qvec), k, now) {
            return Ok((results, Some(kind)));
        }
        let results = self.inner.retrieve_with_scores(query, k).await?;
        self.cache
            .insert(query, qvec, k, results.clone(), Instant::now());
        Ok((results, None))
    }
}

#[async_trait::async_trait]
impl RetrieverTrait for CachedRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let (results, _) = self.lookup_or_retrieve(query, k).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }

    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        let (results, _) = self.lookup_or_retrieve(query, k).await?;
        Ok(results)
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        // Corpus changed: previously cached results are potentially stale.
        self.cache.invalidate();
        self.inner.add_documents(documents).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn result(content: &str, score: f32) -> SearchResult {
        SearchResult {
            document: Document::new(content),
            score,
        }
    }

    /// Inner retriever counting calls and returning fixed results.
    struct CountingRetriever {
        calls: AtomicUsize,
        results: Vec<SearchResult>,
    }

    impl CountingRetriever {
        fn new(results: Vec<SearchResult>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                results,
            }
        }
    }

    #[async_trait]
    impl RetrieverTrait for CountingRetriever {
        async fn retrieve(&self, _query: &str, _k: usize) -> Result<Vec<Document>, RetrieverError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.results.iter().map(|r| r.document.clone()).collect())
        }
        async fn retrieve_with_scores(
            &self,
            _query: &str,
            _k: usize,
        ) -> Result<Vec<SearchResult>, RetrieverError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.results.clone())
        }
        async fn add_documents(&self, _documents: Vec<Document>) -> Result<(), RetrieverError> {
            Ok(())
        }
    }

    /// Identity embedding: unit vector on the axis named by the text ("a"/"b").
    struct AxisEmbeddings;

    #[async_trait]
    impl Embeddings for AxisEmbeddings {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, lc_embeddings::EmbeddingError> {
            self.embed_documents(&[text])
                .await
                .map(|mut v| v.pop().unwrap_or_default())
        }
        async fn embed_documents(
            &self,
            texts: &[&str],
        ) -> Result<Vec<Vec<f32>>, lc_embeddings::EmbeddingError> {
            texts
                .iter()
                .map(|t| {
                    if t.starts_with('a') {
                        Ok(vec![1.0, 0.0])
                    } else if t.starts_with('b') {
                        Ok(vec![0.0, 1.0])
                    } else {
                        Err(lc_embeddings::EmbeddingError::EmptyInput)
                    }
                })
                .collect()
        }
        fn dimension(&self) -> usize {
            2
        }
        fn model_name(&self) -> &str {
            "axis"
        }
    }

    #[test]
    fn cosine_similarity_basics() {
        use lc_embeddings::cosine_similarity;
        let a = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &a).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &[0.0, 1.0]).unwrap().abs() < 1e-6);
        assert!((cosine_similarity(&a, &[-1.0, 0.0]).unwrap() + 1.0).abs() < 1e-6);
        assert!(
            cosine_similarity(&a, &[]).is_err(),
            "length mismatch errors, never NaN"
        );
    }

    #[test]
    fn lexical_hit_requires_same_k() {
        let core = SemanticCacheCore::new(SemanticCacheConfig::new());
        let now = Instant::now();
        core.insert("q", vec![1.0, 0.0], 5, vec![result("doc", 0.9)], now);

        assert!(
            core.lookup("q", None, 5, now).is_some(),
            "exact (q, k) hits"
        );
        assert!(
            core.lookup("q", None, 10, now).is_none(),
            "different k must not be served from the k=5 result set"
        );
    }

    #[test]
    fn lexical_beats_semantic() {
        let core = SemanticCacheCore::new(SemanticCacheConfig::new());
        let now = Instant::now();
        core.insert(
            "a-query",
            vec![1.0, 0.0],
            5,
            vec![result("from-a", 1.0)],
            now,
        );
        core.insert(
            "a-query ",
            vec![1.0, 0.0],
            5,
            vec![result("from-a2", 0.9)],
            now,
        );

        let (results, kind) = core.lookup("a-query", Some(&[1.0, 0.0]), 5, now).unwrap();
        assert_eq!(kind, CacheHitKind::Lexical, "byte-identical wins");
        assert_eq!(results[0].document.content, "from-a");
    }

    #[test]
    fn semantic_hit_respects_threshold() {
        let config = SemanticCacheConfig::new().with_threshold(0.9);
        let core = SemanticCacheCore::new(config);
        let now = Instant::now();
        core.insert(
            "query a",
            vec![1.0, 0.0],
            5,
            vec![result("cached", 0.8)],
            now,
        );

        // Similarity 0.8 < 0.9 → miss.
        let n: f32 = (0.8f32 * 0.8 + 0.6 * 0.6).sqrt();
        let v = vec![0.8 / n, 0.6 / n];
        assert!(core.lookup("query b", Some(&v), 5, now).is_none());

        // Similarity 1.0 ≥ 0.9 → hit.
        let (results, kind) = core.lookup("query c", Some(&[1.0, 0.0]), 5, now).unwrap();
        assert_eq!(kind, CacheHitKind::Semantic);
        assert_eq!(results[0].document.content, "cached");
    }

    #[test]
    fn fifo_eviction_bounded() {
        let config = SemanticCacheConfig::new().with_max_entries(2);
        let core = SemanticCacheCore::new(config);
        let now = Instant::now();
        core.insert("q1", vec![1.0, 0.0], 5, vec![], now);
        core.insert("q2", vec![1.0, 0.0], 5, vec![], now);
        assert_eq!(core.len(), 2);
        core.insert("q3", vec![1.0, 0.0], 5, vec![], now);
        assert_eq!(core.len(), 2, "FIFO evicts the oldest");
        assert!(core.lookup("q1", None, 5, now).is_none(), "q1 evicted");
        assert!(core.lookup("q3", None, 5, now).is_some());
    }

    #[test]
    fn ttl_expiry_is_a_miss() {
        let config = SemanticCacheConfig::new().with_ttl(Some(Duration::from_millis(50)));
        let core = SemanticCacheCore::new(config);
        let now = Instant::now();
        core.insert("q", vec![1.0, 0.0], 5, vec![result("doc", 1.0)], now);

        assert!(core.lookup("q", None, 5, now).is_some());
        let later = now + Duration::from_millis(51);
        assert!(
            core.lookup("q", None, 5, later).is_none(),
            "expired entries are misses"
        );
    }

    #[test]
    fn insert_replaces_same_query_and_k() {
        let core = SemanticCacheCore::new(SemanticCacheConfig::new());
        let now = Instant::now();
        core.insert("q", vec![1.0], 5, vec![result("old", 1.0)], now);
        core.insert("q", vec![1.0], 5, vec![result("new", 1.0)], now);
        assert_eq!(core.len(), 1, "replace, not duplicate");
        let (results, _) = core.lookup("q", None, 5, now).unwrap();
        assert_eq!(results[0].document.content, "new");
    }

    #[test]
    fn invalidate_clears_all() {
        let core = SemanticCacheCore::new(SemanticCacheConfig::new());
        let now = Instant::now();
        core.insert("q", vec![1.0], 5, vec![], now);
        assert!(!core.is_empty());
        core.invalidate();
        assert!(core.is_empty());
        assert!(core.lookup("q", None, 5, now).is_none());
    }

    /// Decorator: identical query → 1 retrieval call; paraphrase above
    /// threshold → still cached (semantic hit); add_documents invalidates.
    #[tokio::test]
    async fn cached_retriever_skips_inner_on_hits() {
        let inner = Arc::new(CountingRetriever::new(vec![result("doc a", 0.9)]));
        let retriever = CachedRetriever::new(
            inner.clone(),
            Arc::new(AxisEmbeddings),
            SemanticCacheConfig::new(),
        );

        let first = retriever.retrieve("apple", 3).await.unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1, "miss → inner call");

        let second = retriever.retrieve("apple", 3).await.unwrap();
        assert_eq!(second[0].content, "doc a");
        assert_eq!(
            inner.calls.load(Ordering::SeqCst),
            1,
            "lexical hit → no call"
        );

        // "avocado" embeds to the same axis (starts with 'a') → semantic hit.
        let third = retriever.retrieve("avocado", 3).await.unwrap();
        assert_eq!(third[0].content, "doc a");
        assert_eq!(
            inner.calls.load(Ordering::SeqCst),
            1,
            "semantic hit → no call"
        );

        // Different axis → miss → second inner call.
        let _ = retriever.retrieve("banana", 3).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);

        // k is part of the cache identity: k change → miss → third call.
        let _ = retriever.retrieve("apple", 5).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 3);

        // add_documents invalidates: "apple" k=3 needs a fresh inner call.
        retriever
            .add_documents(vec![Document::new("new doc")])
            .await
            .unwrap();
        assert!(
            retriever.cache().is_empty(),
            "corpus update invalidates cache"
        );
        let _ = retriever.retrieve("apple", 3).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 4);
    }

    /// Unknown text (embeddings error) surfaces the error rather than caching.
    #[tokio::test]
    async fn embedding_error_propagates() {
        let inner = Arc::new(CountingRetriever::new(vec![]));
        let retriever =
            CachedRetriever::new(inner, Arc::new(AxisEmbeddings), SemanticCacheConfig::new());
        let err = retriever.retrieve("zebra", 3).await;
        assert!(err.is_err(), "embedding failure must not be swallowed");
    }
}
