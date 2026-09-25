// lc-memory/src/semantic.rs
//! B4 (v0.22.4): two-tier semantic memory — a unified [`MemoryStore`] abstraction
//! (namespaced key-value + semantic recall) with a short-/long-term split and
//! weighted-decay ranking.
//!
//! # Why a second memory abstraction
//!
//! [`crate::BaseMemory`] manages **conversation history** (messages injected into the
//! next prompt). The types in this module manage **knowledge**: durable facts about the
//! user / task extracted from completed turns, addressable by namespace (e.g. one per
//! user or session) and retrievable by meaning rather than by recency in a transcript.
//!
//! # The two tiers
//!
//! - [`ShortTermMemory`] is the in-thread working tier: bounded per namespace
//!   (`capacity`, FIFO eviction), exact key-value lookups plus semantic search purely by
//!   similarity. Nothing is persisted; a dropped store is a forgotten store.
//! - [`LongTermMemory`] is unbounded and ranks candidates the way generative-agent
//!   systems do: `score = w_similarity · sim + w_recency · 2^(−age/half_life) +
//!   w_importance · importance` ([`DecayWeights`]). Frequently re-accessed memories
//!   stay fresh; old, unimportant ones naturally sink — nothing is deleted on the read
//!   path.
//! - [`TwoTierMemory`] wires the two together: writes land in the short tier;
//!   [`consolidate`](TwoTierMemory::consolidate) promotes entries that clear the
//!   [`PromotionPolicy`] (high importance **or** enough re-accesses). Recall searches
//!   both tiers with one uniform weighted formula and de-duplicates by key
//!   (short tier wins on collision).
//!
//! # Semantics without a mandatory embedding dependency
//!
//! [`SemanticScorer`] is a pluggable trait; the always-available [`LexicalScorer`]
//! ranks by cosine over term-frequency vectors (Unicode-aware tokenization), so the
//! whole abstraction and its tests run offline with zero extra dependencies. An
//! embedding-backed scorer can be supplied with `with_scorer` without touching the
//! stores.
//!
//! All time-dependent logic takes an injected clock, so decay/consolidation tests are
//! pure — no sleeps.

use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::MemoryError;

/// Injectable wall clock (defaults to [`SystemTime::now`]).
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

fn real_clock() -> Clock {
    Arc::new(SystemTime::now)
}

// ───────────────────────────── data model ─────────────────────────────

/// A storable unit of knowledge.
///
/// `importance` is clamped into `[0, 1]` by the stores; `key` must be stable within a
/// namespace (re-putting the same key updates the entry).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryItem {
    /// Stable identifier within the namespace.
    pub key: String,
    /// The memory text, matched by semantic search.
    pub text: String,
    /// Importance in `[0, 1]` (higher resists decay and promotes sooner).
    pub importance: f64,
    /// Arbitrary structured annotations (user id, source turn, …).
    pub metadata: HashMap<String, String>,
}

impl MemoryItem {
    /// Creates a memory with default importance `0.5` and no metadata.
    pub fn new(key: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            text: text.into(),
            importance: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Sets the importance (clamped to `[0, 1]` at write time anyway).
    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Adds one metadata annotation.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replaces the whole metadata map.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A memory with its lifecycle bookkeeping.
#[derive(Debug, Clone)]
pub struct StoredMemory {
    /// The stored item.
    pub item: MemoryItem,
    /// First write time.
    pub created_at: SystemTime,
    /// Most recent access (put/get/search hit).
    pub last_access_at: SystemTime,
    /// Number of accesses since the entry was created.
    pub access_count: u64,
}

impl StoredMemory {
    fn fresh(item: MemoryItem, now: SystemTime) -> Self {
        Self {
            item,
            created_at: now,
            last_access_at: now,
            access_count: 1,
        }
    }

    fn touch(&mut self, now: SystemTime) {
        self.last_access_at = now;
        self.access_count = self.access_count.saturating_add(1);
    }
}

/// Which tier a recall hit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// The bounded in-thread working tier.
    Short,
    /// The decayed long-term tier.
    Long,
}

/// A recall result.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryHit {
    /// Stable key within the namespace.
    pub key: String,
    /// The memory text.
    pub text: String,
    /// Final ranking score in `[0, 1]`.
    pub score: f64,
    /// Importance in `[0, 1]`.
    pub importance: f64,
    /// Originating tier.
    pub tier: MemoryTier,
    /// Entry metadata.
    pub metadata: HashMap<String, String>,
}

impl MemoryHit {
    fn from_stored(stored: &StoredMemory, score: f64, tier: MemoryTier) -> Self {
        Self {
            key: stored.item.key.clone(),
            text: stored.item.text.clone(),
            score,
            importance: stored.item.importance,
            tier,
            metadata: stored.item.metadata.clone(),
        }
    }
}

/// A semantic recall query against one namespace.
#[derive(Debug, Clone)]
pub struct MemoryQuery<'q> {
    /// Namespace to search (isolated from every other namespace).
    pub namespace: &'q str,
    /// Query text.
    pub text: &'q str,
    /// Maximum hits to return.
    pub k: usize,
    /// Hits scoring below this are dropped.
    pub min_score: f64,
}

impl<'q> MemoryQuery<'q> {
    /// Creates a query with defaults `k = 5`, `min_score = 0.0`.
    pub fn new(namespace: &'q str, text: &'q str) -> Self {
        Self {
            namespace,
            text,
            k: 5,
            min_score: 0.0,
        }
    }

    /// Sets the maximum number of hits.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }

    /// Sets the minimum score filter.
    pub fn min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }
}

/// Namespaced key-value store with semantic recall.
///
/// Implementations are expected to be cheaply `Arc`-shareable (`Send + Sync`) and never
/// to panic on the read path; failures surface as [`MemoryError`].
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Writes/updates an item in the namespace. Re-putting an existing key refreshes
    /// its content and access time.
    async fn put(&self, namespace: &str, item: MemoryItem) -> Result<(), MemoryError>;

    /// Exact key lookup (`None` when absent). Counts as an access.
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryItem>, MemoryError>;

    /// Semantic recall, best score first, capped at `query.k`.
    async fn search(&self, query: &MemoryQuery<'_>) -> Result<Vec<MemoryHit>, MemoryError>;

    /// Removes one entry; returns whether it existed.
    async fn forget(&self, namespace: &str, key: &str) -> Result<bool, MemoryError>;

    /// Drops every entry in the namespace; returns the number removed.
    async fn clear_namespace(&self, namespace: &str) -> Result<usize, MemoryError>;

    /// Number of entries stored under the namespace.
    async fn len_namespace(&self, namespace: &str) -> Result<usize, MemoryError>;
}

fn validate(namespace: &str, item: &MemoryItem) -> Result<(), MemoryError> {
    if namespace.trim().is_empty() {
        return Err(MemoryError::Other(
            "memory namespace must not be empty".into(),
        ));
    }
    if item.key.trim().is_empty() {
        return Err(MemoryError::Other("memory key must not be empty".into()));
    }
    if item.text.trim().is_empty() {
        return Err(MemoryError::Other("memory text must not be empty".into()));
    }
    Ok(())
}

// ───────────────────────────── scoring ─────────────────────────────

/// Pluggable semantic similarity between two texts.
///
/// Implementations return a value in `[0, 1]` (1 = identical meaning). Embedding-based
/// scorers can back this trait; the default [`LexicalScorer`] needs no model.
#[async_trait]
pub trait SemanticScorer: Send + Sync {
    /// Similarity between `query` and `document` in `[0, 1]`.
    async fn similarity(&self, query: &str, document: &str) -> f64;
}

/// Dependency-free scorer: cosine similarity over Unicode word term-frequency vectors.
///
/// Deterministic and offline — the reference scorer for the working tier and tests.
#[derive(Debug, Default, Clone)]
pub struct LexicalScorer;

impl LexicalScorer {
    /// Creates the scorer.
    pub fn new() -> Self {
        Self
    }

    fn token_vector(text: &str) -> HashMap<String, f64> {
        let mut v: HashMap<String, f64> = HashMap::new();
        for token in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            *v.entry(token.to_lowercase()).or_insert(0.0) += 1.0;
        }
        v
    }

    /// Pure cosine over term-frequency vectors (exposed for direct testing).
    pub fn score(query: &str, document: &str) -> f64 {
        let a = Self::token_vector(query);
        let b = Self::token_vector(document);
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        // Iterate the smaller map.
        let (small, large) = if a.len() <= b.len() {
            (&a, &b)
        } else {
            (&b, &a)
        };
        let mut dot = 0.0;
        for (term, freq) in small {
            dot += freq * large.get(term).copied().unwrap_or(0.0);
        }
        let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
        let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

#[async_trait]
impl SemanticScorer for LexicalScorer {
    async fn similarity(&self, query: &str, document: &str) -> f64 {
        Self::score(query, document)
    }
}

// ───────────────────────── short-term tier ─────────────────────────

#[derive(Debug)]
struct ShortNamespace {
    entries: HashMap<String, StoredMemory>,
    /// Insertion order for FIFO eviction (one entry per present key).
    order: VecDeque<String>,
}

impl ShortNamespace {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }
}

/// Bounded in-thread working memory (one FIFO queue per namespace).
///
/// Recall ranks by raw scorer similarity. Cloning is cheap when shared via `Arc`.
pub struct ShortTermMemory {
    inner: Mutex<HashMap<String, ShortNamespace>>,
    capacity: usize,
    scorer: Arc<dyn SemanticScorer>,
    clock: Clock,
}

impl std::fmt::Debug for ShortTermMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShortTermMemory")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl ShortTermMemory {
    /// Creates a store with the given per-namespace capacity and the default
    /// [`LexicalScorer`].
    pub fn new(capacity: usize) -> Self {
        Self::with_scorer(capacity, Arc::new(LexicalScorer::new()))
    }

    /// Creates a store with a custom semantic scorer.
    pub fn with_scorer(capacity: usize, scorer: Arc<dyn SemanticScorer>) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            capacity: capacity.max(1),
            scorer,
            clock: real_clock(),
        }
    }

    /// Overrides the wall clock (tests).
    #[cfg(test)]
    fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// Per-namespace entry capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clones the entries of one namespace without holding the lock across await
    /// (the async scorer must not run under a std Mutex).
    fn snapshot(&self, namespace: &str) -> Vec<StoredMemory> {
        let inner = self.inner.lock().unwrap();
        inner
            .get(namespace)
            .map(|ns| ns.entries.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Refreshes last-access metadata for the given keys present in the namespace.
    fn touch(&self, namespace: &str, keys: &[String]) {
        let now = (self.clock)();
        let mut inner = self.inner.lock().unwrap();
        if let Some(ns) = inner.get_mut(namespace) {
            for key in keys {
                if let Some(stored) = ns.entries.get_mut(key) {
                    stored.touch(now);
                }
            }
        }
    }

    /// D7/L-me1: keys of one namespace (without cloning entry values).
    fn keys(&self, namespace: &str) -> Vec<String> {
        let inner = self.inner.lock().unwrap();
        inner
            .get(namespace)
            .map(|ns| ns.entries.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// D7/L-me1: atomically remove-and-return one entry. Differs from `forget`
    /// in that `consolidate` uses it to promote the *current* entry rather than
    /// a stale snapshot clone — eliminating the read-then-forget race where a
    /// concurrent re-put of a promoted key would be silently deleted.
    async fn take(
        &self,
        namespace: &str,
        key: &str,
    ) -> Result<Option<StoredMemory>, MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::Other(format!("short-term lock poisoned: {e}")))?;
        Ok(inner
            .get_mut(namespace)
            .and_then(|ns| ns.entries.remove(key)))
    }
}

#[async_trait]
impl MemoryStore for ShortTermMemory {
    async fn put(&self, namespace: &str, item: MemoryItem) -> Result<(), MemoryError> {
        validate(namespace, &item)?;
        let now = (self.clock)();
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("short-term lock poisoned: {e}")))?;
        let ns = inner
            .entry(namespace.to_string())
            .or_insert_with(ShortNamespace::new);
        let item = MemoryItem {
            importance: item.importance.clamp(0.0, 1.0),
            ..item
        };
        match ns.entries.get_mut(&item.key) {
            Some(existing) => {
                existing.item = item;
                existing.touch(now);
            }
            None => {
                if ns.entries.len() >= self.capacity {
                    if let Some(oldest) = ns.order.pop_front() {
                        ns.entries.remove(&oldest);
                    }
                }
                ns.order.push_back(item.key.clone());
                ns.entries
                    .insert(item.key.clone(), StoredMemory::fresh(item, now));
            }
        }
        Ok(())
    }

    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryItem>, MemoryError> {
        let now = (self.clock)();
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::LoadError(format!("short-term lock poisoned: {e}")))?;
        Ok(inner.get_mut(namespace).and_then(|ns| {
            ns.entries.get_mut(key).map(|stored| {
                stored.touch(now);
                stored.item.clone()
            })
        }))
    }

    async fn search(&self, query: &MemoryQuery<'_>) -> Result<Vec<MemoryHit>, MemoryError> {
        if query.namespace.trim().is_empty() {
            return Err(MemoryError::Other(
                "memory namespace must not be empty".into(),
            ));
        }
        let entries = self.snapshot(query.namespace);
        let mut scored: Vec<MemoryHit> = Vec::with_capacity(entries.len());
        for stored in &entries {
            let sim = self.scorer.similarity(query.text, &stored.item.text).await;
            if sim >= query.min_score {
                scored.push(MemoryHit::from_stored(stored, sim, MemoryTier::Short));
            }
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let hit_keys: Vec<String> = scored.iter().take(query.k).map(|h| h.key.clone()).collect();
        scored.truncate(query.k);
        self.touch(query.namespace, &hit_keys);
        Ok(scored)
    }

    async fn forget(&self, namespace: &str, key: &str) -> Result<bool, MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("short-term lock poisoned: {e}")))?;
        let removed = inner
            .get_mut(namespace)
            .map(|ns| {
                if ns.entries.remove(key).is_some() {
                    ns.order.retain(|k| k != key);
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false);
        Ok(removed)
    }

    async fn clear_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("short-term lock poisoned: {e}")))?;
        Ok(inner
            .remove(namespace)
            .map(|ns| ns.entries.len())
            .unwrap_or(0))
    }

    async fn len_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::LoadError(format!("short-term lock poisoned: {e}")))?;
        Ok(inner.get(namespace).map(|ns| ns.entries.len()).unwrap_or(0))
    }
}

// ───────────────────────── long-term tier ──────────────────────────

/// Weights and half-life of the long-term ranking formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayWeights {
    /// Weight of semantic similarity.
    pub similarity: f64,
    /// Weight of recency.
    pub recency: f64,
    /// Weight of importance.
    pub importance: f64,
    /// Age at which the recency term is 0.5.
    pub recency_half_life: Duration,
}

impl Default for DecayWeights {
    fn default() -> Self {
        Self {
            similarity: 0.7,
            recency: 0.15,
            importance: 0.15,
            recency_half_life: Duration::from_secs(7 * 24 * 3600),
        }
    }
}

impl DecayWeights {
    /// Creates weights with the default 0.7 / 0.15 / 0.15 mix and a 7-day half-life.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets all three mix weights (need not sum to 1; the score is a plain weighted sum).
    pub fn with_weights(mut self, similarity: f64, recency: f64, importance: f64) -> Self {
        self.similarity = similarity;
        self.recency = recency;
        self.importance = importance;
        self
    }

    /// Sets the recency half-life.
    pub fn with_half_life(mut self, half_life: Duration) -> Self {
        self.recency_half_life = half_life;
        self
    }

    /// Pure ranking score. `age` is time since last access; `similarity`/`importance`
    /// are in `[0, 1]`.
    pub fn score(&self, similarity: f64, importance: f64, age: Duration) -> f64 {
        let half = self.recency_half_life.as_secs_f64().max(f64::MIN_POSITIVE);
        let recency = (-age.as_secs_f64() / half * std::f64::consts::LN_2).exp();
        let raw =
            self.similarity * similarity + self.recency * recency + self.importance * importance;
        raw.clamp(0.0, 1.0)
    }
}

/// Default per-namespace entry ceiling for long-term memory (0.25.0 D4/M-me3).
const DEFAULT_LONG_TERM_CAPACITY: usize = 256;

/// Long-term memory with weighted-decay ranking, bounded by capacity + optional TTL.
pub struct LongTermMemory {
    inner: Mutex<HashMap<String, HashMap<String, StoredMemory>>>,
    weights: DecayWeights,
    scorer: Arc<dyn SemanticScorer>,
    clock: Clock,
    /// Max entries per namespace; `None` = unbounded (avoids only-rising growth).
    capacity: Option<usize>,
    /// Optional age-out: entries idle past this TTL are evicted.
    ttl: Option<Duration>,
}

impl std::fmt::Debug for LongTermMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LongTermMemory")
            .field("weights", &self.weights)
            .finish_non_exhaustive()
    }
}

impl LongTermMemory {
    /// Creates a store with default [`DecayWeights`] and [`LexicalScorer`].
    pub fn new() -> Self {
        Self::with_config(DecayWeights::default(), Arc::new(LexicalScorer::new()))
    }

    /// Creates a store with explicit weights and scorer.
    pub fn with_config(weights: DecayWeights, scorer: Arc<dyn SemanticScorer>) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            weights,
            scorer,
            clock: real_clock(),
            capacity: Some(DEFAULT_LONG_TERM_CAPACITY),
            ttl: None,
        }
    }

    /// Sets the per-namespace entry ceiling (FIFO-by-value eviction of the
    /// lowest-ranked entries above the cap). `None` disables the bound.
    pub fn with_capacity(mut self, capacity: Option<usize>) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets an idle TTL: entries not accessed within `ttl` are evicted.
    pub fn with_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }

    /// Overrides the wall clock (tests).
    #[cfg(test)]
    fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// Ranking configuration.
    pub fn weights(&self) -> &DecayWeights {
        &self.weights
    }

    /// Direct upsert used by [`TwoTierMemory`] promotion (merges with an existing entry).
    fn upsert(&self, namespace: &str, incoming: StoredMemory) -> Result<(), MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("long-term lock poisoned: {e}")))?;
        let map = inner.entry(namespace.to_string()).or_default();
        match map.get_mut(&incoming.item.key) {
            None => {
                map.insert(incoming.item.key.clone(), incoming);
            }
            Some(existing) => {
                // Merge: keep the older creation time, the freshest access, summed
                // accesses, the highest importance, and the newer text/metadata.
                existing.created_at = existing.created_at.min(incoming.created_at);
                existing.last_access_at = existing.last_access_at.max(incoming.last_access_at);
                existing.access_count = existing.access_count.saturating_add(incoming.access_count);
                existing.item.importance = existing.item.importance.max(incoming.item.importance);
                for (k, v) in incoming.item.metadata {
                    existing.item.metadata.insert(k, v);
                }
                existing.item.text = incoming.item.text;
            }
        }
        // D4/M-me3: promotion via upsert must trim the long tier too, otherwise
        // consolidate would only ever grow it ("只升不降").
        let now = (self.clock)();
        self.enforce_bounds(map, now);
        Ok(())
    }

    /// D4/M-me3: enforce the capacity ceiling and optional TTL on a namespace map.
    ///
    /// TTL removes entries idle past the deadline. Capacity evicts the
    /// lowest-value entries first — value is the same weighted decay score used
    /// by ranking (with a zero similarity term), so the bound preserves the
    /// highest-importance/most-recently-accessed memories.
    fn enforce_bounds(&self, map: &mut HashMap<String, StoredMemory>, now: SystemTime) {
        if let Some(ttl) = self.ttl {
            map.retain(|_k, st| {
                now.duration_since(st.last_access_at)
                    .map(|d| d <= ttl)
                    .unwrap_or(true)
            });
        }
        if let Some(cap) = self.capacity {
            while map.len() > cap {
                match map
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        let va = self.eviction_value(a, now);
                        let vb = self.eviction_value(b, now);
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k.clone())
                {
                    Some(k) => {
                        map.remove(&k);
                    }
                    None => break,
                }
            }
        }
    }

    /// Lower-bound value of one entry for capacity eviction.
    fn eviction_value(&self, stored: &StoredMemory, now: SystemTime) -> f64 {
        let age = now
            .duration_since(stored.last_access_at)
            .unwrap_or(Duration::ZERO);
        // Zero similarity term: rank purely on importance × recency × access
        // via the configured weights.
        self.weights.score(0.0, stored.item.importance, age)
    }

    fn snapshot(&self, namespace: &str) -> Vec<StoredMemory> {
        let inner = self.inner.lock().unwrap();
        inner
            .get(namespace)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    fn touch(&self, namespace: &str, keys: &[String]) {
        let now = (self.clock)();
        let mut inner = self.inner.lock().unwrap();
        if let Some(map) = inner.get_mut(namespace) {
            for key in keys {
                if let Some(stored) = map.get_mut(key) {
                    stored.touch(now);
                }
            }
        }
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for LongTermMemory {
    async fn put(&self, namespace: &str, item: MemoryItem) -> Result<(), MemoryError> {
        validate(namespace, &item)?;
        let now = (self.clock)();
        self.upsert(
            namespace,
            StoredMemory::fresh(
                MemoryItem {
                    importance: item.importance.clamp(0.0, 1.0),
                    ..item
                },
                now,
            ),
        )
    }

    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryItem>, MemoryError> {
        let now = (self.clock)();
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::LoadError(format!("long-term lock poisoned: {e}")))?;
        Ok(inner
            .get_mut(namespace)
            .and_then(|m| m.get_mut(key))
            .map(|stored| {
                stored.touch(now);
                stored.item.clone()
            }))
    }

    async fn search(&self, query: &MemoryQuery<'_>) -> Result<Vec<MemoryHit>, MemoryError> {
        if query.namespace.trim().is_empty() {
            return Err(MemoryError::Other(
                "memory namespace must not be empty".into(),
            ));
        }
        let now = (self.clock)();
        let entries = self.snapshot(query.namespace);
        let mut scored: Vec<MemoryHit> = Vec::with_capacity(entries.len());
        for stored in entries {
            let sim = self.scorer.similarity(query.text, &stored.item.text).await;
            let age = now
                .duration_since(stored.last_access_at)
                .unwrap_or(Duration::ZERO);
            let score = self.weights.score(sim, stored.item.importance, age);
            if score >= query.min_score {
                scored.push(MemoryHit::from_stored(&stored, score, MemoryTier::Long));
            }
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let hit_keys: Vec<String> = scored.iter().take(query.k).map(|h| h.key.clone()).collect();
        scored.truncate(query.k);
        self.touch(query.namespace, &hit_keys);
        Ok(scored)
    }

    async fn forget(&self, namespace: &str, key: &str) -> Result<bool, MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("long-term lock poisoned: {e}")))?;
        Ok(inner
            .get_mut(namespace)
            .map(|m| m.remove(key).is_some())
            .unwrap_or(false))
    }

    async fn clear_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::SaveError(format!("long-term lock poisoned: {e}")))?;
        Ok(inner.remove(namespace).map(|m| m.len()).unwrap_or(0))
    }

    async fn len_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| MemoryError::LoadError(format!("long-term lock poisoned: {e}")))?;
        Ok(inner.get(namespace).map(|m| m.len()).unwrap_or(0))
    }
}

// ─────────────────────── promotion / two tiers ──────────────────────

/// Policy deciding which short-term entries consolidate into long-term memory.
///
/// An entry qualifies when **either** condition holds: importance is at least
/// `min_importance`, or it has been accessed at least `min_access_count` times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionPolicy {
    /// Importance floor for immediate promotion.
    pub min_importance: f64,
    /// Re-access threshold for promotion of merely-useful memories.
    pub min_access_count: u64,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            min_importance: 0.8,
            min_access_count: 3,
        }
    }
}

impl PromotionPolicy {
    /// Creates the default policy (importance ≥ 0.8 or ≥ 3 accesses).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the importance floor.
    pub fn with_min_importance(mut self, min_importance: f64) -> Self {
        self.min_importance = min_importance.clamp(0.0, 1.0);
        self
    }

    /// Sets the re-access threshold.
    pub fn with_min_access_count(mut self, min_access_count: u64) -> Self {
        self.min_access_count = min_access_count;
        self
    }

    /// Pure predicate over a stored entry.
    pub fn qualifies(&self, stored: &StoredMemory) -> bool {
        stored.item.importance >= self.min_importance
            || stored.access_count >= self.min_access_count
    }
}

/// Two-tier memory: bounded short-term working store + decayed long-term store.
///
/// - Writes ([`put`](MemoryStore::put)) always land in the short tier.
/// - [`get`](MemoryStore::get) checks the short tier first, then the long tier.
/// - [`search`](MemoryStore::search) scores **both** tiers with the long-term weighted
///   formula (so cross-tier ordering is on one scale), de-duplicating by key with the
///   short tier preferred.
/// - [`consolidate`](TwoTierMemory::consolidate) promotes qualifying short-term entries
///   into long-term memory and removes them from the short tier.
pub struct TwoTierMemory {
    short: Arc<ShortTermMemory>,
    long: Arc<LongTermMemory>,
    policy: Mutex<PromotionPolicy>,
    scorer: Arc<dyn SemanticScorer>,
    weights: DecayWeights,
    clock: Clock,
}

impl std::fmt::Debug for TwoTierMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoTierMemory")
            .field("policy", &self.policy)
            .field("weights", &self.weights)
            .finish_non_exhaustive()
    }
}

impl TwoTierMemory {
    /// Creates a two-tier store with default capacity/weights/scorer/policy.
    pub fn new(short_capacity: usize) -> Self {
        let scorer: Arc<dyn SemanticScorer> = Arc::new(LexicalScorer::new());
        let weights = DecayWeights::default();
        Self {
            short: Arc::new(ShortTermMemory::with_scorer(short_capacity, scorer.clone())),
            long: Arc::new(LongTermMemory::with_config(weights, scorer.clone())),
            policy: Mutex::new(PromotionPolicy::default()),
            scorer,
            weights,
            clock: real_clock(),
        }
    }

    /// Replaces the promotion policy.
    pub fn with_policy(self, policy: PromotionPolicy) -> Self {
        *self.policy.lock().unwrap() = policy;
        self
    }

    /// Overrides the wall clock on both tiers and the unified ranking.
    ///
    /// Test-only constructor: call **before** seeding data — the tiers are rebuilt, so
    /// previously stored entries are lost.
    #[cfg(test)]
    pub(crate) fn with_clock(mut self, clock: Clock) -> Self {
        let capacity = self.short.capacity();
        self.short = Arc::new(
            ShortTermMemory::with_scorer(capacity, self.scorer.clone()).with_clock(clock.clone()),
        );
        self.long = Arc::new(
            LongTermMemory::with_config(self.weights, self.scorer.clone())
                .with_clock(clock.clone()),
        );
        self.clock = clock;
        self
    }

    /// The short-term tier (direct access for seeding/tests).
    pub fn short_term(&self) -> Arc<ShortTermMemory> {
        self.short.clone()
    }

    /// The long-term tier (direct access for seeding/tests).
    pub fn long_term(&self) -> Arc<LongTermMemory> {
        self.long.clone()
    }

    /// Replaces the promotion policy at runtime.
    pub fn set_policy(&self, policy: PromotionPolicy) {
        *self.policy.lock().unwrap() = policy;
    }

    /// Promotes qualifying short-term entries of one namespace into long-term memory,
    /// removing promoted entries from the short tier. Returns the promoted keys.
    pub async fn consolidate_namespace(&self, namespace: &str) -> Result<Vec<String>, MemoryError> {
        let policy = *self
            .policy
            .lock()
            .map_err(|e| MemoryError::Other(format!("promotion policy lock poisoned: {e}")))?;
        // D7/L-me1: key-driven promotion via atomic take-then-decide. We advance
        // by key (never a whole-namespace snapshot clone), taking the *current*
        // entry atomically so a concurrent re-put cannot be deleted by a stale
        // forget. Non-qualifying entries are restored.
        let keys = self.short.keys(namespace);
        let mut promoted = Vec::new();
        for key in keys {
            let Some(stored) = self.short.take(namespace, &key).await? else {
                // Already taken/removed concurrently — nothing to promote.
                continue;
            };
            if policy.qualifies(&stored) {
                self.long.upsert(namespace, stored)?;
                promoted.push(key);
            } else {
                // Not qualified: put it back (it was only removed to make the
                // promotion decision atomic against concurrent re-puts).
                self.short.put(namespace, stored.item).await?;
            }
        }
        Ok(promoted)
    }

    /// Promotes qualifying entries across every short-term namespace.
    pub async fn consolidate(&self) -> Result<Vec<String>, MemoryError> {
        let namespaces: Vec<String> = self.short.inner.lock().unwrap().keys().cloned().collect();
        let mut all = Vec::new();
        for namespace in namespaces {
            all.extend(self.consolidate_namespace(&namespace).await?);
        }
        Ok(all)
    }

    /// Unified weighted score for cross-tier ranking (same scale for every candidate).
    async fn rank_one(&self, stored: &StoredMemory, query: &str, now: SystemTime) -> f64 {
        let sim = self.scorer.similarity(query, &stored.item.text).await;
        let age = now
            .duration_since(stored.last_access_at)
            .unwrap_or(Duration::ZERO);
        self.weights.score(sim, stored.item.importance, age)
    }
}

#[async_trait]
impl MemoryStore for TwoTierMemory {
    async fn put(&self, namespace: &str, item: MemoryItem) -> Result<(), MemoryError> {
        self.short.put(namespace, item).await
    }

    async fn get(&self, namespace: &str, key: &str) -> Result<Option<MemoryItem>, MemoryError> {
        if let Some(item) = self.short.get(namespace, key).await? {
            return Ok(Some(item));
        }
        self.long.get(namespace, key).await
    }

    async fn search(&self, query: &MemoryQuery<'_>) -> Result<Vec<MemoryHit>, MemoryError> {
        if query.namespace.trim().is_empty() {
            return Err(MemoryError::Other(
                "memory namespace must not be empty".into(),
            ));
        }
        let now = (self.clock)();

        // Snapshot both tiers, tag the tier, and score on one uniform scale.
        let mut candidates: Vec<(StoredMemory, MemoryTier)> = self
            .short
            .snapshot(query.namespace)
            .into_iter()
            .map(|s| (s, MemoryTier::Short))
            .collect();
        candidates.extend(
            self.long
                .snapshot(query.namespace)
                .into_iter()
                .map(|s| (s, MemoryTier::Long)),
        );

        let mut hits: Vec<MemoryHit> = Vec::with_capacity(candidates.len());
        for (stored, tier) in &candidates {
            let score = self.rank_one(stored, query.text, now).await;
            if score >= query.min_score {
                hits.push(MemoryHit::from_stored(stored, score, *tier));
            }
        }

        // De-duplicate by key: short tier wins (it holds the freshest write).
        let mut seen = std::collections::HashSet::new();
        hits.retain(|h| seen.insert(h.key.clone()));

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(query.k);

        // Refresh access metadata in the tier each surviving hit came from.
        let mut short_keys = Vec::new();
        let mut long_keys = Vec::new();
        for hit in &hits {
            match hit.tier {
                MemoryTier::Short => short_keys.push(hit.key.clone()),
                MemoryTier::Long => long_keys.push(hit.key.clone()),
            }
        }
        self.short.touch(query.namespace, &short_keys);
        self.long.touch(query.namespace, &long_keys);
        Ok(hits)
    }

    async fn forget(&self, namespace: &str, key: &str) -> Result<bool, MemoryError> {
        let in_short = self.short.forget(namespace, key).await?;
        let in_long = self.long.forget(namespace, key).await?;
        Ok(in_short || in_long)
    }

    async fn clear_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        let s = self.short.clear_namespace(namespace).await?;
        let l = self.long.clear_namespace(namespace).await?;
        Ok(s + l)
    }

    async fn len_namespace(&self, namespace: &str) -> Result<usize, MemoryError> {
        Ok(self.short.len_namespace(namespace).await? + self.long.len_namespace(namespace).await?)
    }
}

// ─────────────────────────── extraction ─────────────────────────────

/// Turns a completed conversation turn into durable memories.
///
/// Implementations call an LLM, apply heuristics, or simply filter; returning an empty
/// vector means "nothing worth remembering from this turn". The agent executor runs
/// this on a detached task so a slow extractor never blocks the control loop.
#[async_trait]
pub trait MemoryExtractor: Send + Sync {
    /// Extracts zero or more memories from one user/assistant exchange.
    async fn extract(
        &self,
        namespace: &str,
        user_input: &str,
        assistant_output: &str,
    ) -> Result<Vec<MemoryItem>, MemoryError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock_at(secs: u64) -> Clock {
        Arc::new(move || std::time::UNIX_EPOCH + Duration::from_secs(secs))
    }

    fn item(key: &str, text: &str, importance: f64) -> MemoryItem {
        MemoryItem::new(key, text).with_importance(importance)
    }

    #[tokio::test]
    async fn lexical_scorer_ranks_shared_terms_first() {
        assert!(LexicalScorer::score("rust memory decay", "rust memory decay") > 0.99);
        let exact =
            LexicalScorer::score("rust agent framework", "the rust agent framework is fast");
        let unrelated = LexicalScorer::score("rust agent framework", "banana bread recipe sunday");
        assert!(exact > unrelated);
        assert_eq!(LexicalScorer::score("", "anything"), 0.0);
    }

    #[tokio::test]
    async fn namespace_isolation_covers_get_search_forget_and_clear() {
        let store = ShortTermMemory::new(10);
        store
            .put("a", item("k1", "shared secret alpha", 0.5))
            .await
            .unwrap();
        store
            .put("b", item("k1", "shared secret beta", 0.5))
            .await
            .unwrap();

        // Exact KV is namespaced.
        assert_eq!(
            store.get("a", "k1").await.unwrap().unwrap().text,
            "shared secret alpha"
        );
        assert_eq!(store.len_namespace("a").await.unwrap(), 1);
        assert_eq!(store.len_namespace("b").await.unwrap(), 1);
        assert_eq!(store.len_namespace("c").await.unwrap(), 0);

        // Semantic recall never crosses namespaces.
        let hits_a = store
            .search(&MemoryQuery::new("a", "secret alpha").k(5))
            .await
            .unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_a[0].text, "shared secret alpha");
        assert!(store
            .search(&MemoryQuery::new("c", "secret"))
            .await
            .unwrap()
            .is_empty());

        // Forget / clear stay inside the namespace.
        assert!(store.forget("a", "k1").await.unwrap());
        assert!(!store.forget("a", "k1").await.unwrap());
        assert_eq!(store.len_namespace("b").await.unwrap(), 1);
        assert_eq!(store.clear_namespace("b").await.unwrap(), 1);
        assert_eq!(store.len_namespace("b").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn short_term_validates_inputs_and_evicts_fifo() {
        let store = ShortTermMemory::new(2);
        assert!(store.put("ns", MemoryItem::new("", "x")).await.is_err());
        assert!(store.put("ns", MemoryItem::new("k", " ")).await.is_err());
        assert!(store.put("", MemoryItem::new("k", "x")).await.is_err());

        store
            .put("ns", item("first", "first entry text", 0.5))
            .await
            .unwrap();
        store
            .put("ns", item("second", "second entry text", 0.5))
            .await
            .unwrap();
        store
            .put("ns", item("third", "third entry text", 0.5))
            .await
            .unwrap();
        assert_eq!(store.len_namespace("ns").await.unwrap(), 2);
        assert!(store.get("ns", "first").await.unwrap().is_none());
        assert!(store.get("ns", "second").await.unwrap().is_some());
        assert!(store.get("ns", "third").await.unwrap().is_some());

        // Re-putting an existing key does not consume an extra slot.
        store
            .put("ns", item("second", "second updated", 0.9))
            .await
            .unwrap();
        assert_eq!(store.len_namespace("ns").await.unwrap(), 2);
        assert_eq!(
            store.get("ns", "second").await.unwrap().unwrap().importance,
            0.9
        );
    }

    #[tokio::test]
    async fn get_counts_as_access_for_promotion() {
        let store = ShortTermMemory::new(10);
        store
            .put("ns", item("k", "watched fact", 0.1))
            .await
            .unwrap();
        store.get("ns", "k").await.unwrap();
        store.get("ns", "k").await.unwrap();
        // put itself counts as access 1 → two more gets reach 3.
        let stored = store.snapshot("ns").pop().unwrap();
        assert_eq!(stored.access_count, 3);
    }

    /// Mutable clock handle shared between the test and the store.
    fn moving_clock(secs: Arc<Mutex<u64>>) -> Clock {
        Arc::new(move || std::time::UNIX_EPOCH + Duration::from_secs(*secs.lock().unwrap()))
    }

    #[tokio::test]
    async fn long_term_decay_rewards_recency_and_importance() {
        // Pure formula: identical similarity, recency and importance move the score.
        let w = DecayWeights::default().with_half_life(Duration::from_secs(10));
        let fresh = w.score(1.0, 0.5, Duration::from_secs(0));
        let stale = w.score(1.0, 0.5, Duration::from_secs(30));
        assert!(fresh > stale);
        let important_stale = w.score(1.0, 1.0, Duration::from_secs(30));
        assert!(important_stale > stale);
        // At one half-life the recency term contributes half its weight.
        let half = w.score(0.0, 0.0, Duration::from_secs(10));
        assert!((half - 0.15 * 0.5).abs() < 1e-9);

        // End-to-end: same text (identical similarity) written at t=0 and t=100, queried
        // at t=100 — the newer entry must outrank the stale one.
        let t = Arc::new(Mutex::new(0u64));
        let long = LongTermMemory::with_config(
            DecayWeights::default().with_half_life(Duration::from_secs(10)),
            Arc::new(LexicalScorer::new()),
        )
        .with_clock(moving_clock(t.clone()));
        long.put("ns", item("old", "same fact wording", 0.5))
            .await
            .unwrap();
        *t.lock().unwrap() = 100;
        long.put("ns", item("new", "same fact wording", 0.5))
            .await
            .unwrap();

        let hits = long
            .search(&MemoryQuery::new("ns", "same fact wording").k(5))
            .await
            .unwrap();
        assert_eq!(hits[0].key, "new");
        assert_eq!(hits[0].tier, MemoryTier::Long);
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn consolidation_promotes_by_importance_or_access_and_merges() {
        let mem = TwoTierMemory::new(10).with_clock(clock_at(0));
        mem.put("ns", item("hot", "important fact", 0.95))
            .await
            .unwrap();
        mem.put("ns", item("warm", "reaccessed fact", 0.2))
            .await
            .unwrap();
        mem.put("ns", item("cold", "ignored fact", 0.2))
            .await
            .unwrap();
        // Warm earns two re-accesses (put = 1, total 3 → threshold).
        mem.get("ns", "warm").await.unwrap();
        mem.get("ns", "warm").await.unwrap();

        let promoted = mem.consolidate_namespace("ns").await.unwrap();
        assert!(promoted.contains(&"hot".to_string()));
        assert!(promoted.contains(&"warm".to_string()));
        assert!(!promoted.contains(&"cold".to_string()));
        assert_eq!(promoted.len(), 2);

        // Promoted entries left the short tier and live in long-term.
        assert!(mem.short_term().get("ns", "hot").await.unwrap().is_none());
        assert_eq!(mem.long_term().len_namespace("ns").await.unwrap(), 2);
        assert_eq!(mem.short_term().len_namespace("ns").await.unwrap(), 1);
        // get() still resolves promoted entries through the long tier.
        assert!(mem.get("ns", "hot").await.unwrap().is_some());

        // Re-promotion merges rather than duplicating: same key returns from short with
        // a lower importance and accumulated accesses.
        mem.put("ns", item("hot", "important fact refined", 0.3))
            .await
            .unwrap();
        mem.short_term().get("ns", "hot").await.unwrap();
        mem.short_term().get("ns", "hot").await.unwrap();
        let again = mem.consolidate_namespace("ns").await.unwrap();
        assert_eq!(again, vec!["hot".to_string()]);
        let merged = mem.long_term().get("ns", "hot").await.unwrap().unwrap();
        assert_eq!(merged.text, "important fact refined");
        assert_eq!(merged.importance, 0.95); // max kept
        assert_eq!(mem.long_term().len_namespace("ns").await.unwrap(), 2);
    }

    #[tokio::test]
    async fn two_tier_search_merges_dedupes_and_ranks_on_one_scale() {
        let mem = TwoTierMemory::new(10)
            .with_policy(PromotionPolicy::default().with_min_importance(0.0))
            .with_clock(clock_at(0));
        mem.put("ns", item("short_only", "alpha distinctive tokens", 0.5))
            .await
            .unwrap();
        mem.put("ns", item("both", "gamma shared wording here", 0.5))
            .await
            .unwrap();
        mem.consolidate_namespace("ns").await.unwrap(); // both → long
                                                        // Re-write "both" into short so it exists in both tiers; add a long-only item.
        mem.put("ns", item("both", "gamma shared wording here fresher", 0.5))
            .await
            .unwrap();
        mem.long_term()
            .put("ns", item("long_only", "beta another memory", 0.5))
            .await
            .unwrap();

        let hits = mem
            .search(
                &MemoryQuery::new("ns", "gamma shared wording")
                    .k(10)
                    .min_score(0.3),
            )
            .await
            .unwrap();
        let keys: Vec<&str> = hits.iter().map(|h| h.key.as_str()).collect();
        assert!(keys.contains(&"both"));
        // A zero-similarity memory (importance-only score 0.075) is filtered out.
        assert!(!keys.contains(&"short_only"));
        assert!(!keys.contains(&"long_only"));
        // No duplicate key from the two tiers.
        assert_eq!(keys.iter().filter(|k| **k == "both").count(), 1);
        // The freshest "both" wins over the stale long-only match.
        assert_eq!(hits[0].key, "both");

        // forget clears whichever tier holds the key; "both" lives in both tiers and
        // is removed from both.
        assert!(mem.forget("ns", "long_only").await.unwrap());
        assert!(mem.forget("ns", "both").await.unwrap());
        // long still holds short_only; short is now empty.
        assert_eq!(mem.len_namespace("ns").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn empty_namespace_and_query_validation() {
        let mem = TwoTierMemory::new(4);
        assert!(mem.search(&MemoryQuery::new(" ", "x")).await.is_err());
        assert!(mem.put(" ", item("k", "v", 0.5)).await.is_err());
    }
}
