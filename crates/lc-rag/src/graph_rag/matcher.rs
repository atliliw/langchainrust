// src/retrieval/graph_rag/matcher.rs
//! Entity matching strategies for GraphRAG local queries.
//!
//! Provides the [`EntityMatcher`] trait and two implementations:
//! - [`KeywordMatcher`]: matches entities by keyword substring (default, zero-cost)
//! - [`EmbeddingMatcher`]: matches entities by embedding cosine similarity

use super::graph_store::GraphStore;
use crate::graph_rag::GraphRAGError;
use crate::hybrid::filter_by_score;
use lc_core::math::cosine_similarity;
use lc_embeddings::Embeddings;
use std::collections::{HashMap, HashSet};

/// Trait for finding relevant entities in a graph store given a query.
///
/// Implementations can use different matching strategies (keyword, embedding,
/// hybrid, etc.). The default is [`KeywordMatcher`].
pub trait EntityMatcher: Send + Sync {
    /// Find entity IDs relevant to the query, returning at most `top_k` results.
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// KeywordMatcher
// ---------------------------------------------------------------------------

/// Query term source kind: P2-4 uses this to apply different decay weights to hits from
/// different sources.
#[derive(Debug, Clone, Copy)]
enum TermKind {
    /// Direct query-term hit (weight 1.0).
    Direct,
    /// Synonym-expansion hit (decayed by `synonym_weight`).
    Synonym,
    /// CJK bigram hit (decayed by `cjk_bigram_weight`).
    Bigram,
}

/// A query term to match: text + source kind.
struct Term {
    text: String,
    kind: TermKind,
}

/// Chinese-English mixed normalization (P2-4): full-width characters are converted to
/// half-width (full-width ASCII differs from half-width by 0xFEE0), so the full-width form
/// of "Rust" normalizes to "rust", consistent with the lowercased entity names.
fn normalize_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            let u = c as u32;
            if (0xFF01..=0xFF5E).contains(&u) {
                char::from_u32(u - 0xFEE0).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Whether the character is a CJK ideograph (Chinese/Japanese kanji, etc.).
fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}'
    )
}

/// Chinese has no spaces, so adjacent bigrams of CJK characters recover recall (e.g. a long
/// Chinese query splits into character pairs), letting long Chinese queries hit short entity
/// names.
fn cjk_bigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| w.iter().collect()).collect()
}

/// Matches entities by keyword substring search.
///
/// This is the default matcher used by GraphRAG. It splits the query into
/// keywords and scores each entity based on how many keywords match the
/// entity's name, type, and description. Name matches are weighted highest.
///
/// P2-4: on top of the fixed name+3/type+2/desc+1 weights, three improvements fix the
/// arbitrary weights and the recall gaps of substring matching for synonyms, polysemes, and
/// Chinese-English mixed text:
/// - **Synonym-table expansion** `synonyms`: when a query term hits a synonym key, the
///   equivalent words are matched too (each hit decayed by `synonym_weight`, default 0.7).
/// - **Chinese-English mixed normalization**: full-width -> half-width plus splitting long
///   Chinese queries into CJK bigrams, fixing the problem that a space-free Chinese single
///   token cannot hit a short entity name.
/// - **TF-IDF weighting**: each query term is weighted by its inverse document frequency in
///   the entity corpus; common words (e.g. "Technology") discriminate little and contribute
///   little, while rare words contribute more; `use_tfidf` can disable it.
pub struct KeywordMatcher {
    /// Weight for name matches (default: 3).
    pub name_weight: usize,
    /// Weight for type matches (default: 2).
    pub type_weight: usize,
    /// Weight for description matches (default: 1).
    pub desc_weight: usize,
    /// Synonym table: query term (normalized lowercase/half-width form) -> list of equivalent
    /// words, also in normalized form. When an equivalent word is hit, it is matched once more
    /// with the contribution multiplied by `synonym_weight`.
    pub synonyms: HashMap<String, Vec<String>>,
    /// Whether TF-IDF weighting is enabled (default true). When disabled, falls back to the
    /// fixed weights.
    pub use_tfidf: bool,
    /// Decay factor for synonym hits (default 0.7).
    pub synonym_weight: f64,
    /// Decay factor for CJK bigram hits (default 0.5).
    pub cjk_bigram_weight: f64,
}

impl Default for KeywordMatcher {
    fn default() -> Self {
        Self {
            name_weight: 3,
            type_weight: 2,
            desc_weight: 1,
            synonyms: HashMap::new(),
            use_tfidf: true,
            synonym_weight: 0.7,
            cjk_bigram_weight: 0.5,
        }
    }
}

impl KeywordMatcher {
    /// Creates a new keyword matcher with default weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures the synonym table (query term -> equivalent word list).
    pub fn with_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        self.synonyms = synonyms;
        self
    }

    /// Toggles TF-IDF weighting (enabled by default).
    pub fn with_tfidf(mut self, enabled: bool) -> Self {
        self.use_tfidf = enabled;
        self
    }

    /// Splits the query into a term sequence (P2-4): direct terms + synonym expansion +
    /// CJK bigrams, deduplicated by text.
    fn build_terms(&self, query: &str) -> Vec<Term> {
        let normalized = normalize_text(query).to_lowercase();
        let tokens: Vec<String> = normalized
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let mut terms: Vec<Term> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tok in tokens {
            // Direct terms come first, avoiding a synonym identical to a direct term being
            // misclassified as a synonym.
            Self::push_term(&mut terms, &mut seen, tok.clone(), TermKind::Direct);

            // Synonym expansion: keys are normalized too, tolerating full-width/case in user keys.
            let syns = self
                .synonyms
                .iter()
                .find(|(k, _)| normalize_text(k).to_lowercase() == tok)
                .map(|(_, v)| v);
            if let Some(syns) = syns {
                for syn in syns {
                    Self::push_term(&mut terms, &mut seen, syn.clone(), TermKind::Synonym);
                }
            }

            // Chinese has no spaces; split long queries into CJK bigrams to recover recall.
            if tok.chars().any(is_cjk) {
                for bg in cjk_bigrams(&tok) {
                    Self::push_term(&mut terms, &mut seen, bg, TermKind::Bigram);
                }
            }
        }
        terms
    }

    fn push_term(terms: &mut Vec<Term>, seen: &mut HashSet<String>, text: String, kind: TermKind) {
        if seen.insert(text.clone()) {
            terms.push(Term { text, kind });
        }
    }

    /// Computes a smoothed IDF for each query term over the entity corpus (P2-4).
    ///
    /// `idf = ln((N+1)/(df+1)) + 1`, where df = the number of entities containing the term.
    /// Common terms have a large df and small IDF; rare terms have a large IDF. The smoothing
    /// term keeps df == N from zeroing out.
    fn compute_idf(&self, terms: &[Term], store: &GraphStore) -> HashMap<String, f64> {
        let n = store.all_entities().len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for term in terms {
            df.entry(term.text.clone()).or_insert(0);
        }
        for entity in store.all_entities().values() {
            let name = normalize_text(&entity.name).to_lowercase();
            let desc = normalize_text(&entity.description).to_lowercase();
            let typ = normalize_text(&entity.entity_type).to_lowercase();
            for term in terms {
                if name.contains(&term.text)
                    || desc.contains(&term.text)
                    || typ.contains(&term.text)
                {
                    if let Some(c) = df.get_mut(&term.text) {
                        *c += 1;
                    }
                }
            }
        }
        let mut idf = HashMap::with_capacity(terms.len());
        for (text, count) in &df {
            let w = ((n + 1.0) / (*count as f64 + 1.0)).ln() + 1.0;
            idf.insert(text.clone(), w);
        }
        idf
    }

    /// Score of a single query term against a single entity: field weight x TF-IDF weight x
    /// source decay.
    fn match_score(
        &self,
        term: &Term,
        name: &str,
        type_name: &str,
        desc: &str,
        idf_weight: f64,
    ) -> f64 {
        let mut score = 0.0f64;
        if name.contains(&term.text) {
            score += self.name_weight as f64 * idf_weight;
        }
        if type_name.contains(&term.text) {
            score += self.type_weight as f64 * idf_weight;
        }
        if desc.contains(&term.text) {
            score += self.desc_weight as f64 * idf_weight;
        }
        match term.kind {
            TermKind::Direct => score,
            TermKind::Synonym => score * self.synonym_weight,
            TermKind::Bigram => score * self.cjk_bigram_weight,
        }
    }
}

impl EntityMatcher for KeywordMatcher {
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String> {
        let terms = self.build_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }

        // P2-4: precompute TF-IDF; each query term is weighted by its corpus inverse document
        // frequency.
        let idf = if self.use_tfidf {
            Some(self.compute_idf(&terms, store))
        } else {
            None
        };

        let mut scored: Vec<(String, f64)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let name_lower = normalize_text(&entity.name).to_lowercase();
            let desc_lower = normalize_text(&entity.description).to_lowercase();
            let type_lower = normalize_text(&entity.entity_type).to_lowercase();

            let mut score = 0.0f64;
            for term in &terms {
                let idf_weight = idf
                    .as_ref()
                    .and_then(|m| m.get(&term.text))
                    .copied()
                    .unwrap_or(1.0);
                score += self.match_score(term, &name_lower, &type_lower, &desc_lower, idf_weight);
            }

            if score > 0.0 {
                scored.push((id.clone(), score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(id, _)| id).collect()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingMatcher
// ---------------------------------------------------------------------------

/// Matches entities by computing embedding similarity between the query and
/// entity representations (name + type + description).
///
/// Requires an [`Embeddings`] implementation to compute vectors. Embeddings
/// are cached internally to avoid recomputation across calls.
pub struct EmbeddingMatcher<E: Embeddings> {
    embeddings: E,
    /// Cached entity vectors: entity_id → embedding.
    cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
    /// Minimum embedding-similarity threshold (P1-2), default 0.0 keeps the old behavior.
    min_score: f64,
}

impl<E: Embeddings> EmbeddingMatcher<E> {
    /// Creates a new embedding matcher with the given embeddings backend.
    pub fn new(embeddings: E) -> Self {
        Self {
            embeddings,
            cache: std::sync::Mutex::new(HashMap::new()),
            min_score: 0.0,
        }
    }

    /// Sets the minimum embedding-similarity threshold (P1-2), default 0.0 keeps the old
    /// behavior.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Returns the embedding for an entity, computing and caching it if needed.
    async fn get_entity_embedding(&self, entity_id: &str, entity_text: &str) -> Option<Vec<f32>> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(vec) = cache.get(entity_id) {
                return Some(vec.clone());
            }
        }

        // Compute and cache
        match self.embeddings.embed_query(entity_text).await {
            Ok(vec) => {
                self.cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(entity_id.to_string(), vec.clone());
                Some(vec)
            }
            Err(e) => {
                // Entity embedding failed: the entity is excluded from graph matching, with a
                // log exposing the degradation
                log::warn!(
                    "entity `{}` embedding failed; excluded from graph matching: {}",
                    entity_id,
                    e
                );
                None
            }
        }
    }
}

impl<E: Embeddings + 'static> EntityMatcher for EmbeddingMatcher<E> {
    fn find_relevant(&self, query: &str, _store: &GraphStore, _top_k: usize) -> Vec<String> {
        // P1-4: no longer silently degrades to KeywordMatcher.
        //
        // A sync trait method cannot call the async `embed_query`; the old implementation
        // quietly fell back to keyword matching, so users thought they were using vector
        // matching while it was actually keywords, with zero warning — more dangerous than an
        // error. Here we refuse silent degradation: return empty results and `log::warn`,
        // making the failure visible.
        // For embedding matching call `find_relevant_async` (the GraphRAG query path), or
        // explicitly configure `KeywordMatcher`.
        log::warn!(
            "EmbeddingMatcher::find_relevant (sync) cannot run embedding matching and no longer \
             silently falls back to keyword matching; returning empty results for query '{}'. \
             Use find_relevant_async instead, or configure KeywordMatcher for sync matching.",
            query
        );
        Vec::new()
    }
}

impl<E: Embeddings + 'static> EmbeddingMatcher<E> {
    /// Async version of entity matching using embeddings.
    ///
    /// This is the preferred method when using embedding-based matching,
    /// since embedding computation is inherently async.
    ///
    /// P0-2: no more silent degradation / silent 0 scores — embedding failures or vector
    /// dimension mismatches now error out explicitly, letting callers know semantic matching
    /// is unavailable or the data is defective, instead of quietly falling back to keyword or
    /// treating "dimension mismatch" as "dissimilar".
    pub async fn find_relevant_async(
        &self,
        query: &str,
        store: &GraphStore,
        top_k: usize,
    ) -> Result<Vec<String>, GraphRAGError> {
        let query_vec = self.embeddings.embed_query(query).await.map_err(|e| {
            GraphRAGError::QueryError(format!("EmbeddingMatcher: query embedding failed: {}", e))
        })?;

        let mut scored: Vec<(String, f64)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let entity_text = format!(
                "{} {} {}",
                entity.name, entity.entity_type, entity.description
            );

            if let Some(entity_vec) = self.get_entity_embedding(id, &entity_text).await {
                match cosine_similarity(&query_vec, &entity_vec) {
                    Ok(score) => {
                        scored.push((id.clone(), score as f64));
                    }
                    // A vector dimension mismatch is a data defect (e.g. switching embedding
                    // models midway); error out rather than treating it as "dissimilar".
                    Err(lc_core::math::MathError::LengthMismatch(a, b)) => {
                        return Err(GraphRAGError::QueryError(format!(
                            "EmbeddingMatcher: vector dimension mismatch {} vs {} (embedding model changed?)",
                            a, b
                        )));
                    }
                    // `MathError` is `#[non_exhaustive]`; treat any other
                    // similarity failure as a query error.
                    Err(other) => {
                        return Err(GraphRAGError::QueryError(format!(
                            "EmbeddingMatcher: similarity computation failed: {}",
                            other
                        )));
                    }
                }
            }
        }

        let mut scored = filter_by_score(scored, self.min_score);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(top_k).map(|(id, _)| id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_rag::graph_store::{Entity, Relation};
    use lc_embeddings::MockEmbeddings;

    fn make_test_store() -> GraphStore {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "Python".into(),
            entity_type: "Technology".into(),
            description: "A scripting language".into(),
        });
        store.add_entity(Entity {
            id: "e3".into(),
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: "A developer who uses Rust".into(),
        });
        store.add_entity(Entity {
            id: "e4".into(),
            name: "Tokio".into(),
            entity_type: "Library".into(),
            description: "An async runtime for Rust".into(),
        });
        store.add_relation(Relation {
            source: "e3".into(),
            target: "e1".into(),
            relation_type: "uses".into(),
            description: "Alice uses Rust".into(),
            doc_id: None,
        });
        store
    }

    #[test]
    fn test_keyword_matcher_basic() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("Rust programming", &store, 10);
        assert!(!results.is_empty());
        // "Rust" entity should rank first (name match + description match)
        assert_eq!(results[0], "e1");
    }

    #[test]
    fn test_keyword_matcher_top_k() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("Technology", &store, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_keyword_matcher_no_match() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("cooking recipe", &store, 10);
        assert!(results.is_empty());
    }

    /// P1-4: the sync `find_relevant` no longer silently degrades — "Rust" clearly hits
    /// KeywordMatcher (e1) in the store, yet the EmbeddingMatcher sync path must return
    /// empty, refusing to quietly fall back to keyword matching so that "embedding matching
    /// unavailable" is visible.
    #[test]
    fn test_embedding_matcher_sync_returns_empty_not_keyword_fallback() {
        let store = make_test_store();
        let matcher = EmbeddingMatcher::new(MockEmbeddings::new(8));
        let results = matcher.find_relevant("Rust", &store, 10);
        assert!(
            results.is_empty(),
            "sync find_relevant must NOT silently fall back to keyword matching"
        );
    }

    /// P1-4: embedding matching goes through the async path — it still returns the truly
    /// similar entities.
    ///
    /// MockEmbeddings produces identical vectors for identical text, so when the query and
    /// the entity text match exactly, cosine = 1.0 > min_score (0.0), guaranteed to be
    /// recalled; the assertion is deterministic.
    #[tokio::test]
    async fn test_embedding_matcher_async_still_works() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        let matcher = EmbeddingMatcher::new(MockEmbeddings::new(8));
        let query = "Rust Technology A systems programming language";
        let results = matcher
            .find_relevant_async(query, &store, 10)
            .await
            .unwrap();
        assert_eq!(results, vec!["e1".to_string()]);
    }

    #[test]
    fn test_keyword_matcher_custom_weights() {
        let store = make_test_store();
        let matcher = KeywordMatcher {
            name_weight: 10,
            type_weight: 1,
            desc_weight: 0,
            ..Default::default()
        };
        let results = matcher.find_relevant("Rust", &store, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0], "e1");
    }

    /// P2-4: synonym-table expansion — a query term hitting a synonym key matches via the
    /// equivalent words, recovering missed recall.
    #[test]
    fn test_keyword_matcher_synonym_expansion() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "PostgreSQL".into(),
            entity_type: "Database".into(),
            description: "relational database".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "机器学习".into(),
            entity_type: "Technology".into(),
            description: "AI 领域".into(),
        });

        let synonyms: HashMap<String, Vec<String>> =
            HashMap::from([("数据库".to_string(), vec!["database".to_string()])]);
        let matcher = KeywordMatcher::new().with_synonyms(synonyms);

        // The query term cannot directly substring-match "PostgreSQL"; the synonym
        // "database" hits type/desc instead.
        let results = matcher.find_relevant("数据库", &store, 10);
        assert!(
            results.contains(&"e1".to_string()),
            "同义词 'database' 应能召回 PostgreSQL(e1)"
        );
    }

    /// P2-4: Chinese-English mixed normalization — full-width characters converted to
    /// half-width can match entity names.
    #[test]
    fn test_keyword_matcher_fullwidth_normalization() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "systems language".into(),
        });

        let matcher = KeywordMatcher::new();
        // Full-width letters are normalized to their half-width lowercase form, e.g. matching
        // "rust".
        let results = matcher.find_relevant("Ｒｕｓｔ", &store, 10);
        assert_eq!(results, vec!["e1".to_string()]);
    }

    /// P2-4: Chinese has no spaces; a long query split into CJK bigrams can hit short entity
    /// names.
    #[test]
    fn test_keyword_matcher_cjk_bigram_recall() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "机器学习".into(),
            entity_type: "Technology".into(),
            description: "AI 领域".into(),
        });

        let matcher = KeywordMatcher::new();
        // The full query is not a substring of the entity name; bigrams recover the match.
        let results = matcher.find_relevant("机器学习算法", &store, 10);
        assert!(
            results.contains(&"e1".to_string()),
            "CJK 二元组应能召回 '机器学习' 实体"
        );
    }

    /// P2-4: TF-IDF property — the IDF of a common term is lower than that of a rare term.
    #[test]
    fn test_keyword_matcher_tfidf_common_lower_than_rare() {
        let mut store = GraphStore::new();
        for name in ["Rust", "Python", "Ruby"] {
            store.add_entity(Entity {
                id: name.to_lowercase(),
                name: name.to_string(),
                entity_type: "Technology".to_string(),
                description: String::new(),
            });
        }

        let matcher = KeywordMatcher::new();
        let terms = matcher.build_terms("rust technology");
        let idf = matcher.compute_idf(&terms, &store);

        // "technology" appears in the type of 3 entities -> low IDF; "rust" only in e1 -> high IDF.
        let tech_idf = idf["technology"];
        let rust_idf = idf["rust"];
        assert!(
            rust_idf > tech_idf,
            "稀有词 'rust' 的 IDF ({:.3}) 应高于常见词 'technology' ({:.3})",
            rust_idf,
            tech_idf
        );
    }

    /// P2-4: TF-IDF changes the ordering — when fixed weights tie, the rare-term hit wins.
    #[test]
    fn test_keyword_matcher_tfidf_breaks_fixed_weight_tie() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "data".into(),
            entity_type: "Technology".into(),
            description: "machine learning".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "learning".into(),
            entity_type: "Technology".into(),
            description: "data".into(),
        });
        store.add_entity(Entity {
            id: "e3".into(),
            name: "extra".into(),
            entity_type: "Technology".into(),
            description: "data warehouse".into(),
        });

        let matcher = KeywordMatcher::new();
        // With fixed weights e1/e2 tie (4 points); with TF-IDF, "learning" is rarer (only 2
        // entities), and e2's name hits the rare term -> it should win.
        let results = matcher.find_relevant("data learning", &store, 10);
        assert_eq!(results[0], "e2");
    }

    /// P0-2: converges on the single lc-core implementation (the Result<f32, MathError>
    /// contract).
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        // A zero vector is not a dimension error — lc-core returns Ok(0.0).
        let sim = cosine_similarity(&[], &[]).unwrap();
        assert_eq!(sim, 0.0);
    }

    /// P0-2: dimension mismatches must error, no longer a silent 0.0 (otherwise "dimension
    /// mismatch" is taken as "dissimilar").
    #[test]
    fn test_cosine_similarity_different_lengths_errors() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }
}
