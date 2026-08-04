// src/retrieval/graph_rag/matcher.rs
//! Entity matching strategies for GraphRAG local queries.
//!
//! Provides the [`EntityMatcher`] trait and two implementations:
//! - [`KeywordMatcher`]: matches entities by keyword substring (default, zero-cost)
//! - [`EmbeddingMatcher`]: matches entities by embedding cosine similarity

use super::graph_store::GraphStore;
use lc_embeddings::Embeddings;
use std::collections::HashMap;

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

/// Matches entities by keyword substring search.
///
/// This is the default matcher used by GraphRAG. It splits the query into
/// keywords and scores each entity based on how many keywords match the
/// entity's name, type, and description. Name matches are weighted highest.
pub struct KeywordMatcher {
    /// Weight for name matches (default: 3).
    pub name_weight: usize,
    /// Weight for type matches (default: 2).
    pub type_weight: usize,
    /// Weight for description matches (default: 1).
    pub desc_weight: usize,
}

impl Default for KeywordMatcher {
    fn default() -> Self {
        Self {
            name_weight: 3,
            type_weight: 2,
            desc_weight: 1,
        }
    }
}

impl KeywordMatcher {
    /// Creates a new keyword matcher with default weights.
    pub fn new() -> Self {
        Self::default()
    }
}

impl EntityMatcher for KeywordMatcher {
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        let mut scored: Vec<(String, usize)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let name_lower = entity.name.to_lowercase();
            let desc_lower = entity.description.to_lowercase();
            let type_lower = entity.entity_type.to_lowercase();

            let mut score = 0usize;
            for kw in &keywords {
                if name_lower.contains(kw) {
                    score += self.name_weight;
                }
                if type_lower.contains(kw) {
                    score += self.type_weight;
                }
                if desc_lower.contains(kw) {
                    score += self.desc_weight;
                }
            }

            if score > 0 {
                scored.push((id.clone(), score));
            }
        }

        scored.sort_by(|a, b| b.1.cmp(&a.1));
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
}

impl<E: Embeddings> EmbeddingMatcher<E> {
    /// Creates a new embedding matcher with the given embeddings backend.
    pub fn new(embeddings: E) -> Self {
        Self {
            embeddings,
            cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Returns the embedding for an entity, computing and caching it if needed.
    async fn get_entity_embedding(&self, entity_id: &str, entity_text: &str) -> Option<Vec<f32>> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(vec) = cache.get(entity_id) {
                return Some(vec.clone());
            }
        }

        // Compute and cache
        match self.embeddings.embed_query(entity_text).await {
            Ok(vec) => {
                self.cache
                    .lock()
                    .unwrap()
                    .insert(entity_id.to_string(), vec.clone());
                Some(vec)
            }
            Err(_) => None,
        }
    }
}

impl<E: Embeddings + 'static> EntityMatcher for EmbeddingMatcher<E> {
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String> {
        // Synchronous wrapper: we can't call async embed_query in a sync trait method.
        // Fallback to keyword matching for the sync interface.
        // The async version is available via `find_relevant_async`.
        let fallback = KeywordMatcher::new();
        fallback.find_relevant(query, store, top_k)
    }
}

impl<E: Embeddings + 'static> EmbeddingMatcher<E> {
    /// Async version of entity matching using embeddings.
    ///
    /// This is the preferred method when using embedding-based matching,
    /// since embedding computation is inherently async.
    pub async fn find_relevant_async(
        &self,
        query: &str,
        store: &GraphStore,
        top_k: usize,
    ) -> Vec<String> {
        let query_vec = match self.embeddings.embed_query(query).await {
            Ok(v) => v,
            Err(_) => {
                // Fallback to keyword matching if embedding fails
                let fallback = KeywordMatcher::new();
                return fallback.find_relevant(query, store, top_k);
            }
        };

        let mut scored: Vec<(String, f64)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let entity_text = format!(
                "{} {} {}",
                entity.name, entity.entity_type, entity.description
            );

            if let Some(entity_vec) = self.get_entity_embedding(id, &entity_text).await {
                let similarity = cosine_similarity(&query_vec, &entity_vec);
                if similarity > 0.0 {
                    scored.push((id.clone(), similarity));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(id, _)| id).collect()
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_rag::graph_store::{Entity, Relation};

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

    #[test]
    fn test_keyword_matcher_custom_weights() {
        let store = make_test_store();
        let matcher = KeywordMatcher {
            name_weight: 10,
            type_weight: 1,
            desc_weight: 0,
        };
        let results = matcher.find_relevant("Rust", &store, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0], "e1");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
