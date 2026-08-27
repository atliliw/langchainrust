// lc-vector-stores/src/memory.rs
//! In-memory vector store
//!
//! Stores documents and vectors in memory, suitable for small-scale data and tests.

use crate::{
    cosine_similarity, Document, MetadataFilter, SearchResult, VectorDocument, VectorStore,
    VectorStoreError,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// In-memory vector store
pub struct InMemoryVectorStore {
    /// Document storage
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,
}

impl InMemoryVectorStore {
    /// Creates a new in-memory vector store
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "document count and embedding count mismatch".to_string(),
            ));
        }

        let mut store = self.documents.write().await;
        let mut ids = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

            let vector_doc = VectorDocument {
                document: Document {
                    id: Some(id.clone()),
                    content: doc.content,
                    metadata: doc.metadata,
                },
                embedding,
            };

            store.insert(id.clone(), vector_doc);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Q2: no longer hard-filters score > 0 — under an all-negative corpus the top-k
        // should still be returned; whether to set a threshold is the caller's explicit
        // decision via similarity_search_with_min_score.
        self.similarity_search_with_min_score(query_embedding, k, None)
            .await
    }

    async fn similarity_search_with_min_score(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let store = self.documents.read().await;

        // compute similarity for all documents, filter by threshold first, then take top-k (Q2)
        let mut results: Vec<SearchResult> = store
            .values()
            .filter_map(|vd| {
                let score = cosine_similarity(query_embedding, &vd.embedding).unwrap_or(0.0);
                if min_score.is_none_or(|t| score >= t) {
                    Some(SearchResult {
                        document: vd.document.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        // sort by similarity descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // return the top k results
        Ok(results.into_iter().take(k).collect())
    }

    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let store = self.documents.read().await;

        // S3: in-memory metadata filtering — filter documents by the condition first, then compute similarity and take top-k.
        let mut results: Vec<SearchResult> = store
            .values()
            .filter(|vd| filter.is_none_or(|f| f.matches(&vd.document.metadata)))
            .map(|vd| {
                let score = cosine_similarity(query_embedding, &vd.embedding).unwrap_or(0.0);
                SearchResult {
                    document: vd.document.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results.into_iter().take(k).collect())
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.document.clone()))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.embedding.clone()))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().await;
        store.remove(id);
        Ok(())
    }

    async fn count(&self) -> usize {
        let store = self.documents.read().await;
        store.len()
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().await;
        store.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_search() {
        let store = InMemoryVectorStore::new();

        // add documents
        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];

        // create simple mock embedding vectors
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Rust-related
            vec![0.0, 1.0, 0.0], // Python-related
            vec![0.0, 0.0, 1.0], // JavaScript-related
        ];

        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.count().await, 3);

        // search for similar documents
        let query = vec![0.9, 0.1, 0.0]; // closer to Rust
        let results = store.similarity_search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("Rust"));
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let store = InMemoryVectorStore::new();

        let doc = Document::new("Test document").with_id("test-id");
        let embeddings = vec![vec![1.0, 0.0, 0.0]];

        store.add_documents(vec![doc], embeddings).await.unwrap();

        // get the document
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test document");

        // delete the document
        store.delete_document("test-id").await.unwrap();
        assert_eq!(store.count().await, 0);

        // fetching again should return None
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryVectorStore::new();

        let docs = vec![Document::new("Doc 1"), Document::new("Doc 2")];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(store.count().await, 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    /// Q1: without a configured embedder, similarity_search_text must report EmbeddingError
    /// explicitly, rather than silently succeeding or panicking.
    #[tokio::test]
    async fn test_similarity_search_text_without_embedder_errors() {
        let store = InMemoryVectorStore::new();
        let err = store.similarity_search_text("hello", 3).await.unwrap_err();
        assert!(matches!(err, VectorStoreError::EmbeddingError(_)));
    }

    /// Q2: under an all-non-positive-score corpus, similarity_search still returns the top-k
    /// (no longer cleared by a score>0 hard filter); similarity_search_with_min_score filters
    /// explicitly by threshold.
    #[tokio::test]
    async fn test_negative_scores_not_dropped() {
        let store = InMemoryVectorStore::new();
        store
            .add_documents(
                vec![
                    Document::new("orthogonal-up"),
                    Document::new("opposite"),
                    Document::new("orthogonal-down"),
                ],
                vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]],
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0];

        // the old implementation hard-filtered score > 0.0, which would return empty here; now it returns the top-k (3 items, all non-positive).
        let results = store.similarity_search(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.score <= 0.0));

        // explicit threshold: score >= -0.5 excludes the score = -1.0 entry
        let filtered = store
            .similarity_search_with_min_score(&query, 3, Some(-0.5))
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2);

        // min_score = None behaves identically to similarity_search
        let all = store
            .similarity_search_with_min_score(&query, 3, None)
            .await
            .unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 0.0001);
    }

    /// S3: in-memory metadata filtering — single condition + AND/OR combination; `filter: None` matches the legacy path.
    #[tokio::test]
    async fn test_metadata_filter() {
        use crate::FilterOp;

        let store = InMemoryVectorStore::new();
        store
            .add_documents(
                vec![
                    Document::new("rust doc")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2024),
                    Document::new("python doc")
                        .with_metadata("lang", "python")
                        .with_metadata("year", 2023),
                    Document::new("rust legacy")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2020),
                ],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.9, 0.1, 0.0],
                ],
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];

        // single condition
        let eq = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&eq))
            .await
            .unwrap();
        assert_eq!(r.len(), 2);
        assert!(r
            .iter()
            .all(|s| s.document.metadata.get("lang").and_then(|v| v.as_str()) == Some("rust")));

        // AND combination
        let and = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::field("year", FilterOp::Gte, 2021),
        ]);
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&and))
            .await
            .unwrap();
        assert_eq!(r.len(), 1);
        assert!(r[0].document.content.contains("rust doc"));

        // OR combination
        let or = MetadataFilter::or(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "python"),
            MetadataFilter::field("year", FilterOp::Lt, 2021),
        ]);
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&or))
            .await
            .unwrap();
        assert_eq!(r.len(), 2);

        // filter: None behaves identically to similarity_search (regression)
        let none = store
            .similarity_search_with_filter(&query, 5, None)
            .await
            .unwrap();
        let base = store.similarity_search(&query, 5).await.unwrap();
        assert_eq!(none.len(), base.len());
    }
}
