// lc-vector-stores/src/chunked_vector_store.rs
//! Chunked Vector Store — split-document vector storage
//!
//! Stores only vectors + chunk_id references; content is fetched from the DocumentStore.
//! Supports the Parent-Child document structure, suitable for long-document splitting scenarios.

use crate::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait, DocumentStore};
use crate::{
    cosine_similarity, Document, MetadataFilter, SearchResult, VectorStore, VectorStoreError,
};
use async_trait::async_trait;
use futures_util::future;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Vector index entry (stores only the vector + chunk_id)
struct VectorEntry {
    chunk_id: String,
    embedding: Vec<f32>,
}

/// Chunked Vector Store
pub struct ChunkedVectorStore {
    document_store: Arc<ChunkedDocumentStore>,
    vectors: Arc<RwLock<HashMap<String, VectorEntry>>>,
    vector_size: usize,
}

impl ChunkedVectorStore {
    /// Creates a new ChunkedVectorStore
    pub fn new(document_store: Arc<ChunkedDocumentStore>, vector_size: usize) -> Self {
        Self {
            document_store,
            vectors: Arc::new(RwLock::new(HashMap::new())),
            vector_size,
        }
    }

    /// Adds a chunk vector (chunk_id + embedding)
    pub async fn add_chunk_vector(
        &self,
        chunk_id: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Result<(), VectorStoreError> {
        if embedding.len() != self.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.vector_size,
                embedding.len()
            )));
        }

        let chunk_id = chunk_id.into();
        let mut vectors = self.vectors.write().await;
        vectors.insert(
            chunk_id.clone(),
            VectorEntry {
                chunk_id,
                embedding,
            },
        );

        Ok(())
    }

    /// Adds chunk vectors in bulk
    pub async fn add_chunk_vectors(
        &self,
        chunk_ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<(), VectorStoreError> {
        if chunk_ids.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "chunk_id count and embedding count mismatch".to_string(),
            ));
        }

        for (chunk_id, embedding) in chunk_ids.into_iter().zip(embeddings.into_iter()) {
            self.add_chunk_vector(chunk_id, embedding).await?;
        }

        Ok(())
    }

    /// Adds from a Parent document (auto-split + vectorized)
    pub async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
        embeddings_fn: impl Fn(&str) -> Vec<f32>,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let (parent_id, chunk_ids) = self
            .document_store
            .add_parent_document(document, chunk_size)
            .await?;

        for chunk_id in &chunk_ids {
            let chunk = self
                .document_store
                .get_chunk(chunk_id)
                .await?
                .ok_or_else(|| VectorStoreError::DocumentNotFound(chunk_id.clone()))?;

            let embedding = embeddings_fn(&chunk.content);
            self.add_chunk_vector(chunk_id.clone(), embedding).await?;
        }

        Ok((parent_id, chunk_ids))
    }

    /// Gets the vector for a chunk_id (M4: O(1) HashMap lookup)
    pub async fn get_embedding(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        Ok(vectors.get(chunk_id).map(|e| e.embedding.clone()))
    }

    /// Gets the number of stored vectors
    pub async fn vector_count(&self) -> usize {
        let vectors = self.vectors.read().await;
        vectors.len()
    }
}

#[async_trait]
impl VectorStore for ChunkedVectorStore {
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

        let mut ids = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            let chunk_id = doc
                .id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

            self.document_store.add_document(doc).await?;
            self.add_chunk_vector(chunk_id.clone(), embedding).await?;

            ids.push(chunk_id);
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Q2: no longer hard-filters score > 0 — top-k must be returned even for all-negative
        // corpora; whether to set a threshold is the caller's explicit choice via similarity_search_with_min_score.
        self.similarity_search_with_min_score(query_embedding, k, None)
            .await
    }

    async fn similarity_search_with_min_score(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let vectors = self.vectors.read().await;

        // compute similarity for all vectors, filter by threshold first, then take top-k (Q2)
        let mut results: Vec<(String, f32)> = vectors
            .values()
            .filter_map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.embedding).unwrap_or(0.0);
                if min_score.is_none_or(|t| score >= t) {
                    Some((entry.chunk_id.clone(), score))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_ids: Vec<(String, f32)> = results.into_iter().take(k).collect();

        let search_results: Vec<SearchResult> =
            future::join_all(top_k_ids.iter().map(|(chunk_id, score)| async move {
                let doc = match self.document_store.get_chunk_document(chunk_id).await {
                    Ok(doc) => doc,
                    Err(e) => {
                        // stop silently swallowing errors: log read failures; the chunk is missing from the top-k results
                        log::error!(
                            "failed to read document for chunk `{}` while retrieving (chunk dropped from results): {}",
                            chunk_id,
                            e
                        );
                        None
                    }
                };
                doc.map(|d| SearchResult {
                    document: d,
                    score: *score,
                })
            }))
            .await
            .into_iter()
            .flatten()
            .collect();

        Ok(search_results)
    }

    /// S3: chunked-store metadata filtering.
    ///
    /// The vector index carries no metadata, so filtering requires fetching documents from the
    /// document store by chunk_id. The semantics are "score everything → scan descending,
    /// filtering each item by metadata → stop once k items are collected", guaranteeing the
    /// top-k by similarity after filtering (rather than truncating first and filtering later).
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // filter: None → delegate to plain retrieval (no filtering).
        let Some(filter) = filter else {
            return self.similarity_search(query_embedding, k).await;
        };

        let vectors = self.vectors.read().await;

        // 1. score everything
        let mut scored: Vec<(String, f32)> = vectors
            .values()
            .map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.embedding).unwrap_or(0.0);
                (entry.chunk_id.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. scan descending, fetch each document and filter by metadata, stopping once k items
        //    are collected. Missing documents / read failures are logged and the candidate is
        //    skipped (consistent with the with_min_score handling).
        let mut results: Vec<SearchResult> = Vec::new();
        for (chunk_id, score) in scored {
            // read failures are handled like with_min_score: log, skip the candidate, do not abort the whole retrieval.
            let doc = match self.document_store.get_chunk_document(&chunk_id).await {
                Ok(doc) => doc,
                Err(e) => {
                    log::error!(
                        "failed to read document for chunk `{}` while filtering (chunk skipped): {}",
                        chunk_id,
                        e
                    );
                    continue;
                }
            };
            let Some(doc) = doc else {
                log::error!(
                    "document for chunk `{}` is missing while filtering (chunk skipped)",
                    chunk_id
                );
                continue;
            };
            if filter.matches(&doc.metadata) {
                results.push(SearchResult {
                    document: doc,
                    score,
                });
                if results.len() >= k {
                    break;
                }
            }
        }

        Ok(results)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        self.document_store.get_chunk_document(id).await
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        Ok(vectors.get(id).map(|e| e.embedding.clone()))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut vectors = self.vectors.write().await;
        vectors.remove(id);

        self.document_store.delete_document(id).await?;

        Ok(())
    }

    async fn count(&self) -> usize {
        self.vector_count().await
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut vectors = self.vectors.write().await;
        vectors.clear();

        ChunkedDocumentStoreTrait::clear(&*self.document_store).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_embedding(content: &str) -> Vec<f32> {
        let len = content.len() as f32;
        vec![len / 100.0, 0.0, 0.0]
    }

    #[tokio::test]
    async fn test_chunked_vector_store_basic() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);

        let chunk_id = "chunk_001".to_string();
        let embedding = vec![1.0, 0.0, 0.0];

        vector_store
            .add_chunk_vector(chunk_id.clone(), embedding.clone())
            .await
            .unwrap();

        assert_eq!(vector_store.vector_count().await, 1);

        let retrieved = vector_store.get_embedding(&chunk_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);

        vector_store
            .add_chunk_vector("chunk_001".to_string(), vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        vector_store
            .add_chunk_vector("chunk_002".to_string(), vec![0.0, 1.0, 0.0])
            .await
            .unwrap();

        doc_store
            .add_document(Document::new("Rust content").with_id("chunk_001"))
            .await
            .unwrap();
        doc_store
            .add_document(Document::new("Python content").with_id("chunk_002"))
            .await
            .unwrap();

        let query = vec![0.9, 0.1, 0.0];
        let results = vector_store.similarity_search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_add_parent_document() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);

        let doc = Document::new("这是一段很长的测试文本，用于验证分割功能。").with_id("parent_001");

        let (parent_id, chunk_ids) = vector_store
            .add_parent_document(doc, 20, mock_embedding)
            .await
            .unwrap();

        assert_eq!(parent_id, "parent_001");
        assert!(chunk_ids.len() > 1);
        assert_eq!(vector_store.vector_count().await, chunk_ids.len());
    }

    /// Q2: similarity_search still returns top-k for an all-non-positive corpus (no longer emptied
    /// by a hard score > 0 filter), and similarity_search_with_min_score allows explicit filtering.
    #[tokio::test]
    async fn test_negative_scores_not_dropped() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 2);

        for (cid, v) in [
            ("chunk_001", vec![0.0, 1.0]),
            ("chunk_002", vec![-1.0, 0.0]),
            ("chunk_003", vec![0.0, -1.0]),
        ] {
            vector_store
                .add_chunk_vector(cid.to_string(), v)
                .await
                .unwrap();
            doc_store
                .add_document(Document::new(cid).with_id(cid))
                .await
                .unwrap();
        }

        let query = vec![1.0, 0.0];

        let results = vector_store.similarity_search(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.score <= 0.0));

        let filtered = vector_store
            .similarity_search_with_min_score(&query, 3, Some(-0.5))
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2);
    }

    /// S3: chunked-store metadata filtering — filtering happens before top-k, returning the similarity top-k among matching documents.
    #[tokio::test]
    async fn test_similarity_search_with_filter() {
        use crate::FilterOp;

        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);

        vector_store
            .add_chunk_vector("chunk_001".to_string(), vec![1.0, 0.0, 0.0])
            .await
            .unwrap();
        vector_store
            .add_chunk_vector("chunk_002".to_string(), vec![0.0, 1.0, 0.0])
            .await
            .unwrap();
        vector_store
            .add_chunk_vector("chunk_003".to_string(), vec![0.9, 0.1, 0.0])
            .await
            .unwrap();

        doc_store
            .add_document(
                Document::new("rust doc")
                    .with_id("chunk_001")
                    .with_metadata("lang", "rust"),
            )
            .await
            .unwrap();
        doc_store
            .add_document(
                Document::new("python doc")
                    .with_id("chunk_002")
                    .with_metadata("lang", "python"),
            )
            .await
            .unwrap();
        doc_store
            .add_document(
                Document::new("rust legacy")
                    .with_id("chunk_003")
                    .with_metadata("lang", "rust"),
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];

        // single condition: only rust documents are returned, in similarity descending order (chunk_001 > chunk_003)
        let eq = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let r = vector_store
            .similarity_search_with_filter(&query, 5, Some(&eq))
            .await
            .unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].document.content, "rust doc");
        assert_eq!(r[1].document.content, "rust legacy");

        // k applies after filtering
        let r = vector_store
            .similarity_search_with_filter(&query, 1, Some(&eq))
            .await
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].document.content, "rust doc");

        // filter: None behaves identically to similarity_search
        let none = vector_store
            .similarity_search_with_filter(&query, 5, None)
            .await
            .unwrap();
        let base = vector_store.similarity_search(&query, 5).await.unwrap();
        assert_eq!(none.len(), base.len());
    }

    #[tokio::test]
    async fn test_dimension_validation() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 128);

        let result = vector_store
            .add_chunk_vector("chunk_001".to_string(), vec![1.0, 0.0])
            .await;

        assert!(result.is_err());
    }
}
