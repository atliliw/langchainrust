#![warn(missing_docs)]
// lc-vector-stores/src/lib.rs
//! Vector store implementations.
//!
//! Provides document vector storage and retrieval functionality.

pub mod chromadb;
pub mod chunked_vector_store;
pub mod document_store;
mod file_store;
pub mod filter;
pub mod lancedb;
mod memory;
pub mod neo4j;
mod provider;

#[cfg(feature = "mongodb-persistence")]
mod mongo_document_store;

#[cfg(feature = "qdrant-integration")]
mod qdrant;

#[cfg(feature = "redis-storage")]
pub mod redis_store;

#[cfg(feature = "sqlite-storage")]
pub mod sqlite_store;

#[cfg(feature = "pgvector-storage")]
pub mod pgvector;

pub mod pinecone;

pub use chunked_vector_store::ChunkedVectorStore;
pub use document_store::{
    ChunkedDocumentStore, ChunkedDocumentStoreTrait, DocumentStore, InMemoryChunkedDocumentStore,
    InMemoryDocumentStore,
};
pub use file_store::FileVectorStore;
pub use filter::{FilterOp, MetadataFilter};
pub use lancedb::{LanceDBConfig, LanceDBVectorStore};
pub use memory::InMemoryVectorStore;
pub use neo4j::{Neo4jConfig, Neo4jVectorStore};
pub use pinecone::PineconeStore;
pub use provider::{VectorStoreBuilder, VectorStoreProvider, VectorStoreType};

#[cfg(feature = "mongodb-persistence")]
pub use mongo_document_store::{MongoChunkedDocumentStore, MongoStoreConfig};

#[cfg(feature = "qdrant-integration")]
pub use qdrant::{QdrantConfig, QdrantDistance, QdrantVectorStore};

pub use chromadb::{ChromaDBConfig, ChromaDBVectorStore};

#[cfg(feature = "redis-storage")]
pub use redis_store::{RedisDocumentStore, RedisStoreConfig};

#[cfg(feature = "sqlite-storage")]
pub use sqlite_store::{SQLiteDocumentStore, SQLiteStoreConfig};

#[cfg(feature = "pgvector-storage")]
pub use pgvector::{build_filter_sql, FilterBinding, FilterSql, PGVectorConfig, PGVectorStore};

use async_trait::async_trait;

// Re-export shared document types from lc-shared
pub use lc_shared::document::{ChunkDocument, Document, SearchResult, VectorDocument};

// Re-export cosine_similarity from lc-core
pub use lc_core::math::cosine_similarity;

/// Vector store error types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VectorStoreError {
    /// Document not found.
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Embedding error.
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Connection error (for remote vector databases).
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Configuration error (e.g. missing environment variables, invalid settings).
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Metadata filtering is not supported (the backend does not override filtered retrieval).
    #[error("Metadata filter not supported: {0}")]
    UnsupportedFilter(String),
}

/// Vector store trait.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds documents.
    ///
    /// # Arguments
    /// * `documents` - Document list.
    /// * `embeddings` - Embedding vectors for documents.
    ///
    /// # Returns
    /// Document ID list.
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError>;

    /// Searches similar documents.
    ///
    /// # Arguments
    /// * `query_embedding` - Query vector.
    /// * `k` - Number of documents to return.
    ///
    /// # Returns
    /// Similar document list (sorted by similarity descending).
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Returns this vector store's built-in text embedder, if any.
    ///
    /// Q1: previously the trait only took `query_embedding: &[f32]`, so callers had to embed
    /// the query themselves without a contract saying which embedder to use. With this getter,
    /// implementations that embed internally can accept text directly via
    /// [`similarity_search_text`](Self::similarity_search_text); those without one return
    /// `None`, and the caller gets an explicit error instead of silently using the wrong model.
    fn embed_query(&self) -> Option<&dyn Embeddings> {
        None
    }

    /// Text similarity search: vectorizes `query` with the embedder returned by
    /// [`embed_query`](Self::embed_query), then searches.
    ///
    /// Returns [`VectorStoreError::EmbeddingError`] when no embedder is configured, suggesting
    /// [`similarity_search`](Self::similarity_search) with a query vector instead.
    async fn similarity_search_text(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let Some(embeddings) = self.embed_query() else {
            return Err(VectorStoreError::EmbeddingError(
                "this vector store has no embedder configured; cannot auto-vectorize the query \
                 text; call similarity_search with a query vector instead"
                    .to_string(),
            ));
        };
        let query_embedding = embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
        self.similarity_search(&query_embedding, k).await
    }

    /// Similarity search with metadata filtering.
    ///
    /// - `filter: None` — no filtering, equivalent to [`similarity_search`](Self::similarity_search).
    /// - `filter: Some(f)` — returns only documents matching the filter, at most `k` entries.
    ///
    /// Default implementation: delegates to [`similarity_search`](Self::similarity_search) when
    /// there is no filter; returns [`VectorStoreError::UnsupportedFilter`] when a filter is given
    /// but the backend does not override it, **without silently ignoring** the filter. Backends
    /// that support filtering should override this method and translate [`MetadataFilter`] into
    /// their native query syntax (Qdrant payload filter / Pinecone filter / Chroma where / …).
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        match filter {
            None => self.similarity_search(query_embedding, k).await,
            Some(_) => Err(VectorStoreError::UnsupportedFilter(
                "this vector store does not support metadata filtering; pass filter: None or \
                 switch to a store that implements similarity_search_with_filter"
                    .to_string(),
            )),
        }
    }

    /// Similarity search with a minimum score threshold.
    ///
    /// - `min_score: None` — no filtering, returns the store's full top-k (even negative scores).
    /// - `min_score: Some(t)` — returns only results with `score >= t`, at most `k` entries.
    ///
    /// Q2: the default implementation re-filters the results of
    /// [`similarity_search`](Self::similarity_search), the best approximation for backends that
    /// cannot threshold directly at retrieval time. Implementations that compute similarity
    /// locally should override this method for the precise "filter first, then take top-k"
    /// semantics.
    async fn similarity_search_with_min_score(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let results = self.similarity_search(query_embedding, k).await?;
        match min_score {
            Some(threshold) => Ok(results
                .into_iter()
                .filter(|r| r.score >= threshold)
                .collect()),
            None => Ok(results),
        }
    }

    /// Gets document by ID.
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;

    /// Gets document embedding by ID.
    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError>;

    /// Deletes document.
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;

    /// Returns document count.
    async fn count(&self) -> usize;

    /// Clears store.
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

/// Embedding model trait — re-exported from lc-embeddings.
///
/// Used by vector store implementations that need to embed documents on the fly
/// (e.g., Pinecone's `upsert` method).
pub use lc_embeddings::{EmbeddingError, Embeddings};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello, world!")
            .with_metadata("source", "test")
            .with_id("doc-1");

        assert_eq!(doc.content, "Hello, world!");
        assert_eq!(
            doc.metadata.get("source"),
            Some(&serde_json::Value::String("test".to_string()))
        );
        assert_eq!(doc.id, Some("doc-1".to_string()));
    }

    #[test]
    fn test_document_page_content() {
        let doc = Document::new("Test content");
        assert_eq!(doc.page_content(), "Test content");
    }
}
