// lc-vector-stores/src/lib.rs
//! Vector store implementations.
//!
//! Provides document vector storage and retrieval functionality.

pub mod chromadb;
pub mod chunked_vector_store;
pub mod document_store;
mod file_store;
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

use async_trait::async_trait;
use std::error::Error;

// Re-export shared document types from lc-shared
pub use lc_shared::document::{ChunkDocument, Document, SearchResult, VectorDocument};

// Re-export cosine_similarity from lc-core
pub use lc_core::math::cosine_similarity;

/// Vector store error types.
#[derive(Debug)]
pub enum VectorStoreError {
    /// Document not found.
    DocumentNotFound(String),

    /// Embedding error.
    EmbeddingError(String),

    /// Storage error.
    StorageError(String),

    /// Connection error (for remote vector databases).
    ConnectionError(String),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::DocumentNotFound(id) => write!(f, "Document not found: {}", id),
            VectorStoreError::EmbeddingError(msg) => write!(f, "Embedding error: {}", msg),
            VectorStoreError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            VectorStoreError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
        }
    }
}

impl Error for VectorStoreError {}

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

    /// 返回该向量存储自带的文本嵌入器(若有)。
    ///
    /// Q1: 此前 trait 只接收 `query_embedding: &[f32]`,调用方必须自己嵌入查询
    /// 文本,却没有契约告诉它"该用哪个嵌入器"。有了该 getter,内嵌嵌入器的实现
    /// 可以直接用 [`similarity_search_text`](Self::similarity_search_text) 传文本;
    /// 没有的返回 `None`,调用方会收到显式错误而不是静默地用错模型。
    fn embed_query(&self) -> Option<&dyn Embeddings> {
        None
    }

    /// 文本相似度检索:用 [`embed_query`](Self::embed_query) 返回的嵌入器把
    /// `query` 向量化后再检索。
    ///
    /// 未配置嵌入器时返回 [`VectorStoreError::EmbeddingError`],提示改用
    /// [`similarity_search`](Self::similarity_search) 直接传入查询向量。
    async fn similarity_search_text(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let Some(embeddings) = self.embed_query() else {
            return Err(VectorStoreError::EmbeddingError(
                "该向量存储未配置嵌入器,无法自动向量化查询文本;请改用 similarity_search 直接传入查询向量"
                    .to_string(),
            ));
        };
        let query_embedding = embeddings
            .embed_query(query)
            .await
            .map_err(|e| VectorStoreError::EmbeddingError(e.to_string()))?;
        self.similarity_search(&query_embedding, k).await
    }

    /// 带最低分数阈值的相似度检索。
    ///
    /// - `min_score: None` —— 不过滤,返回全库 top-k(即使分数为负)。
    /// - `min_score: Some(t)` —— 只返回 `score >= t` 的结果,最多 `k` 条。
    ///
    /// Q2: 默认实现基于 [`similarity_search`](Self::similarity_search) 的结果做二次过滤
    /// (对检索期无法直接按阈值过滤的后端是最佳近似)。本地计算相似度的实现应覆盖此
    /// 方法,以获得"先过滤再取 top-k"的精确语义。
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
        assert_eq!(doc.metadata.get("source"), Some(&"test".to_string()));
        assert_eq!(doc.id, Some("doc-1".to_string()));
    }

    #[test]
    fn test_document_page_content() {
        let doc = Document::new("Test content");
        assert_eq!(doc.page_content(), "Test content");
    }
}
