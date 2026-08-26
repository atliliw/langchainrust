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

    /// 元数据过滤不受支持(后端未覆写过滤检索)。
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

    /// 带元数据过滤的相似度检索。
    ///
    /// - `filter: None` —— 不过滤,等价 [`similarity_search`](Self::similarity_search)。
    /// - `filter: Some(f)` —— 只返回满足过滤条件的文档,最多 `k` 条。
    ///
    /// 默认实现:无过滤时委托 [`similarity_search`](Self::similarity_search);有过滤
    /// 但后端未覆写(不支持)时返回 [`VectorStoreError::UnsupportedFilter`],**不静默
    /// 忽略**过滤。支持过滤的后端应覆写本方法,把 [`MetadataFilter`] 翻译成各自
    /// 原生查询语法(Qdrant payload filter / Pinecone filter / Chroma where / …)。
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
