// src/vector_stores/mod.rs
//! Vector store implementations.
//!
//! Provides document vector storage and retrieval functionality.

mod memory;
mod provider;
pub mod document_store;
pub mod chunked_vector_store;

#[cfg(feature = "mongodb-persistence")]
mod mongo_document_store;

#[cfg(feature = "qdrant-integration")]
mod qdrant;

pub use memory::InMemoryVectorStore;
pub use provider::{VectorStoreProvider, VectorStoreType, VectorStoreBuilder};
pub use document_store::{DocumentStore, InMemoryDocumentStore, ChunkedDocumentStoreTrait, InMemoryChunkedDocumentStore, ChunkedDocumentStore, ChunkDocument};
pub use chunked_vector_store::ChunkedVectorStore;

#[cfg(feature = "mongodb-persistence")]
pub use mongo_document_store::{MongoChunkedDocumentStore, MongoStoreConfig};

#[cfg(feature = "qdrant-integration")]
pub use qdrant::{QdrantVectorStore, QdrantConfig, QdrantDistance};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

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

/// Document structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document content.
    pub content: String,
    
    /// Document metadata.
    pub metadata: HashMap<String, String>,
    
    /// Document ID (optional).
    pub id: Option<String>,
}

impl Document {
    /// Creates a new document.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
            id: None,
        }
    }
    
    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Sets ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    
    /// Returns page content (alias).
    pub fn page_content(&self) -> &str {
        &self.content
    }
}

/// Vector document with embedding.
#[derive(Debug, Clone)]
pub struct VectorDocument {
    /// Document.
    pub document: Document,
    
    /// Embedding vector.
    pub embedding: Vec<f32>,
}

/// Search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document.
    pub document: Document,
    
    /// Similarity score.
    pub score: f32,
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