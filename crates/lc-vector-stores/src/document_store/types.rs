// lc-vector-stores/src/document_store/types.rs
//! Core types and traits for document storage.
//!
//! Defines `DocumentStore` trait and `ChunkedDocumentStoreTrait` trait
//! used by all storage backends. `ChunkDocument` and `Document` are
//! re-exported from `lc-shared`.

use crate::VectorStoreError;
use async_trait::async_trait;
use lc_shared::document::Document;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export ChunkDocument from lc-shared for backward compatibility
// (used by sibling modules via `super::types::ChunkDocument`)
pub use lc_shared::document::ChunkDocument;

// ============================================================================
// DocumentStore Trait
// ============================================================================

/// Document store trait
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// Adds a document
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError>;

    /// Adds documents in bulk
    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError>;

    /// Gets a document
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;

    /// Deletes a document
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;

    /// Gets the document count
    async fn count(&self) -> usize;

    /// Clears the store
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

// ============================================================================
// ChunkedDocumentStore Trait (abstract interface, supporting multiple storage backends)
// ============================================================================

/// Document store trait supporting a parent-child structure
///
/// This trait defines a unified interface for the back-reference store, supporting:
/// - MongoDB (production)
/// - InMemory (development/test)
/// - Redis (cache layer, reserved)
/// - SQLite (local storage, reserved)
#[async_trait]
pub trait ChunkedDocumentStoreTrait: Send + Sync {
    /// Adds a Parent document and automatically splits it into chunks
    ///
    /// # Arguments
    /// - `document`: the original document
    /// - `chunk_size`: the character size of each chunk
    ///
    /// # Returns
    /// - `(parent_id, chunk_ids)`: the Parent ID and the generated Chunk ID list
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;

    /// Adds Parent documents in bulk
    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError>;

    /// Gets a Parent document
    async fn get_parent_document(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// Gets a single Chunk
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError>;

    /// Gets a single Chunk (as a Document)
    async fn get_chunk_document(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// Gets all Chunks of a Parent
    async fn get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError>;

    /// Gets all Chunks of a Parent (as Documents)
    async fn get_chunk_documents_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<Document>, VectorStoreError>;

    /// Deletes a Parent document and all of its Chunks
    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError>;

    /// Gets the Parent document count
    async fn parent_count(&self) -> usize;

    /// Gets the Chunk document count
    async fn chunk_count(&self) -> usize;

    /// Gets all Chunks
    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError>;

    /// Clears all data
    async fn clear(&self) -> Result<(), VectorStoreError>;

    // ========================================================================
    // Blocking methods (synchronous versions, used in synchronous retrieval scenarios such as BM25)
    // ========================================================================

    /// Adds a Parent document (blocking version)
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;

    /// Gets a Parent document (blocking version)
    fn get_parent_document_blocking(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError>;

    /// Gets a single Chunk (blocking version)
    fn get_chunk_blocking(&self, chunk_id: &str)
        -> Result<Option<ChunkDocument>, VectorStoreError>;

    /// Gets all Chunks of a Parent (blocking version)
    fn blocking_get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError>;
}

// ============================================================================
// Internal serialization helper
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChunkedStoreData {
    pub parent_docs: HashMap<String, Document>,
    pub chunks: HashMap<String, ChunkDocument>,
    pub parent_to_chunks: HashMap<String, Vec<String>>,
}
