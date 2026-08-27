// lc-vector-stores/src/document_store/store.rs
//! In-memory document store implementation.

use crate::document_store::chunked::{lock_error, InMemoryChunkedDocumentStore};
use crate::document_store::types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use crate::{Document, VectorStoreError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// InMemoryDocumentStore
// ============================================================================

/// In-memory document store
///
/// Q5: uses `tokio::sync::RwLock` (consistent with InMemoryVectorStore), methods `.await`
/// directly without blocking the executor; and there are no synchronous `_blocking` methods
/// called from async contexts, so there is no `blocking_read/write` panic constraint.
pub struct InMemoryDocumentStore {
    /// Document collection
    documents: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocumentStore {
    /// Creates a new in-memory document store
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryDocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentStore for InMemoryDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let mut store = self.documents.write().await;
        store.insert(id.clone(), document);

        Ok(id)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut store = self.documents.write().await;
        let mut ids = Vec::new();

        for doc in documents {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            store.insert(id.clone(), doc);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).cloned())
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

// ============================================================================
// DocumentStore impl for InMemoryChunkedDocumentStore
// ============================================================================

#[async_trait]
impl DocumentStore for InMemoryChunkedDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Write to parent_docs so the document is retrievable as a parent
        {
            let mut parents = lock_error(self.parent_docs.write())?;
            parents.insert(id.clone(), document.clone());
        }

        let mut chunks = lock_error(self.chunks.write())?;

        // S3: chunks inherit the parent document's metadata; otherwise the document metadata
        // returned by get_chunk_document is empty and chunked-backend metadata filtering (and
        // any metadata-based retrieval) would mismatch.
        let chunk = ChunkDocument::new(id.clone(), id.clone(), document.content.clone(), 0)
            .with_metadata_map(document.metadata.clone());

        chunks.insert(id.clone(), chunk);

        // Also update parent_to_chunks mapping
        {
            let mut mapping = lock_error(self.parent_to_chunks.write())?;
            mapping.entry(id.clone()).or_default().push(id.clone());
        }

        Ok(id)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut ids = Vec::new();
        for doc in documents {
            let id = self.add_document(doc).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        self.get_chunk_document(id).await
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut chunks = lock_error(self.chunks.write())?;
        chunks.remove(id);
        Ok(())
    }

    async fn count(&self) -> usize {
        self.chunk_count().await
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        ChunkedDocumentStoreTrait::clear(self).await
    }
}
