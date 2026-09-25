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

        // Also update parent_to_chunks mapping. H7: idempotent — re-adding the same
        // document id must not push a duplicate mapping, else get_chunks_for_parent
        // returns the same chunk twice (duplicate document).
        {
            let mut mapping = lock_error(self.parent_to_chunks.write())?;
            let ids = mapping.entry(id.clone()).or_default();
            if !ids.contains(&id) {
                ids.push(id.clone());
            }
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
        // H7: the plain DocumentStore path also registers the id as a single-chunk
        // parent (parent_to_chunks / parent_docs), so clean all three structures —
        // mirroring delete_parent_document. Otherwise residue survives (parent_count()
        // inflated, re-add re-pushes a duplicate mapping).
        let chunk_ids = {
            let mapping = lock_error(self.parent_to_chunks.read())?;
            mapping.get(id).cloned().unwrap_or_default()
        };

        {
            let mut chunks = lock_error(self.chunks.write())?;
            for chunk_id in &chunk_ids {
                chunks.remove(chunk_id);
            }
        }

        {
            let mut mapping = lock_error(self.parent_to_chunks.write())?;
            mapping.remove(id);
        }

        {
            let mut parents = lock_error(self.parent_docs.write())?;
            parents.remove(id);
        }

        Ok(())
    }

    async fn count(&self) -> usize {
        self.chunk_count().await
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        ChunkedDocumentStoreTrait::clear(self).await
    }
}
