// lc-vector-stores/src/document_store/chunked.rs
//! In-memory chunked document store implementation.

use crate::document_store::types::{ChunkDocument, ChunkedDocumentStoreTrait, ChunkedStoreData};
use crate::{Document, VectorStoreError};
use async_trait::async_trait;
use lc_shared::splitter::{RecursiveCharacterSplitter, TextSplitter};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// InMemoryChunkedDocumentStore (in-memory implementation)
// ============================================================================

/// Converts a std lock poison into [`VectorStoreError::StorageError`], instead of an unwrap panic.
///
/// Q5: `std::sync::RwLock` becomes "poisoned" when its holder panics, after which every
/// read/write returns Err; the old `.read().unwrap()`/`.write().unwrap()` would panic directly
/// once poisoned. Here the poison is explicitly converted to a StorageError and propagated.
pub(crate) fn lock_error<T>(
    result: Result<T, std::sync::PoisonError<T>>,
) -> Result<T, VectorStoreError> {
    result
        .map_err(|_| VectorStoreError::StorageError("document store lock is poisoned".to_string()))
}

/// In-memory store implementation (development/test)
///
/// Q5: `std::sync::RwLock` is deliberately kept here (rather than `tokio::sync::RwLock` like
/// InMemoryVectorStore): the `_blocking` synchronous methods of `ChunkedDocumentStoreTrait`
/// are called from sync retrieval paths such as BM25 inside async contexts (see lc-rag's
/// hybrid retriever), and tokio's `blocking_read/blocking_write` panic when called in an
/// async context (see the tokio docs "Panics if called within an asynchronous execution
/// context"). Therefore only the `.unwrap()` lock-poison panic risk is removed; see `lock_error`.
pub struct InMemoryChunkedDocumentStore {
    pub(crate) parent_docs: Arc<std::sync::RwLock<HashMap<String, Document>>>,
    pub(crate) chunks: Arc<std::sync::RwLock<HashMap<String, ChunkDocument>>>,
    pub(crate) parent_to_chunks: Arc<std::sync::RwLock<HashMap<String, Vec<String>>>>,
}

impl InMemoryChunkedDocumentStore {
    /// Creates a new in-memory document store
    pub fn new() -> Self {
        Self {
            parent_docs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            chunks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            parent_to_chunks: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Synchronously gets the document for the given chunk
    pub fn get_chunk_document_blocking(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let chunks = lock_error(self.chunks.read())?;
        Ok(chunks.get(chunk_id).map(|c| c.to_document()))
    }

    fn split_and_store_chunks_blocking(
        &self,
        parent_id: &str,
        content: &str,
        chunk_size: usize,
    ) -> Result<Vec<String>, VectorStoreError> {
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks = splitter.split_text(content);

        let mut chunk_ids = Vec::new();

        // S3: split chunks inherit the parent document's metadata (for chunked-backend metadata filtering).
        let parent_meta = self
            .get_parent_document_blocking(parent_id)?
            .map(|d| d.metadata)
            .unwrap_or_default();

        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}::{}", parent_id, segment);

            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            )
            .with_metadata_map(parent_meta.clone());

            {
                let mut chunks_store = lock_error(self.chunks.write())?;
                chunks_store.insert(chunk_id.clone(), chunk);
            }

            {
                let mut mapping = lock_error(self.parent_to_chunks.write())?;
                mapping
                    .entry(parent_id.to_string())
                    .or_default()
                    .push(chunk_id.clone());
            }

            chunk_ids.push(chunk_id);
        }

        Ok(chunk_ids)
    }

    async fn split_and_store_chunks_async(
        &self,
        parent_id: &str,
        content: &str,
        chunk_size: usize,
    ) -> Result<Vec<String>, VectorStoreError> {
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks = splitter.split_text(content);

        let mut chunk_ids = Vec::new();

        // S3: split chunks inherit the parent document's metadata (for chunked-backend metadata filtering).
        let parent_meta = self
            .get_parent_document(parent_id)
            .await?
            .map(|d| d.metadata)
            .unwrap_or_default();

        // Acquire locks once for all chunks
        let mut chunks_store = lock_error(self.chunks.write())?;
        let mut mapping = lock_error(self.parent_to_chunks.write())?;

        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}::{}", parent_id, segment);

            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            )
            .with_metadata_map(parent_meta.clone());

            chunks_store.insert(chunk_id.clone(), chunk);

            mapping
                .entry(parent_id.to_string())
                .or_default()
                .push(chunk_id.clone());

            chunk_ids.push(chunk_id);
        }

        Ok(chunk_ids)
    }
}

impl Default for InMemoryChunkedDocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChunkedDocumentStoreTrait for InMemoryChunkedDocumentStore {
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        {
            let mut parents = lock_error(self.parent_docs.write())?;
            parents.insert(parent_id.clone(), document.clone());
        }

        let chunk_ids = self
            .split_and_store_chunks_async(&parent_id, &document.content, chunk_size)
            .await?;

        Ok((parent_id, chunk_ids))
    }

    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError> {
        let mut results = Vec::new();
        for doc in documents {
            let result = self.add_parent_document(doc, chunk_size).await?;
            results.push(result);
        }
        Ok(results)
    }

    async fn get_parent_document(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let parents = lock_error(self.parent_docs.read())?;
        Ok(parents.get(parent_id).cloned())
    }

    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = lock_error(self.chunks.read())?;
        Ok(chunks.get(chunk_id).cloned())
    }

    async fn get_chunk_document(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let chunks = lock_error(self.chunks.read())?;
        Ok(chunks.get(chunk_id).map(|c| c.to_document()))
    }

    async fn get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = lock_error(self.parent_to_chunks.read())?;
        let chunks = lock_error(self.chunks.read())?;

        let chunk_ids = mapping.get(parent_id).cloned().unwrap_or_default();

        let result = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();

        Ok(result)
    }

    async fn get_chunk_documents_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<Document>, VectorStoreError> {
        let chunks = self.get_chunks_for_parent(parent_id).await?;
        Ok(chunks.iter().map(|c| c.to_document()).collect())
    }

    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError> {
        let chunk_ids = {
            let mapping = lock_error(self.parent_to_chunks.read())?;
            mapping.get(parent_id).cloned().unwrap_or_default()
        };

        {
            let mut chunks = lock_error(self.chunks.write())?;
            for chunk_id in &chunk_ids {
                chunks.remove(chunk_id);
            }
        }

        {
            let mut mapping = lock_error(self.parent_to_chunks.write())?;
            mapping.remove(parent_id);
        }

        {
            let mut parents = lock_error(self.parent_docs.write())?;
            parents.remove(parent_id);
        }

        Ok(())
    }

    async fn parent_count(&self) -> usize {
        // count returns usize; on lock poison recover the inner value (only a count, not worth failing outright)
        self.parent_docs
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    async fn chunk_count(&self) -> usize {
        self.chunks
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let chunks = lock_error(self.chunks.read())?;
        Ok(chunks.values().cloned().collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut parents = lock_error(self.parent_docs.write())?;
        let mut chunks = lock_error(self.chunks.write())?;
        let mut mapping = lock_error(self.parent_to_chunks.write())?;

        parents.clear();
        chunks.clear();
        mapping.clear();

        Ok(())
    }

    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        {
            let mut parents = lock_error(self.parent_docs.write())?;
            parents.insert(parent_id.clone(), document.clone());
        }

        let chunk_ids =
            self.split_and_store_chunks_blocking(&parent_id, &document.content, chunk_size)?;

        Ok((parent_id, chunk_ids))
    }

    fn get_parent_document_blocking(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let parents = lock_error(self.parent_docs.read())?;
        Ok(parents.get(parent_id).cloned())
    }

    fn get_chunk_blocking(
        &self,
        chunk_id: &str,
    ) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = lock_error(self.chunks.read())?;
        Ok(chunks.get(chunk_id).cloned())
    }

    fn blocking_get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = lock_error(self.parent_to_chunks.read())?;
        let chunks = lock_error(self.chunks.read())?;

        let chunk_ids = mapping.get(parent_id).cloned().unwrap_or_default();

        let result = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();

        Ok(result)
    }
}

impl InMemoryChunkedDocumentStore {
    /// Serializes the in-memory parent documents and child chunks (bincode) to disk.
    ///
    /// C3: the default `save/load` on `ChunkedDocumentStoreTrait` only returned a
    /// "not implemented" runtime error and has been removed from the trait; persistence
    /// is now exposed through each backend's own inherent methods. This method is the
    /// real implementation for the InMemory backend, called directly on the concrete type.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<(), VectorStoreError> {
        let parents = lock_error(self.parent_docs.read())?;
        let chunks = lock_error(self.chunks.read())?;
        let mapping = lock_error(self.parent_to_chunks.read())?;

        let data = ChunkedStoreData {
            parent_docs: parents.clone(),
            chunks: chunks.clone(),
            parent_to_chunks: mapping.clone(),
        };

        let encoded =
            bincode::serialize(&data).map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        std::fs::write(path.as_ref(), encoded)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Rebuilds the store by deserializing a file written by [`save`](Self::save), preserving the parent-child relationships.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self, VectorStoreError> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let data: ChunkedStoreData = bincode::deserialize(&bytes)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        Ok(Self {
            parent_docs: Arc::new(std::sync::RwLock::new(data.parent_docs)),
            chunks: Arc::new(std::sync::RwLock::new(data.chunks)),
            parent_to_chunks: Arc::new(std::sync::RwLock::new(data.parent_to_chunks)),
        })
    }
}

/// Type alias for `InMemoryChunkedDocumentStore`
pub type ChunkedDocumentStore = InMemoryChunkedDocumentStore;
