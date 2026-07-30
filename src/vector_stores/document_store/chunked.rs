// src/vector_stores/document_store/chunked.rs
//! In-memory chunked document store implementation.

use super::super::{Document, VectorStoreError};
use super::types::{ChunkDocument, ChunkedDocumentStoreTrait, ChunkedStoreData};
use crate::retrieval::{RecursiveCharacterSplitter, TextSplitter};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// InMemoryChunkedDocumentStore（内存实现）
// ============================================================================

/// 内存存储实现（开发/测试用）
pub struct InMemoryChunkedDocumentStore {
    pub(crate) parent_docs: Arc<std::sync::RwLock<HashMap<String, Document>>>,
    pub(crate) chunks: Arc<std::sync::RwLock<HashMap<String, ChunkDocument>>>,
    pub(crate) parent_to_chunks: Arc<std::sync::RwLock<HashMap<String, Vec<String>>>>,
}

impl InMemoryChunkedDocumentStore {
    pub fn new() -> Self {
        Self {
            parent_docs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            chunks: Arc::new(std::sync::RwLock::new(HashMap::new())),
            parent_to_chunks: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    pub fn get_chunk_document_blocking(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let chunks = self.chunks.read().unwrap();
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

        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}::{}", parent_id, segment);

            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            );

            {
                let mut chunks_store = self.chunks.write().unwrap();
                chunks_store.insert(chunk_id.clone(), chunk);
            }

            {
                let mut mapping = self.parent_to_chunks.write().unwrap();
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

        // Acquire locks once for all chunks
        let mut chunks_store = self.chunks.write().unwrap();
        let mut mapping = self.parent_to_chunks.write().unwrap();

        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}::{}", parent_id, segment);

            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            );

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
            let mut parents = self.parent_docs.write().unwrap();
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
        let parents = self.parent_docs.read().unwrap();
        Ok(parents.get(parent_id).cloned())
    }

    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.read().unwrap();
        Ok(chunks.get(chunk_id).cloned())
    }

    async fn get_chunk_document(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let chunks = self.chunks.read().unwrap();
        Ok(chunks.get(chunk_id).map(|c| c.to_document()))
    }

    async fn get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = self.parent_to_chunks.read().unwrap();
        let chunks = self.chunks.read().unwrap();

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
            let mapping = self.parent_to_chunks.read().unwrap();
            mapping.get(parent_id).cloned().unwrap_or_default()
        };

        {
            let mut chunks = self.chunks.write().unwrap();
            for chunk_id in &chunk_ids {
                chunks.remove(chunk_id);
            }
        }

        {
            let mut mapping = self.parent_to_chunks.write().unwrap();
            mapping.remove(parent_id);
        }

        {
            let mut parents = self.parent_docs.write().unwrap();
            parents.remove(parent_id);
        }

        Ok(())
    }

    async fn parent_count(&self) -> usize {
        let parents = self.parent_docs.read().unwrap();
        parents.len()
    }

    async fn chunk_count(&self) -> usize {
        let chunks = self.chunks.read().unwrap();
        chunks.len()
    }

    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.read().unwrap();
        Ok(chunks.values().cloned().collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut parents = self.parent_docs.write().unwrap();
        let mut chunks = self.chunks.write().unwrap();
        let mut mapping = self.parent_to_chunks.write().unwrap();

        parents.clear();
        chunks.clear();
        mapping.clear();

        Ok(())
    }

    async fn save(&self, path: impl AsRef<Path> + Send) -> Result<(), VectorStoreError> {
        let parents = self.parent_docs.read().unwrap();
        let chunks = self.chunks.read().unwrap();
        let mapping = self.parent_to_chunks.read().unwrap();

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

    async fn load(path: impl AsRef<Path> + Send) -> Result<Self, VectorStoreError> {
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
            let mut parents = self.parent_docs.write().unwrap();
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
        let parents = self.parent_docs.read().unwrap();
        Ok(parents.get(parent_id).cloned())
    }

    fn get_chunk_blocking(
        &self,
        chunk_id: &str,
    ) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.read().unwrap();
        Ok(chunks.get(chunk_id).cloned())
    }

    fn blocking_get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = self.parent_to_chunks.read().unwrap();
        let chunks = self.chunks.read().unwrap();

        let chunk_ids = mapping.get(parent_id).cloned().unwrap_or_default();

        let result = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();

        Ok(result)
    }
}

pub type ChunkedDocumentStore = InMemoryChunkedDocumentStore;
