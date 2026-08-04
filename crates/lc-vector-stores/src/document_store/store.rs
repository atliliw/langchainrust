// lc-vector-stores/src/document_store/store.rs
//! In-memory document store implementation.

use crate::document_store::chunked::InMemoryChunkedDocumentStore;
use crate::document_store::types::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use crate::{Document, VectorStoreError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// InMemoryDocumentStore
// ============================================================================

/// 内存文档存储
pub struct InMemoryDocumentStore {
    /// 文档集合
    documents: Arc<std::sync::RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocumentStore {
    /// 创建新的内存文档存储
    pub fn new() -> Self {
        Self {
            documents: Arc::new(std::sync::RwLock::new(HashMap::new())),
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

        let mut store = self.documents.write().unwrap();
        store.insert(id.clone(), document);

        Ok(id)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut store = self.documents.write().unwrap();
        let mut ids = Vec::new();

        for doc in documents {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            store.insert(id.clone(), doc);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let store = self.documents.read().unwrap();
        Ok(store.get(id).cloned())
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().unwrap();
        store.remove(id);
        Ok(())
    }

    async fn count(&self) -> usize {
        let store = self.documents.read().unwrap();
        store.len()
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().unwrap();
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
            let mut parents = self.parent_docs.write().unwrap();
            parents.insert(id.clone(), document.clone());
        }

        let mut chunks = self.chunks.write().unwrap();

        let chunk = ChunkDocument::new(id.clone(), id.clone(), document.content.clone(), 0);

        chunks.insert(id.clone(), chunk);

        // Also update parent_to_chunks mapping
        {
            let mut mapping = self.parent_to_chunks.write().unwrap();
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
        let mut chunks = self.chunks.write().unwrap();
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
