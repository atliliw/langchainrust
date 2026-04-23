// src/vector_stores/document_store.rs
//! 文档存储模块
//!
//! 单独存储文档内容，供 BM25 和向量检索共用。
//! 支持原始文档和分割后的 chunk 文档。

use super::{Document, VectorStoreError};
use async_trait::async_trait;
use crate::retrieval::{RecursiveCharacterSplitter, TextSplitter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// ============================================================================
// Chunk 文档结构
// ============================================================================

/// Chunk 文档（分割后的文档片段）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDocument {
    /// Chunk ID
    pub chunk_id: String,
    
    /// 原始文档 ID（Parent ID）
    pub parent_id: String,
    
    /// Chunk 内容
    pub content: String,
    
    /// Chunk 序号（第几个 chunk）
    pub segment: usize,
    
    /// Chunk 元数据
    pub metadata: HashMap<String, String>,
}

impl ChunkDocument {
    /// 创建新的 Chunk 文档
    pub fn new(
        chunk_id: String,
        parent_id: String,
        content: String,
        segment: usize,
    ) -> Self {
        Self {
            chunk_id,
            parent_id,
            content,
            segment,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 转换为 Document
    pub fn to_document(&self) -> Document {
        Document {
            content: self.content.clone(),
            metadata: self.metadata.clone(),
            id: Some(self.chunk_id.clone()),
        }
    }
}

// ============================================================================
// DocumentStore Trait
// ============================================================================

/// 文档存储 trait
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// 添加文档
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError>;
    
    /// 批量添加文档
    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError>;
    
    /// 获取文档
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;
    
    /// 删除文档
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;
    
    /// 获取文档数量
    async fn count(&self) -> usize;
    
    /// 清空存储
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

// ============================================================================
// ChunkedDocumentStore Trait（抽象接口，支持多种存储后端）
// ============================================================================

/// 支持 Parent-Child 结构的文档存储 trait
///
/// 此 trait 定义了回表存储的统一接口，支持：
/// - MongoDB（生产环境）
/// - InMemory（开发/测试）
/// - Redis（缓存层，预留）
/// - SQLite（本地存储，预留）
#[async_trait]
pub trait ChunkedDocumentStoreTrait: Send + Sync {
    /// 添加 Parent 文档并自动分割为 chunks
    ///
    /// # 参数
    /// - `document`: 原始文档
    /// - `chunk_size`: 每个 chunk 的字符大小
    ///
    /// # 返回
    /// - `(parent_id, chunk_ids)`: Parent ID 和生成的 Chunk ID 列表
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;
    
    /// 批量添加 Parent 文档
    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError>;
    
    /// 获取 Parent 文档
    async fn get_parent_document(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError>;
    
    /// 获取单个 Chunk
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError>;
    
    /// 获取单个 Chunk（转为 Document）
    async fn get_chunk_document(&self, chunk_id: &str) -> Result<Option<Document>, VectorStoreError>;
    
    /// 获取 Parent 的所有 Chunks
    async fn get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError>;
    
    /// 获取 Parent 的所有 Chunks（转为 Document）
    async fn get_chunk_documents_for_parent(&self, parent_id: &str) -> Result<Vec<Document>, VectorStoreError>;
    
    /// 删除 Parent 文档及其所有 Chunks
    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError>;
    
    /// 获取 Parent 文档数量
    async fn parent_count(&self) -> usize;
    
    /// 获取 Chunk 文档数量
    async fn chunk_count(&self) -> usize;
    
    /// 获取所有 Chunks
    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError>;
    
    /// 清空所有数据
    async fn clear(&self) -> Result<(), VectorStoreError>;
    
/// 持久化存储（可选实现）
    async fn save(&self, _path: impl AsRef<Path> + Send) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::StorageError("save not implemented for this store".to_string()))
    }
    
    async fn load(_path: impl AsRef<Path> + Send) -> Result<Self, VectorStoreError> where Self: Sized {
        Err(VectorStoreError::StorageError("load not implemented for this store".to_string()))
    }
    
    // ========================================================================
    // Blocking 方法（同步版本，用于 BM25 等同步检索场景）
    // ========================================================================
    
    /// 添加 Parent 文档（阻塞版本）
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError>;
    
    /// 获取 Parent 文档（阻塞版本）
    fn get_parent_document_blocking(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError>;
    
    /// 获取单个 Chunk（阻塞版本）
    fn get_chunk_blocking(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError>;
    
    /// 获取 Parent 的所有 Chunks（阻塞版本）
    fn blocking_get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError>;
}

// ============================================================================
// InMemoryDocumentStore
// ============================================================================

/// 内存文档存储
pub struct InMemoryDocumentStore {
    /// 文档集合
    documents: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocumentStore {
    /// 创建新的内存文档存储
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
        let id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        let mut store = self.documents.write().await;
        store.insert(id.clone(), document);
        
        Ok(id)
    }
    
    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError> {
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
// InMemoryChunkedDocumentStore（内存实现）
// ============================================================================

/// 内存存储实现（开发/测试用）
pub struct InMemoryChunkedDocumentStore {
    parent_docs: Arc<RwLock<HashMap<String, Document>>>,
    chunks: Arc<RwLock<HashMap<String, ChunkDocument>>>,
    parent_to_chunks: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl InMemoryChunkedDocumentStore {
    pub fn new() -> Self {
        Self {
            parent_docs: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            parent_to_chunks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn get_chunk_document_blocking(&self, chunk_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let chunks = self.chunks.blocking_read();
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
            let chunk_id = format!("{}_{}", parent_id, segment);
            
            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            );
            
            {
                let mut chunks_store = self.chunks.blocking_write();
                chunks_store.insert(chunk_id.clone(), chunk);
            }
            
            {
                let mut mapping = self.parent_to_chunks.blocking_write();
                mapping
                    .entry(parent_id.to_string())
                    .or_insert_with(Vec::new)
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
        
        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}_{}", parent_id, segment);
            
            let chunk = ChunkDocument::new(
                chunk_id.clone(),
                parent_id.to_string(),
                chunk_content,
                segment,
            );
            
            {
                let mut chunks_store = self.chunks.write().await;
                chunks_store.insert(chunk_id.clone(), chunk);
            }
            
            {
                let mut mapping = self.parent_to_chunks.write().await;
                mapping
                    .entry(parent_id.to_string())
                    .or_insert_with(Vec::new)
                    .push(chunk_id.clone());
            }
            
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
        let parent_id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        {
            let mut parents = self.parent_docs.write().await;
            parents.insert(parent_id.clone(), document.clone());
        }
        
        let chunk_ids = self.split_and_store_chunks_async(&parent_id, &document.content, chunk_size).await?;
        
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
    
    async fn get_parent_document(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let parents = self.parent_docs.read().await;
        Ok(parents.get(parent_id).cloned())
    }
    
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.read().await;
        Ok(chunks.get(chunk_id).cloned())
    }
    
    async fn get_chunk_document(&self, chunk_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let chunks = self.chunks.read().await;
        Ok(chunks.get(chunk_id).map(|c| c.to_document()))
    }
    
    async fn get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = self.parent_to_chunks.read().await;
        let chunks = self.chunks.read().await;
        
        let chunk_ids = mapping.get(parent_id).cloned().unwrap_or_default();
        
        let result = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();
        
        Ok(result)
    }
    
    async fn get_chunk_documents_for_parent(&self, parent_id: &str) -> Result<Vec<Document>, VectorStoreError> {
        let chunks = self.get_chunks_for_parent(parent_id).await?;
        Ok(chunks.iter().map(|c| c.to_document()).collect())
    }
    
    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError> {
        let chunk_ids = {
            let mapping = self.parent_to_chunks.read().await;
            mapping.get(parent_id).cloned().unwrap_or_default()
        };
        
        {
            let mut chunks = self.chunks.write().await;
            for chunk_id in &chunk_ids {
                chunks.remove(chunk_id);
            }
        }
        
        {
            let mut mapping = self.parent_to_chunks.write().await;
            mapping.remove(parent_id);
        }
        
        {
            let mut parents = self.parent_docs.write().await;
            parents.remove(parent_id);
        }
        
        Ok(())
    }
    
    async fn parent_count(&self) -> usize {
        let parents = self.parent_docs.read().await;
        parents.len()
    }
    
    async fn chunk_count(&self) -> usize {
        let chunks = self.chunks.read().await;
        chunks.len()
    }
    
    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.read().await;
        Ok(chunks.values().cloned().collect())
    }
    
    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut parents = self.parent_docs.write().await;
        let mut chunks = self.chunks.write().await;
        let mut mapping = self.parent_to_chunks.write().await;
        
        parents.clear();
        chunks.clear();
        mapping.clear();
        
        Ok(())
    }
    
    async fn save(&self, path: impl AsRef<Path> + Send) -> Result<(), VectorStoreError> {
        let parents = self.parent_docs.read().await;
        let chunks = self.chunks.read().await;
        let mapping = self.parent_to_chunks.read().await;
        
        let data = ChunkedStoreData {
            parent_docs: parents.clone(),
            chunks: chunks.clone(),
            parent_to_chunks: mapping.clone(),
        };
        
        let encoded = bincode::serialize(&data)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
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
            parent_docs: Arc::new(RwLock::new(data.parent_docs)),
            chunks: Arc::new(RwLock::new(data.chunks)),
            parent_to_chunks: Arc::new(RwLock::new(data.parent_to_chunks)),
        })
    }
    
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let parent_id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        
        {
            let mut parents = self.parent_docs.blocking_write();
            parents.insert(parent_id.clone(), document.clone());
        }
        
        let chunk_ids = self.split_and_store_chunks_blocking(&parent_id, &document.content, chunk_size)?;
        
        Ok((parent_id, chunk_ids))
    }
    
    fn get_parent_document_blocking(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let parents = self.parent_docs.blocking_read();
        Ok(parents.get(parent_id).cloned())
    }
    
    fn get_chunk_blocking(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let chunks = self.chunks.blocking_read();
        Ok(chunks.get(chunk_id).cloned())
    }
    
    fn blocking_get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mapping = self.parent_to_chunks.blocking_read();
        let chunks = self.chunks.blocking_read();
        
        let chunk_ids = mapping.get(parent_id).cloned().unwrap_or_default();
        
        let result = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();
        
        Ok(result)
    }
}

#[async_trait]
impl DocumentStore for InMemoryChunkedDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        
        let mut chunks = self.chunks.write().await;
        
        let chunk = ChunkDocument::new(
            id.clone(),
            id.clone(),
            document.content.clone(),
            0,
        );
        
        chunks.insert(id.clone(), chunk);
        
        Ok(id)
    }
    
    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError> {
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
        let mut chunks = self.chunks.write().await;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkedStoreData {
    parent_docs: HashMap<String, Document>,
    chunks: HashMap<String, ChunkDocument>,
    parent_to_chunks: HashMap<String, Vec<String>>,
}

pub type ChunkedDocumentStore = InMemoryChunkedDocumentStore;

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_in_memory_document_store() {
        let store = InMemoryDocumentStore::new();
        
        // 添加文档
        let doc = Document::new("测试内容").with_id("doc_001");
        let id = store.add_document(doc).await.unwrap();
        
        assert_eq!(id, "doc_001");
        assert_eq!(store.count().await, 1);
        
        // 获取文档
        let retrieved = store.get_document("doc_001").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "测试内容");
        
        // 删除文档
        store.delete_document("doc_001").await.unwrap();
        assert_eq!(store.count().await, 0);
    }
    
    #[tokio::test]
    async fn test_chunked_document_store() {
        let store = ChunkedDocumentStore::new();
        
        // 添加 Parent 文档（chunk_size=20）
        let doc = Document::new("这是一段很长的测试文本，用于验证文档分割功能。").with_id("parent_001");
        
        let (parent_id, chunk_ids) = store.add_parent_document(doc, 20).await.unwrap();
        
        assert_eq!(parent_id, "parent_001");
        assert!(chunk_ids.len() > 1);  // 应该分割成多个 chunk
        
        // 获取 Parent 文档
        let parent = store.get_parent_document("parent_001").await.unwrap();
        assert!(parent.is_some());
        
        // 获取所有 Chunk
        let chunks = store.get_chunks_for_parent("parent_001").await.unwrap();
        assert_eq!(chunks.len(), chunk_ids.len());
        
        // 获取单个 Chunk
        let chunk = store.get_chunk(&chunk_ids[0]).await.unwrap();
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap().parent_id, "parent_001");
        
        // 删除 Parent 及所有 Chunk
        store.delete_parent_document("parent_001").await.unwrap();
        assert_eq!(store.parent_count().await, 0);
        assert_eq!(store.chunk_count().await, 0);
    }
    
    #[tokio::test]
    async fn test_chunk_to_document() {
        let chunk = ChunkDocument::new(
            "chunk_001".to_string(),
            "parent_001".to_string(),
            "Chunk内容".to_string(),
            0,
        ).with_metadata("source", "test");
        
        let doc = chunk.to_document();
        
        assert_eq!(doc.id, Some("chunk_001".to_string()));
        assert_eq!(doc.content, "Chunk内容");
        assert_eq!(doc.metadata.get("source"), Some(&"test".to_string()));
    }
    
    #[tokio::test]
    async fn test_persistence() {
        let store = ChunkedDocumentStore::new();
        
        // 添加文档
        let doc = Document::new("测试持久化功能的内容").with_id("parent_001");
        store.add_parent_document(doc, 10).await.unwrap();
        
        // 保存
        let temp_path = tempfile::NamedTempFile::new().unwrap();
        store.save(temp_path.path()).await.unwrap();
        
        // 加载
        let loaded = ChunkedDocumentStore::load(temp_path.path()).await.unwrap();
        
        assert_eq!(loaded.parent_count().await, store.parent_count().await);
        assert_eq!(loaded.chunk_count().await, store.chunk_count().await);
        
        let parent = loaded.get_parent_document("parent_001").await.unwrap();
        assert!(parent.is_some());
    }
}