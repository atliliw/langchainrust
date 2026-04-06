// src/vector_stores/mod.rs
//! 向量存储实现
//!
//! 提供文档向量存储和检索功能。

mod memory;

pub use memory::InMemoryVectorStore;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

/// 向量存储错误类型
#[derive(Debug)]
pub enum VectorStoreError {
    /// 文档不存在
    DocumentNotFound(String),
    
    /// 嵌入错误
    EmbeddingError(String),
    
    /// 存储错误
    StorageError(String),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::DocumentNotFound(id) => write!(f, "文档不存在: {}", id),
            VectorStoreError::EmbeddingError(msg) => write!(f, "嵌入错误: {}", msg),
            VectorStoreError::StorageError(msg) => write!(f, "存储错误: {}", msg),
        }
    }
}

impl Error for VectorStoreError {}

/// 文档结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// 文档内容
    pub content: String,
    
    /// 文档元数据
    pub metadata: HashMap<String, String>,
    
    /// 文档 ID（可选）
    pub id: Option<String>,
}

impl Document {
    /// 创建新文档
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
            id: None,
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 设置 ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    
    /// 获取页面内容（别名）
    pub fn page_content(&self) -> &str {
        &self.content
    }
}

/// 向量文档（带嵌入向量）
#[derive(Debug, Clone)]
pub struct VectorDocument {
    /// 文档
    pub document: Document,
    
    /// 嵌入向量
    pub embedding: Vec<f32>,
}

/// 检索结果
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 文档
    pub document: Document,
    
    /// 相似度分数
    pub score: f32,
}

/// 向量存储 trait
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// 添加文档
    ///
    /// # 参数
    /// * `documents` - 文档列表
    /// * `embeddings` - 文档的嵌入向量列表
    ///
    /// # 返回
    /// 文档 ID 列表
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError>;
    
    /// 检索相似文档
    ///
    /// # 参数
    /// * `query_embedding` - 查询向量
    /// * `k` - 返回的文档数量
    ///
    /// # 返回
    /// 相似文档列表（按相似度降序）
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;
    
    /// 根据 ID 获取文档
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError>;
    
    /// 删除文档
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError>;
    
    /// 获取文档数量
    async fn count(&self) -> usize;
    
    /// 清空存储
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