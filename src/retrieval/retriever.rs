// src/retrieval/retriever.rs
//! 检索器实现
//!
//! 提供基于相似度的文档检索功能。

use crate::embeddings::Embeddings;
use crate::vector_stores::{Document, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use std::sync::Arc;

/// 检索器错误类型
#[derive(Debug)]
pub enum RetrieverError {
    /// 向量存储错误
    StoreError(VectorStoreError),
    
    /// 嵌入错误
    EmbeddingError(String),
    
    /// 无结果
    NoResults,
}

impl std::fmt::Display for RetrieverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrieverError::StoreError(e) => write!(f, "存储错误: {}", e),
            RetrieverError::EmbeddingError(msg) => write!(f, "嵌入错误: {}", msg),
            RetrieverError::NoResults => write!(f, "没有找到相关文档"),
        }
    }
}

impl std::error::Error for RetrieverError {}

impl From<VectorStoreError> for RetrieverError {
    fn from(e: VectorStoreError) -> Self {
        RetrieverError::StoreError(e)
    }
}

/// 检索器 trait
#[async_trait]
pub trait RetrieverTrait: Send + Sync {
    /// 检索相关文档
    ///
    /// # 参数
    /// * `query` - 查询文本
    /// * `k` - 返回的文档数量
    ///
    /// # 返回
    /// 相关文档列表
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError>;
    
    /// 检索相关文档（带分数）
    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError>;
    
    /// 添加文档
    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError>;
}

/// 基于相似度的检索器
pub struct SimilarityRetriever {
    /// 向量存储
    store: Arc<dyn VectorStore>,
    
    /// 嵌入模型
    embeddings: Arc<dyn Embeddings>,
}

impl SimilarityRetriever {
    /// 创建新的相似度检索器
    pub fn new(store: Arc<dyn VectorStore>, embeddings: Arc<dyn Embeddings>) -> Self {
        Self { store, embeddings }
    }
}

#[async_trait]
impl RetrieverTrait for SimilarityRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>, RetrieverError> {
        let results = self.retrieve_with_scores(query, k).await?;
        Ok(results.into_iter().map(|r| r.document).collect())
    }
    
    async fn retrieve_with_scores(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>, RetrieverError> {
        // 生成查询向量
        let query_embedding = self.embeddings
            .embed_query(query)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        
        // 检索相似文档
        let results = self.store
            .similarity_search(&query_embedding, k)
            .await?;
        
        Ok(results)
    }
    
    async fn add_documents(&self, documents: Vec<Document>) -> Result<(), RetrieverError> {
        // 生成文档嵌入
        let texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.embeddings
            .embed_documents(&texts)
            .await
            .map_err(|e| RetrieverError::EmbeddingError(e.to_string()))?;
        
        // 添加到存储
        self.store.add_documents(documents, embeddings).await?;
        
        Ok(())
    }
}

/// 简化的 Retriever 类型别名（用于快速使用）
pub type Retriever = SimilarityRetriever;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbeddings;
    use crate::vector_stores::InMemoryVectorStore;
    
    #[tokio::test]
    async fn test_retriever() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(128));
        
        let retriever = SimilarityRetriever::new(store.clone(), embeddings.clone());
        
        // 添加文档
        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];
        
        retriever.add_documents(docs).await.unwrap();
        assert_eq!(store.count().await, 3);
        
        // 检索文档
        let results = retriever.retrieve("programming language", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }
    
    #[tokio::test]
    async fn test_retriever_with_scores() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(MockEmbeddings::new(64));
        
        let retriever = SimilarityRetriever::new(store, embeddings);
        
        let docs = vec![
            Document::new("Document A"),
            Document::new("Document B"),
        ];
        
        retriever.add_documents(docs).await.unwrap();
        
        let results = retriever.retrieve_with_scores("query", 2).await.unwrap();
        assert_eq!(results.len(), 2);
        
        // 结果应该包含分数
        assert!(results[0].score >= -1.0 && results[0].score <= 1.0);
    }
}