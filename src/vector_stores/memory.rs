// src/vector_stores/memory.rs
//! 内存向量存储
//!
//! 将文档和向量存储在内存中，适用于小规模数据和测试。

use super::{Document, SearchResult, VectorDocument, VectorStore, VectorStoreError};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// 内存向量存储
pub struct InMemoryVectorStore {
    /// 文档存储
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,
}

impl InMemoryVectorStore {
    /// 创建新的内存向量存储
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 计算余弦相似度
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "文档数量和嵌入向量数量不匹配".to_string()
            ));
        }
        
        let mut store = self.documents.write().await;
        let mut ids = Vec::new();
        
        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            
            let vector_doc = VectorDocument {
                document: Document {
                    id: Some(id.clone()),
                    content: doc.content,
                    metadata: doc.metadata,
                },
                embedding,
            };
            
            store.insert(id.clone(), vector_doc);
            ids.push(id);
        }
        
        Ok(ids)
    }
    
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let store = self.documents.read().await;
        
        // 计算所有文档的相似度
        let mut results: Vec<SearchResult> = store
            .values()
            .map(|vd| {
                let score = Self::cosine_similarity(query_embedding, &vd.embedding);
                SearchResult {
                    document: vd.document.clone(),
                    score,
                }
            })
            .collect();
        
        // 按相似度降序排序
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // 返回前 k 个结果
        Ok(results.into_iter().take(k).collect())
    }
    
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.document.clone()))
    }
    
    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.embedding.clone()))
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_add_and_search() {
        let store = InMemoryVectorStore::new();
        
        // 添加文档
        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];
        
        // 创建简单的模拟嵌入向量
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],  // Rust 相关
            vec![0.0, 1.0, 0.0],  // Python 相关
            vec![0.0, 0.0, 1.0],  // JavaScript 相关
        ];
        
        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.count().await, 3);
        
        // 搜索相似文档
        let query = vec![0.9, 0.1, 0.0];  // 更接近 Rust
        let results = store.similarity_search(&query, 2).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("Rust"));
        assert!(results[0].score > results[1].score);
    }
    
    #[tokio::test]
    async fn test_get_and_delete() {
        let store = InMemoryVectorStore::new();
        
        let doc = Document::new("Test document").with_id("test-id");
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        
        store.add_documents(vec![doc], embeddings).await.unwrap();
        
        // 获取文档
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test document");
        
        // 删除文档
        store.delete_document("test-id").await.unwrap();
        assert_eq!(store.count().await, 0);
        
        // 再次获取应该返回 None
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_none());
    }
    
    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryVectorStore::new();
        
        let docs = vec![
            Document::new("Doc 1"),
            Document::new("Doc 2"),
        ];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        
        store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(store.count().await, 2);
        
        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }
    
    #[test]
    fn test_cosine_similarity() {
        // 相同向量
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((InMemoryVectorStore::cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
        
        // 正交向量
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((InMemoryVectorStore::cosine_similarity(&a, &b) - 0.0).abs() < 0.0001);
    }
}