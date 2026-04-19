// src/vector_stores/chunked_vector_store.rs
//! Chunked Vector Store - 分割文档向量存储
//!
//! 只存储向量 + chunk_id 引用，内容从 DocumentStore 获取。
//! 支持 Parent-Child 文档结构，适合长文档分割场景。

use super::document_store::{ChunkedDocumentStore, ChunkedDocumentStoreTrait, DocumentStore};
use super::{Document, SearchResult, VectorStore, VectorStoreError};
use async_trait::async_trait;
use futures_util::future;
use std::sync::Arc;
use tokio::sync::RwLock;

/// 向量索引条目（只存向量 + chunk_id）
struct VectorEntry {
    chunk_id: String,
    embedding: Vec<f32>,
}

/// Chunked Vector Store
pub struct ChunkedVectorStore {
    document_store: Arc<ChunkedDocumentStore>,
    vectors: Arc<RwLock<Vec<VectorEntry>>>,
    vector_size: usize,
}

impl ChunkedVectorStore {
    pub fn new(document_store: Arc<ChunkedDocumentStore>, vector_size: usize) -> Self {
        Self {
            document_store,
            vectors: Arc::new(RwLock::new(Vec::new())),
            vector_size,
        }
    }
    
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
    
    /// 添加 chunk 向量（chunk_id + embedding）
    pub async fn add_chunk_vector(
        &self,
        chunk_id: String,
        embedding: Vec<f32>,
    ) -> Result<(), VectorStoreError> {
        if embedding.len() != self.vector_size {
            return Err(VectorStoreError::StorageError(format!(
                "向量维度不匹配: 期望 {}, 实际 {}",
                self.vector_size,
                embedding.len()
            )));
        }
        
        let mut vectors = self.vectors.write().await;
        vectors.push(VectorEntry { chunk_id, embedding });
        
        Ok(())
    }
    
    /// 批量添加 chunk 向量
    pub async fn add_chunk_vectors(
        &self,
        chunk_ids: Vec<String>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<(), VectorStoreError> {
        if chunk_ids.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "chunk_id 数量和向量数量不匹配".to_string()
            ));
        }
        
        for (chunk_id, embedding) in chunk_ids.into_iter().zip(embeddings.into_iter()) {
            self.add_chunk_vector(chunk_id, embedding).await?;
        }
        
        Ok(())
    }
    
    /// 从 Parent 文档添加（自动分割 + 向量化）
    pub async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
        embeddings_fn: impl Fn(&str) -> Vec<f32>,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let (parent_id, chunk_ids) = self.document_store
            .add_parent_document(document, chunk_size)
            .await?;
        
        for chunk_id in &chunk_ids {
            let chunk = self.document_store.get_chunk(chunk_id).await?
                .ok_or_else(|| VectorStoreError::DocumentNotFound(chunk_id.clone()))?;
            
            let embedding = embeddings_fn(&chunk.content);
            self.add_chunk_vector(chunk_id.clone(), embedding).await?;
        }
        
        Ok((parent_id, chunk_ids))
    }
    
    /// 获取 chunk_id 对应的向量
    pub async fn get_embedding(&self, chunk_id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        
        for entry in vectors.iter() {
            if entry.chunk_id == chunk_id {
                return Ok(Some(entry.embedding.clone()));
            }
        }
        
        Ok(None)
    }
    
    /// 获取向量数量
    pub async fn vector_count(&self) -> usize {
        let vectors = self.vectors.read().await;
        vectors.len()
    }
}

#[async_trait]
impl VectorStore for ChunkedVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "文档数量和向量数量不匹配".to_string()
            ));
        }
        
        let mut ids = Vec::new();
        
        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            let chunk_id = doc.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            
            self.document_store.add_document(doc).await?;
            self.add_chunk_vector(chunk_id.clone(), embedding).await?;
            
            ids.push(chunk_id);
        }
        
        Ok(ids)
    }
    
    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        
        let mut results: Vec<(String, f32)> = vectors
            .iter()
            .map(|entry| {
                let score = Self::cosine_similarity(query_embedding, &entry.embedding);
                (entry.chunk_id.clone(), score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_k_ids: Vec<(String, f32)> = results.into_iter().take(k).collect();
        
        let search_results: Vec<SearchResult> = future::join_all(
            top_k_ids.iter().map(|(chunk_id, score)| async move {
                let doc = self.document_store.get_chunk_document(chunk_id).await.ok().flatten();
                doc.map(|d| SearchResult { document: d, score: *score })
            })
        ).await.into_iter().flatten().collect();
        
        Ok(search_results)
    }
    
    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        self.document_store.get_chunk_document(id).await
    }
    
    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        
        for entry in vectors.iter() {
            if entry.chunk_id == id {
                return Ok(Some(entry.embedding.clone()));
            }
        }
        
        Ok(None)
    }
    
    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut vectors = self.vectors.write().await;
        vectors.retain(|entry| entry.chunk_id != id);
        
        self.document_store.delete_document(id).await?;
        
        Ok(())
    }
    
    async fn count(&self) -> usize {
        self.vector_count().await
    }
    
    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut vectors = self.vectors.write().await;
        vectors.clear();
        
        ChunkedDocumentStoreTrait::clear(&*self.document_store).await?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn mock_embedding(content: &str) -> Vec<f32> {
        let len = content.len() as f32;
        vec![len / 100.0, 0.0, 0.0]
    }
    
    #[tokio::test]
    async fn test_chunked_vector_store_basic() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);
        
        let chunk_id = "chunk_001".to_string();
        let embedding = vec![1.0, 0.0, 0.0];
        
        vector_store.add_chunk_vector(chunk_id.clone(), embedding.clone()).await.unwrap();
        
        assert_eq!(vector_store.vector_count().await, 1);
        
        let retrieved = vector_store.get_embedding(&chunk_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), embedding);
    }
    
    #[tokio::test]
    async fn test_similarity_search() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);
        
        vector_store.add_chunk_vector("chunk_001".to_string(), vec![1.0, 0.0, 0.0]).await.unwrap();
        vector_store.add_chunk_vector("chunk_002".to_string(), vec![0.0, 1.0, 0.0]).await.unwrap();
        
        doc_store.add_document(Document::new("Rust content").with_id("chunk_001")).await.unwrap();
        doc_store.add_document(Document::new("Python content").with_id("chunk_002")).await.unwrap();
        
        let query = vec![0.9, 0.1, 0.0];
        let results = vector_store.similarity_search(&query, 2).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
    }
    
    #[tokio::test]
    async fn test_add_parent_document() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 3);
        
        let doc = Document::new("这是一段很长的测试文本，用于验证分割功能。").with_id("parent_001");
        
        let (parent_id, chunk_ids) = vector_store
            .add_parent_document(doc, 20, mock_embedding)
            .await
            .unwrap();
        
        assert_eq!(parent_id, "parent_001");
        assert!(chunk_ids.len() > 1);
        assert_eq!(vector_store.vector_count().await, chunk_ids.len());
    }
    
    #[tokio::test]
    async fn test_dimension_validation() {
        let doc_store = Arc::new(ChunkedDocumentStore::new());
        let vector_store = ChunkedVectorStore::new(doc_store.clone(), 128);
        
        let result = vector_store
            .add_chunk_vector("chunk_001".to_string(), vec![1.0, 0.0])
            .await;
        
        assert!(result.is_err());
    }
}