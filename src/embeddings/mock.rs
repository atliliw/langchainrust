// src/embeddings/mock.rs
//! Mock embeddings implementation for testing.
//!
//! Generates deterministic pseudo-random embeddings based on text hash.

use super::{Embeddings, EmbeddingError};
use async_trait::async_trait;

/// Mock embeddings for testing purposes.
///
/// Generates fixed-pattern embedding vectors based on text hash,
/// useful for unit tests without real API calls.
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    /// Creates a new MockEmbeddings with specified dimension.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for MockEmbeddings {
    fn default() -> Self {
        Self::new(1536)
    }
}

#[async_trait]
impl Embeddings for MockEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        
        // 生成基于文本哈希的伪随机向量
        let hash = Self::hash_text(text);
        let mut embedding = Vec::with_capacity(self.dimension);
        
        for i in 0..self.dimension {
            // 使用哈希和索引生成伪随机值
            let value = ((hash.wrapping_add(i as u64)) % 1000) as f32 / 1000.0 - 0.5;
            embedding.push(value);
        }
        
        // 归一化
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        
        Ok(embedding)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn model_name(&self) -> &str {
        "mock-embeddings"
    }
}

impl MockEmbeddings {
    /// 简单的文本哈希函数
    fn hash_text(text: &str) -> u64 {
        let mut hash: u64 = 0;
        for (i, c) in text.chars().enumerate() {
            hash = hash.wrapping_add((c as u64).wrapping_mul((i + 1) as u64));
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mock_embedding() {
        let embeddings = MockEmbeddings::new(128);
        
        let result = embeddings.embed_query("Hello, world!").await.unwrap();
        assert_eq!(result.len(), 128);
        
        // 相同文本应该产生相同向量
        let result2 = embeddings.embed_query("Hello, world!").await.unwrap();
        assert_eq!(result, result2);
        
        // 不同文本应该产生不同向量
        let result3 = embeddings.embed_query("Different text").await.unwrap();
        assert_ne!(result, result3);
    }
    
    #[tokio::test]
    async fn test_mock_embedding_empty() {
        let embeddings = MockEmbeddings::new(128);
        
        let result = embeddings.embed_query("").await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_mock_embedding_normalized() {
        let embeddings = MockEmbeddings::new(128);
        
        let result = embeddings.embed_query("Test normalization").await.unwrap();
        
        // 向量应该是归一化的（模长约为 1）
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.0001);
    }
}