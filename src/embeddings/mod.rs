// src/embeddings/mod.rs
//! Embedding 模型实现

mod openai;
mod mock;
mod deepseek;
mod qwen;
mod local;

pub use openai::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};
pub use mock::MockEmbeddings;
pub use deepseek::{DeepSeekEmbeddings, DeepSeekEmbeddingsConfig, DEEPSEEK_EMBED_MODEL};
pub use qwen::{QwenEmbeddings, QwenEmbeddingsConfig, QWEN_EMBED_MODEL};
pub use local::{BagOfWordsEmbeddings, LocalEmbeddings};

use async_trait::async_trait;

/// Embedding error type
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// HTTP request error
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// API error
    #[error("API error: {0}")]
    ApiError(String),

    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Empty input
    #[error("Input is empty")]
    EmptyInput,
}

/// Embedding 模型 trait
///
/// 定义文本嵌入向量的生成接口。
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// 为单个文本生成嵌入向量
    ///
    /// # 参数
    /// * `text` - 输入文本
    ///
    /// # 返回
    /// 嵌入向量（通常是 1536 维或更高）
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    
    /// 为多个文档生成嵌入向量
    ///
    /// # 参数
    /// * `texts` - 输入文本列表
    ///
    /// # 返回
    /// 嵌入向量列表
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed_query(text).await?);
        }
        Ok(embeddings)
    }
    
    /// 获取嵌入向量维度
    fn dimension(&self) -> usize;
    
    /// 获取模型名称
    fn model_name(&self) -> &str;
}

/// Compute cosine similarity between two vectors.
///
/// Re-exported from [`crate::core::math::cosine_similarity`].
pub use crate::core::math::cosine_similarity;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        // 相同向量
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
        
        // 正交向量
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.0001);
        
        // 相反向量
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 0.0001);
    }
    
    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}