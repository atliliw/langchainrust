// src/embeddings/mod.rs
//! Embedding 模型实现

mod openai;
mod mock;
mod deepseek;
mod qwen;

pub use openai::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};
pub use mock::MockEmbeddings;
pub use deepseek::{DeepSeekEmbeddings, DeepSeekEmbeddingsConfig, DEEPSEEK_EMBED_MODEL};
pub use qwen::{QwenEmbeddings, QwenEmbeddingsConfig, QWEN_EMBED_MODEL};

use async_trait::async_trait;
use std::error::Error;

/// Embedding 错误类型
#[derive(Debug)]
pub enum EmbeddingError {
    /// HTTP 请求错误
    HttpError(String),
    
    /// API 错误
    ApiError(String),
    
    /// 解析错误
    ParseError(String),
    
    /// 空输入
    EmptyInput,
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::HttpError(msg) => write!(f, "HTTP 错误: {}", msg),
            EmbeddingError::ApiError(msg) => write!(f, "API 错误: {}", msg),
            EmbeddingError::ParseError(msg) => write!(f, "解析错误: {}", msg),
            EmbeddingError::EmptyInput => write!(f, "输入为空"),
        }
    }
}

impl Error for EmbeddingError {}

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

/// 计算两个向量的余弦相似度
///
/// # 参数
/// * `a` - 第一个向量
/// * `b` - 第二个向量
///
/// # 返回
/// 相似度值（-1 到 1，1 表示完全相似）
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
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