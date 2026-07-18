// src/embeddings/local.rs
//! 轻量本地嵌入(纯 Rust,无外部依赖)
//!
//! 基于词频 + 哈希的本地嵌入,不调用任何 API,适合离线、隐私、零成本场景的粗粒度检索。
//!
//! 注:这是轻量实现(词袋 hash),语义质量有限;若需高质量神经网络嵌入
//! (BGE/E5 via `ort`),见计划书 #3 的 feature gate 版本(待补)。

use async_trait::async_trait;

use super::{Embeddings, EmbeddingError};

/// 轻量本地嵌入(词频 hash + L2 归一化)
pub struct LocalEmbeddings {
    dim: usize,
}

impl LocalEmbeddings {
    /// 创建指定维度的本地嵌入
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(1),
        }
    }

    /// 默认维度 256
    pub fn default_dim() -> Self {
        Self::new(256)
    }

    /// 分词:英文按非字母数字切分(小写化),中文/非 ASCII 按单字
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                if c.is_ascii() {
                    current.push(c.to_ascii_lowercase());
                } else {
                    // 非ASCII(中文等)单字成 token
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                    tokens.push(c.to_string());
                }
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// FNV-1a 哈希
    fn hash(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// 计算嵌入向量(词频 hash + L2 归一化)
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in Self::tokenize(text) {
            let idx = (Self::hash(&token) as usize) % self.dim;
            v[idx] += 1.0;
        }
        // L2 归一化
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for LocalEmbeddings {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[async_trait]
impl Embeddings for LocalEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.embed(text))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "local-bow"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::cosine_similarity;

    #[tokio::test]
    async fn test_dimension() {
        let e = LocalEmbeddings::new(128);
        let v = e.embed_query("hello world").await.unwrap();
        assert_eq!(v.len(), 128);
        assert_eq!(e.dimension(), 128);
    }

    #[tokio::test]
    async fn test_same_text_same_vector() {
        let e = LocalEmbeddings::new(64);
        let a = e.embed_query("rust programming").await.unwrap();
        let b = e.embed_query("rust programming").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn test_different_text_different_vector() {
        let e = LocalEmbeddings::new(64);
        let a = e.embed_query("rust programming").await.unwrap();
        let b = e.embed_query("cooking recipe pasta").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn test_shared_words_more_similar() {
        // 共享词的文本应比无共享词的更相似
        let e = LocalEmbeddings::new(256);
        let base = e.embed_query("rust programming language").await.unwrap();
        let similar = e.embed_query("rust programming tutorial").await.unwrap();
        let different = e.embed_query("cooking pasta recipe").await.unwrap();

        let sim_similar = cosine_similarity(&base, &similar);
        let sim_different = cosine_similarity(&base, &different);
        assert!(
            sim_similar > sim_different,
            "共享词应更相似: {} vs {}",
            sim_similar,
            sim_different
        );
    }

    #[tokio::test]
    async fn test_normalized() {
        let e = LocalEmbeddings::new(64);
        let v = e.embed_query("some text here").await.unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // 非空文本应归一化为单位向量
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[tokio::test]
    async fn test_empty_text_zero_vector() {
        let e = LocalEmbeddings::new(64);
        let v = e.embed_query("").await.unwrap();
        // 空文本 -> 零向量(不报错)
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[tokio::test]
    async fn test_chinese_tokenize() {
        let e = LocalEmbeddings::new(128);
        let a = e.embed_query("机器学习").await.unwrap();
        let b = e.embed_query("机器学习").await.unwrap();
        assert_eq!(a, b);
        // 中文单字应被分词
        let c = e.embed_query("深度学习").await.unwrap();
        // 共享"机器学习"的"学习"两字 vs "深度学习"的"学习"两字 -> 部分共享
        let sim = cosine_similarity(&a, &c);
        assert!(sim > 0.0, "共享\"学习\"应有正相似度: {}", sim);
    }

    #[test]
    fn test_tokenize_english() {
        let t = LocalEmbeddings::tokenize("Hello, World! 123");
        assert!(t.contains(&"hello".to_string()));
        assert!(t.contains(&"world".to_string()));
        assert!(t.contains(&"123".to_string()));
    }

    #[test]
    fn test_tokenize_chinese() {
        let t = LocalEmbeddings::tokenize("机器学习");
        // 中文按单字
        assert!(t.contains(&"机".to_string()));
        assert!(t.contains(&"学".to_string()));
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn test_model_name() {
        let e = LocalEmbeddings::default_dim();
        assert_eq!(e.model_name(), "local-bow");
    }
}
