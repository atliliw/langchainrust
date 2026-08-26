#![warn(missing_docs)]
// lc-embeddings/src/lib.rs
//! Embedding model implementations for LangChainRust.
//!
//! Provides embedding generation via multiple backends:
//! - OpenAI (`text-embedding-ada-002`, `text-embedding-3-small/large`)
//! - DeepSeek
//! - Qwen (Alibaba Cloud / DashScope)
//! - Local: `BagOfWordsEmbeddings` (always available) and `LocalEmbeddings` (ONNX, feature-gated)
//! - `MockEmbeddings` for testing

mod cohere;
mod deepseek;
mod local;
mod mock;
mod openai;
pub mod openai_compat;
mod qwen;
mod retry;

#[cfg(test)]
mod test_support;

#[cfg(feature = "fastembed")]
mod fastembed_emb;

pub use cohere::{
    CohereEmbedInputType, CohereEmbeddings, CohereEmbeddingsConfig, COHERE_EMBED_BASE_URL,
    COHERE_EMBED_MODEL,
};
pub use deepseek::{DeepSeekEmbeddings, DeepSeekEmbeddingsConfig, DEEPSEEK_EMBED_MODEL};
pub use local::BagOfWordsEmbeddings;
// 1.0:无 `local-embeddings` feature 时 `LocalEmbeddings` 名字不可用(原降级别名
// 已移除),需显式选边——`BagOfWordsEmbeddings` 或开启 feature 用 ONNX 版。
#[cfg(feature = "local-embeddings")]
pub use local::LocalEmbeddings;
pub use mock::MockEmbeddings;
pub use openai::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};
pub use qwen::{QwenEmbeddings, QwenEmbeddingsConfig, QWEN_EMBED_MODEL};

#[cfg(feature = "fastembed")]
pub use fastembed_emb::FastEmbedEmbeddings;

use async_trait::async_trait;

/// Embedding error type
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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

    /// 配置错误（如 API key 为空、模型维度未知）——构造期 fail fast。
    #[error("Configuration error: {0}")]
    Config(String),

    /// Empty input
    #[error("Input is empty")]
    EmptyInput,

    /// 批量 embedding 数量错位：请求 N 条文本，服务端返回的向量数量或 index
    /// 超出了预期范围（某 chunk 少返回/乱序导致）。
    ///
    /// P0-1: 拒绝静默错数据——绝不把缺失向量当成"不相似"。
    #[error("Embedding batch mismatch: expected {expected} vectors, got position {actual}")]
    BatchMismatch {
        /// 期望返回的向量数量
        expected: usize,
        /// 出现错位的位置索引
        actual: usize,
    },

    /// 批量 embedding 中某条文本未取到向量（服务端返回量 < 请求量）。
    ///
    /// P0-1: 拒绝静默空向量——缺失即显式报错，而非留下零向量。
    #[error("Embedding batch contains an empty vector (provider returned fewer embeddings than requested)")]
    EmptyVectorInBatch,
}

/// Embedding model trait
///
/// Defines the interface for generating text embedding vectors.
///
/// # 归一化契约（P2-8）
///
/// 所有返回的向量都是 **L2 归一化**的单位向量（零向量除外），与 provider 内部
/// 是否已归一化无关。这保证下游无论用 cosine、点积还是 L2 距离，结果都不随
/// provider 漂移。HTTP provider 在返回前统一调用 [`l2_normalize`]。
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// Generate an embedding vector for a single text.
    ///
    /// # Arguments
    /// * `text` - Input text
    ///
    /// # Returns
    /// Embedding vector (typically 1536 dimensions or higher)
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embedding vectors for multiple documents.
    ///
    /// # Arguments
    /// * `texts` - List of input texts
    ///
    /// # Returns
    /// List of embedding vectors
    ///
    /// # 语义约定（P1-1）
    ///
    /// - 任一文本为空或全空白（`trim().is_empty()`）→ `Err(EmbeddingError::EmptyInput)`；
    /// - 空切片 `&[]` 视为"没有要嵌入的文本"→ `Ok(vec![])`（无事可做不算错误）。
    ///
    /// 默认实现循环调用 [`Self::embed_query`]，并前置统一判空；各 provider
    /// 覆写时须遵循同样的契约，不得再各自分裂。
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed_query(text).await?);
        }
        Ok(embeddings)
    }

    /// Get the embedding vector dimension.
    fn dimension(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// Compute cosine similarity between two vectors.
///
/// Re-exported from [`lc_core::math::cosine_similarity`].
pub use lc_core::math::cosine_similarity;

/// In-place L2 normalization: scale `vec` to unit length.
///
/// P2-8: 各 provider 返回向量的归一化口径不一(OpenAI 已归一化、BOW 自归一化、
/// Cohere 等远程 provider 可能不),下游一旦用点积/L2 距离而非 cosine,结果会因
/// provider 漂移。本函数是**唯一**的归一化实现,HTTP provider 在返回前统一调用,
/// 保证 `Embeddings` 产出的向量恒为单位长度。
///
/// 零向量保持零向量(不产生 NaN)。
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Mutex for synchronizing environment-variable mutations in tests.
///
/// Tests that set/remove env vars must acquire this lock to avoid data races
/// when running tests in parallel (the default for `cargo test`).
#[cfg(test)]
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 0.0001);

        // Opposite vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }
}
