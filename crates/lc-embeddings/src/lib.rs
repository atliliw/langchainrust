// lc-embeddings/src/lib.rs
//! Embedding model implementations for LangChainRust.
//!
//! Provides embedding generation via multiple backends:
//! - OpenAI (`text-embedding-ada-002`, `text-embedding-3-small/large`)
//! - DeepSeek
//! - Qwen (Alibaba Cloud / DashScope)
//! - Local: `BagOfWordsEmbeddings` (always available) and `LocalEmbeddings` (ONNX, feature-gated)
//! - `MockEmbeddings` for testing

mod deepseek;
mod cohere;
mod local;
mod mock;
mod openai;
mod qwen;

#[cfg(feature = "fastembed")]
mod fastembed_emb;

pub use cohere::{
    CohereEmbeddings, CohereEmbeddingsConfig, CohereEmbedInputType, COHERE_EMBED_BASE_URL,
    COHERE_EMBED_MODEL,
};
pub use deepseek::{DeepSeekEmbeddings, DeepSeekEmbeddingsConfig, DEEPSEEK_EMBED_MODEL};
pub use local::{BagOfWordsEmbeddings, LocalEmbeddings};
pub use mock::MockEmbeddings;
pub use openai::{OpenAIEmbeddings, OpenAIEmbeddingsConfig};
pub use qwen::{QwenEmbeddings, QwenEmbeddingsConfig, QWEN_EMBED_MODEL};

#[cfg(feature = "fastembed")]
pub use fastembed_emb::FastEmbedEmbeddings;

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

/// Embedding model trait
///
/// Defines the interface for generating text embedding vectors.
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
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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
