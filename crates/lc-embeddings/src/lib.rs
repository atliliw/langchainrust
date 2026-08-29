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
// As of 1.0, `LocalEmbeddings` is not available without the `local-embeddings` feature (the
// old fallback alias was removed); users must explicitly choose — `BagOfWordsEmbeddings` or
// the ONNX version via the feature.
#[cfg(feature = "local-embeddings")]
pub use local::{LocalEmbeddings, LocalEmbeddingsBuilder};
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

    /// Configuration error (e.g. empty API key, unknown model dimension) — fails fast at construction.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Empty input
    #[error("Input is empty")]
    EmptyInput,

    /// Batch embedding misalignment: for N requested texts, the provider returned a vector
    /// count or index outside the expected range (a chunk missing entries / out of order).
    ///
    /// P0-1: reject silently misaligned data — never treat a missing vector as "dissimilar".
    #[error("Embedding batch mismatch: expected {expected} vectors, got position {actual}")]
    BatchMismatch {
        /// The expected number of vectors
        expected: usize,
        /// The position index where the misalignment occurred
        actual: usize,
    },

    /// Some text in a batch got no vector (the provider returned fewer embeddings than requested).
    ///
    /// P0-1: reject silently empty vectors — a missing vector is an explicit error, not a zero vector.
    #[error("Embedding batch contains an empty vector (provider returned fewer embeddings than requested)")]
    EmptyVectorInBatch,
}

/// Embedding model trait
///
/// Defines the interface for generating text embedding vectors.
///
/// # Normalization contract (P2-8)
///
/// All returned vectors are **L2-normalized** unit vectors (except zero vectors),
/// regardless of whether the provider normalizes internally. This guarantees downstream
/// cosine, dot-product, or L2-distance results do not drift across providers. HTTP
/// providers call [`l2_normalize`] uniformly before returning.
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
    /// # Semantic contract (P1-1)
    ///
    /// - Any empty or all-whitespace text (`trim().is_empty()`) → `Err(EmbeddingError::EmptyInput)`;
    /// - An empty slice `&[]` is treated as "no text to embed" → `Ok(vec![])` (nothing to do is not an error).
    ///
    /// The default implementation loops over [`Self::embed_query`] with a uniform emptiness
    /// check up front; providers that override it must follow the same contract and not diverge.
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
/// P2-8: providers normalize returned vectors differently (OpenAI normalizes, BOW
/// self-normalizes, remote providers such as Cohere may not); downstream results using
/// dot-product/L2 distance instead of cosine would drift across providers. This function
/// is the **single** normalization implementation; HTTP providers call it uniformly before
/// returning, ensuring `Embeddings` always produces unit-length vectors.
///
/// A zero vector stays zero (no NaN is produced).
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
