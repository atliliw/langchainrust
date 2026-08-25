// lc-embeddings/src/local/mod.rs
//! Local embedding implementations
//!
//! Contains two implementations:
//! - `BagOfWordsEmbeddings`: Lightweight word-frequency hash embedding (pure Rust, no external deps), always available
//! - `LocalEmbeddings`: ONNX Runtime-based neural network embedding (requires `local-embeddings` feature)
//!
//! `BagOfWordsEmbeddings` is suitable for offline, privacy, zero-cost coarse-grained retrieval;
//! `LocalEmbeddings` is suitable for high-quality semantic embedding scenarios (e.g., BGE/E5 models).

use async_trait::async_trait;

#[cfg(feature = "local-embeddings")]
use std::path::Path;

use crate::{EmbeddingError, Embeddings};

// ---------------------------------------------------------------------------
// BagOfWordsEmbeddings — word-frequency hash + L2 normalization (always available)
// ---------------------------------------------------------------------------

/// Lightweight local embedding (word-frequency hash + L2 normalization)
///
/// Based on word frequency + hashing, no API calls, suitable for offline, privacy, zero-cost coarse-grained retrieval.
///
/// Note: This is a lightweight implementation (bag-of-words hash) with limited semantic quality;
/// for high-quality neural network embeddings (BGE/E5 via `ort`), enable the `local-embeddings` feature
/// and use [`LocalEmbeddings`].
pub struct BagOfWordsEmbeddings {
    dim: usize,
}

impl BagOfWordsEmbeddings {
    /// Create local embedding with specified dimension
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    /// Default dimension 256
    pub fn default_dim() -> Self {
        Self::new(256)
    }

    /// Tokenize: English by non-alphanumeric split (lowercased), Chinese/non-ASCII by single character
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                if c.is_ascii() {
                    current.push(c.to_ascii_lowercase());
                } else {
                    // Non-ASCII (Chinese etc.) single character as token
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

    /// FNV-1a hash
    fn hash(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Compute embedding vector (word-frequency hash + L2 normalization)
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in Self::tokenize(text) {
            let idx = (Self::hash(&token) as usize) % self.dim;
            v[idx] += 1.0;
        }
        // L2 normalization
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for BagOfWordsEmbeddings {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[async_trait]
impl Embeddings for BagOfWordsEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        Ok(self.embed(text))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "local-bow"
    }
}

// ---------------------------------------------------------------------------
// LocalEmbeddings — ONNX Runtime neural network embedding (requires local-embeddings feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "local-embeddings")]
mod nn;

// When local-embeddings feature is enabled, re-export LocalEmbeddings and its builder
#[cfg(feature = "local-embeddings")]
pub use nn::{LocalEmbeddings, LocalEmbeddingsBuilder};

// ---------------------------------------------------------------------------
// Backward compatibility: LocalEmbeddings without feature points to BagOfWordsEmbeddings
// ---------------------------------------------------------------------------

/// Without the `local-embeddings` feature, `LocalEmbeddings` is a type alias for `BagOfWordsEmbeddings`,
/// maintaining backward compatibility.
///
/// With the `local-embeddings` feature enabled, `LocalEmbeddings` becomes the ONNX Runtime-based neural network implementation.
///
/// P2-1: 消除静默降级。无 feature 时 `LocalEmbeddings` 静默退化为词袋哈希嵌入,
/// 用户以为在用语义向量、实际是词频——"好像能用,但不对"。这里加
/// `#[deprecated]` 让降级在编译期可见:使用者需显式改用 `BagOfWordsEmbeddings`,
/// 或开启 `local-embeddings` feature 使用真正的 ONNX 神经嵌入。
#[cfg(not(feature = "local-embeddings"))]
#[deprecated(
    note = "LocalEmbeddings without the `local-embeddings` feature degrades to \
            BagOfWordsEmbeddings (bag-of-words hash), not semantic neural embedding. \
            Enable the `local-embeddings` feature, or use BagOfWordsEmbeddings explicitly."
)]
pub type LocalEmbeddings = BagOfWordsEmbeddings;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
