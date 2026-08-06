// lc-embeddings/src/fastembed_emb.rs
//! FastEmbed local embeddings — ONNX Runtime with pre-downloaded models.
//!
//! FastEmbed (from Qdrant) provides fast, local embedding generation using
//! ONNX Runtime. No API calls needed — models run entirely on-device.
//!
//! Requires the `fastembed` feature to be enabled.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_embeddings::FastEmbedEmbeddings;
//! use fastembed::EmbeddingModel;
//!
//! let embedder = FastEmbedEmbeddings::with_model(EmbeddingModel::BGESmallENV15)?;
//! let vec = embedder.embed_query("hello world").await?;
//! ```

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::{EmbeddingError, Embeddings};

/// FastEmbed embedding provider.
///
/// Wraps the `fastembed::TextEmbedding` ONNX Runtime engine.
pub struct FastEmbedEmbeddings {
    model: TextEmbedding,
    model_name: String,
    dimension: usize,
}

impl std::fmt::Debug for FastEmbedEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedEmbeddings")
            .field("model", &self.model_name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl FastEmbedEmbeddings {
    /// Creates a new FastEmbedEmbeddings with the given init options.
    pub fn new(options: TextInitOptions) -> Result<Self, EmbeddingError> {
        let model_name = format!("{:?}", options.model_name);
        let model_type = options.model_name;
        let dimension = Self::infer_dimension(model_type);

        let model = TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::ApiError(format!("Failed to initialize FastEmbed model: {}", e))
        })?;

        Ok(Self {
            model,
            model_name,
            dimension,
        })
    }

    /// Creates with the default model (BAAI/bge-small-en-v1.5).
    pub fn default_model() -> Result<Self, EmbeddingError> {
        Self::new(
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(false),
        )
    }

    /// Creates with a specific model.
    pub fn with_model(model: EmbeddingModel) -> Result<Self, EmbeddingError> {
        Self::new(TextInitOptions::new(model).with_show_download_progress(false))
    }

    /// Infers embedding dimension from the model type.
    fn infer_dimension(model: EmbeddingModel) -> usize {
        match model {
            EmbeddingModel::BGESmallENV15 | EmbeddingModel::BGESmallENV15Q => 384,
            EmbeddingModel::BGEBaseENV15 | EmbeddingModel::BGEBaseENV15Q => 768,
            EmbeddingModel::BGELargeENV15 | EmbeddingModel::BGELargeENV15Q => 1024,
            EmbeddingModel::AllMiniLML6V2 | EmbeddingModel::AllMiniLML6V2Q => 384,
            EmbeddingModel::AllMiniLML12V2 | EmbeddingModel::AllMiniLML12V2Q => 384,
            EmbeddingModel::AllMpnetBaseV2 => 768,
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::NomicEmbedTextV15 | EmbeddingModel::NomicEmbedTextV15Q => 768,
            EmbeddingModel::MxbaiEmbedLargeV1 | EmbeddingModel::MxbaiEmbedLargeV1Q => 1024,
            EmbeddingModel::GTEBaseENV15 | EmbeddingModel::GTEBaseENV15Q => 768,
            EmbeddingModel::GTELargeENV15 | EmbeddingModel::GTELargeENV15Q => 1024,
            EmbeddingModel::MultilingualE5Small => 384,
            EmbeddingModel::MultilingualE5Base => 768,
            EmbeddingModel::MultilingualE5Large => 1024,
            EmbeddingModel::BGESmallZHV15 => 512,
            EmbeddingModel::BGELargeZHV15 => 1024,
            EmbeddingModel::BGEM3 => 1024,
            EmbeddingModel::ClipVitB32 => 512,
            EmbeddingModel::JinaEmbeddingsV2BaseEN => 768,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 768,
            EmbeddingModel::ParaphraseMLMiniLML12V2
            | EmbeddingModel::ParaphraseMLMiniLML12V2Q => 384,
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
            EmbeddingModel::ModernBertEmbedLarge => 1536,
            EmbeddingModel::EmbeddingGemma300M
            | EmbeddingModel::EmbeddingGemma300MQ
            | EmbeddingModel::EmbeddingGemma300MQ4 => 768,
            EmbeddingModel::SnowflakeArcticEmbedXS | EmbeddingModel::SnowflakeArcticEmbedXSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedS | EmbeddingModel::SnowflakeArcticEmbedSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedM | EmbeddingModel::SnowflakeArcticEmbedMQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedMLong
            | EmbeddingModel::SnowflakeArcticEmbedMLongQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedL | EmbeddingModel::SnowflakeArcticEmbedLQ => 1024,
            // Default fallback for unknown/future models
            _ => 384,
        }
    }
}

#[async_trait]
impl Embeddings for FastEmbedEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let text = text.to_string();
        let model_ref = &self.model;

        tokio::task::spawn_blocking(move || {
            let result = model_ref
                .embed(vec![text.as_str()], None)
                .map_err(|e| {
                    EmbeddingError::ApiError(format!("FastEmbed inference failed: {}", e))
                })?;

            result
                .first()
                .map(|v| v.to_vec())
                .ok_or_else(|| EmbeddingError::ApiError("No embedding returned".to_string()))
        })
        .await
        .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model_ref = &self.model;

        tokio::task::spawn_blocking(move || {
            let str_vec: Vec<&str> = text_vec.iter().map(|s| s.as_str()).collect();
            let result = model_ref
                .embed(str_vec, None)
                .map_err(|e| {
                    EmbeddingError::ApiError(format!(
                        "FastEmbed batch inference failed: {}",
                        e
                    ))
                })?;

            Ok(result.into_iter().map(|v| v.to_vec()).collect())
        })
        .await
        .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_dimension_bge_small() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::BGESmallENV15);
        assert_eq!(dim, 384);
    }

    #[test]
    fn test_infer_dimension_bge_base() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::BGEBaseENV15);
        assert_eq!(dim, 768);
    }

    #[test]
    fn test_infer_dimension_bge_large() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::BGELargeENV15);
        assert_eq!(dim, 1024);
    }

    #[test]
    fn test_infer_dimension_mini_lm() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::AllMiniLML6V2);
        assert_eq!(dim, 384);
    }

    #[test]
    fn test_infer_dimension_nomic() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(dim, 768);
    }

    #[test]
    fn test_infer_dimension_mxbai() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::MxbaiEmbedLargeV1);
        assert_eq!(dim, 1024);
    }

    #[test]
    fn test_infer_dimension_gte_base() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::GTEBaseENV15);
        assert_eq!(dim, 768);
    }

    #[test]
    fn test_infer_dimension_multilingual_e5() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::MultilingualE5Base);
        assert_eq!(dim, 768);
    }

    #[test]
    fn test_infer_dimension_snowflake_xs() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::SnowflakeArcticEmbedXS);
        assert_eq!(dim, 384);
    }

    #[test]
    fn test_infer_dimension_modern_bert() {
        let dim = FastEmbedEmbeddings::infer_dimension(EmbeddingModel::ModernBertEmbedLarge);
        assert_eq!(dim, 1536);
    }
}
