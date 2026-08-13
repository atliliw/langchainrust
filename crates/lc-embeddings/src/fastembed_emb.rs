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
        // P1-7: model_name 存用户友好名称（如 "BGE-small-en-v1.5"），
        // 而非内部 Debug 标识（如 "BGESmallENV15"）。
        let model_name = Self::friendly_model_name(options.model_name.clone()).to_string();
        // P1-2: 维度未知即报错，不得回落默认 384 撒谎。
        let dimension = Self::infer_dimension(options.model_name.clone())?;

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
            TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )
    }

    /// Creates with a specific model.
    pub fn with_model(model: EmbeddingModel) -> Result<Self, EmbeddingError> {
        Self::new(TextInitOptions::new(model).with_show_download_progress(false))
    }

    /// Infers embedding dimension from the model type.
    ///
    /// P1-2: 维度表穷举已知模型；fastembed 新增未知模型会在编译期以
    /// "non-exhaustive match" 显式暴露，而不是运行时静默回落默认值撒谎。
    fn infer_dimension(model: EmbeddingModel) -> Result<usize, EmbeddingError> {
        let dim = match model {
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
            EmbeddingModel::ParaphraseMLMiniLML12V2 | EmbeddingModel::ParaphraseMLMiniLML12V2Q => {
                384
            }
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
            EmbeddingModel::ModernBertEmbedLarge => 1024,
            EmbeddingModel::EmbeddingGemma300M
            | EmbeddingModel::EmbeddingGemma300MQ
            | EmbeddingModel::EmbeddingGemma300MQ4 => 768,
            EmbeddingModel::SnowflakeArcticEmbedXS | EmbeddingModel::SnowflakeArcticEmbedXSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedS | EmbeddingModel::SnowflakeArcticEmbedSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedM | EmbeddingModel::SnowflakeArcticEmbedMQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedMLong
            | EmbeddingModel::SnowflakeArcticEmbedMLongQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedL | EmbeddingModel::SnowflakeArcticEmbedLQ => 1024,
        };
        Ok(dim)
    }

    /// Returns a user-friendly model name (P1-7), e.g. `"BGE-small-en-v1.5"`
    /// instead of the internal Debug identifier `"BGESmallENV15"`.
    fn friendly_model_name(model: EmbeddingModel) -> &'static str {
        match model {
            EmbeddingModel::AllMiniLML6V2 => "all-MiniLM-L6-v2",
            EmbeddingModel::AllMiniLML6V2Q => "all-MiniLM-L6-v2 (quantized)",
            EmbeddingModel::AllMiniLML12V2 => "all-MiniLM-L12-v2",
            EmbeddingModel::AllMiniLML12V2Q => "all-MiniLM-L12-v2 (quantized)",
            EmbeddingModel::AllMpnetBaseV2 => "all-mpnet-base-v2",
            EmbeddingModel::BGEBaseENV15 => "BGE-base-en-v1.5",
            EmbeddingModel::BGEBaseENV15Q => "BGE-base-en-v1.5 (quantized)",
            EmbeddingModel::BGELargeENV15 => "BGE-large-en-v1.5",
            EmbeddingModel::BGELargeENV15Q => "BGE-large-en-v1.5 (quantized)",
            EmbeddingModel::BGESmallENV15 => "BGE-small-en-v1.5",
            EmbeddingModel::BGESmallENV15Q => "BGE-small-en-v1.5 (quantized)",
            EmbeddingModel::NomicEmbedTextV1 => "nomic-embed-text-v1",
            EmbeddingModel::NomicEmbedTextV15 => "nomic-embed-text-v1.5",
            EmbeddingModel::NomicEmbedTextV15Q => "nomic-embed-text-v1.5 (quantized)",
            EmbeddingModel::ParaphraseMLMiniLML12V2 => "paraphrase-multilingual-MiniLM-L12-v2",
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => {
                "paraphrase-multilingual-MiniLM-L12-v2 (quantized)"
            }
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => "paraphrase-multilingual-mpnet-base-v2",
            EmbeddingModel::BGESmallZHV15 => "BGE-small-zh-v1.5",
            EmbeddingModel::BGELargeZHV15 => "BGE-large-zh-v1.5",
            EmbeddingModel::BGEM3 => "BGE-m3",
            EmbeddingModel::ModernBertEmbedLarge => "modernbert-embed-large",
            EmbeddingModel::MultilingualE5Small => "multilingual-e5-small",
            EmbeddingModel::MultilingualE5Base => "multilingual-e5-base",
            EmbeddingModel::MultilingualE5Large => "multilingual-e5-large",
            EmbeddingModel::MxbaiEmbedLargeV1 => "mxbai-embed-large-v1",
            EmbeddingModel::MxbaiEmbedLargeV1Q => "mxbai-embed-large-v1 (quantized)",
            EmbeddingModel::GTEBaseENV15 => "gte-base-en-v1.5",
            EmbeddingModel::GTEBaseENV15Q => "gte-base-en-v1.5 (quantized)",
            EmbeddingModel::GTELargeENV15 => "gte-large-en-v1.5",
            EmbeddingModel::GTELargeENV15Q => "gte-large-en-v1.5 (quantized)",
            EmbeddingModel::ClipVitB32 => "clip-ViT-B-32-text",
            EmbeddingModel::JinaEmbeddingsV2BaseCode => "jina-embeddings-v2-base-code",
            EmbeddingModel::JinaEmbeddingsV2BaseEN => "jina-embeddings-v2-base-en",
            EmbeddingModel::EmbeddingGemma300M => "embeddinggemma-300m",
            EmbeddingModel::EmbeddingGemma300MQ4 => "embeddinggemma-300m (q4)",
            EmbeddingModel::EmbeddingGemma300MQ => "embeddinggemma-300m (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedXS => "snowflake-arctic-embed-xs",
            EmbeddingModel::SnowflakeArcticEmbedXSQ => "snowflake-arctic-embed-xs (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedS => "snowflake-arctic-embed-s",
            EmbeddingModel::SnowflakeArcticEmbedSQ => "snowflake-arctic-embed-s (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedM => "snowflake-arctic-embed-m",
            EmbeddingModel::SnowflakeArcticEmbedMQ => "snowflake-arctic-embed-m (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedMLong => "snowflake-arctic-embed-m-long",
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => {
                "snowflake-arctic-embed-m-long (quantized)"
            }
            EmbeddingModel::SnowflakeArcticEmbedL => "snowflake-arctic-embed-l",
            EmbeddingModel::SnowflakeArcticEmbedLQ => "snowflake-arctic-embed-l (quantized)",
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
            let result = model_ref.embed(vec![text.as_str()], None).map_err(|e| {
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
        // P1-1: 空切片不是错误（无事可做），含空/全空白文本才报错——契约统一。
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let model_ref = &self.model;

        tokio::task::spawn_blocking(move || {
            let str_vec: Vec<&str> = text_vec.iter().map(|s| s.as_str()).collect();
            let result = model_ref.embed(str_vec, None).map_err(|e| {
                EmbeddingError::ApiError(format!("FastEmbed batch inference failed: {}", e))
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

    fn dim(model: EmbeddingModel) -> usize {
        FastEmbedEmbeddings::infer_dimension(model).unwrap()
    }

    #[test]
    fn test_infer_dimension_bge_small() {
        assert_eq!(dim(EmbeddingModel::BGESmallENV15), 384);
    }

    #[test]
    fn test_infer_dimension_bge_base() {
        assert_eq!(dim(EmbeddingModel::BGEBaseENV15), 768);
    }

    #[test]
    fn test_infer_dimension_bge_large() {
        assert_eq!(dim(EmbeddingModel::BGELargeENV15), 1024);
    }

    #[test]
    fn test_infer_dimension_mini_lm() {
        assert_eq!(dim(EmbeddingModel::AllMiniLML6V2), 384);
    }

    #[test]
    fn test_infer_dimension_nomic() {
        assert_eq!(dim(EmbeddingModel::NomicEmbedTextV15), 768);
    }

    #[test]
    fn test_infer_dimension_mxbai() {
        assert_eq!(dim(EmbeddingModel::MxbaiEmbedLargeV1), 1024);
    }

    #[test]
    fn test_infer_dimension_gte_base() {
        assert_eq!(dim(EmbeddingModel::GTEBaseENV15), 768);
    }

    #[test]
    fn test_infer_dimension_multilingual_e5() {
        assert_eq!(dim(EmbeddingModel::MultilingualE5Base), 768);
    }

    #[test]
    fn test_infer_dimension_snowflake_xs() {
        assert_eq!(dim(EmbeddingModel::SnowflakeArcticEmbedXS), 384);
    }

    #[test]
    fn test_infer_dimension_modern_bert() {
        // 权威维度来自 fastembed 5.17.4 ModelInfo：modernbert-embed-large = 1024。
        assert_eq!(dim(EmbeddingModel::ModernBertEmbedLarge), 1024);
    }

    /// P1-7: model_name() 返回用户友好名称而非内部 Debug 标识。
    #[test]
    fn test_friendly_model_name() {
        assert_eq!(
            FastEmbedEmbeddings::friendly_model_name(EmbeddingModel::BGESmallENV15),
            "BGE-small-en-v1.5"
        );
        assert_eq!(
            FastEmbedEmbeddings::friendly_model_name(EmbeddingModel::SnowflakeArcticEmbedMLong),
            "snowflake-arctic-embed-m-long"
        );
    }
}
