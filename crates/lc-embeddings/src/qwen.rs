// lc-embeddings/src/qwen.rs
//! Qwen (Alibaba Cloud) embeddings implementation.
//!
//! Qwen (DashScope compatible mode) speaks the OpenAI-compatible `/embeddings` protocol and
//! reuses the [`crate::openai_compat`] shared base class (P1-5); this file only configures
//! the spec (URL / model / dimension / batch size).

use crate::openai_compat::{CompatConfigAccess, CompatSpec, OpenAICompatEmbeddings};
use crate::EmbeddingError;

/// Default base URL for the Qwen (DashScope) API.
pub const QWEN_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// Default embedding model for Qwen.
pub const QWEN_EMBED_MODEL: &str = "text-embedding-v1";

/// Qwen3-Embedding series models (0.21.0 S5.1).
///
/// All three support **matryoshka** output dimensions (32-4096) via the
/// `dimensions` request parameter, and token-level output (late chunking
/// prerequisite). Defaults: 0.6B=1024, 4B=2560, 8B=4096.
pub const QWEN3_EMBEDDING_0_6B: &str = "qwen3-embedding-0.6b";
/// Qwen3-Embedding 4B model.
pub const QWEN3_EMBEDDING_4B: &str = "qwen3-embedding-4b";
/// Qwen3-Embedding 8B model.
pub const QWEN3_EMBEDDING_8B: &str = "qwen3-embedding-8b";

/// Whether `model` is a Qwen3-Embedding series model (supports `dimensions`).
pub fn is_qwen3_embedding(model: &str) -> bool {
    model.starts_with("qwen3-embedding")
}

/// Configuration for Qwen embeddings API.
#[derive(Debug, Clone)]
pub struct QwenEmbeddingsConfig {
    /// Qwen API key.
    pub api_key: String,
    /// Base URL for the Qwen (DashScope) embeddings API.
    pub base_url: String,
    /// Embedding model name.
    pub model: String,
    /// Optional matryoshka output dimension (0.21.0 S5.1): 32-4096. Only valid
    /// on Qwen3-Embedding models; `None` keeps the model default.
    pub dimensions: Option<usize>,
}

impl Default for QwenEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("QWEN_API_KEY").unwrap_or_default(),
            base_url: QWEN_BASE_URL.to_string(),
            model: QWEN_EMBED_MODEL.to_string(),
            dimensions: None,
        }
    }
}

impl QwenEmbeddingsConfig {
    /// Creates a new QwenEmbeddingsConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a QwenEmbeddingsConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `QWEN_API_KEY`: API key (required)
    /// - `QWEN_BASE_URL`: API endpoint (optional)
    /// - `QWEN_EMBED_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("QWEN_API_KEY").map_err(|_| {
            EmbeddingError::Config("QWEN_API_KEY environment variable not set".to_string())
        })?;
        let base_url = std::env::var("QWEN_BASE_URL").unwrap_or_else(|_| QWEN_BASE_URL.to_string());
        let model =
            std::env::var("QWEN_EMBED_MODEL").unwrap_or_else(|_| QWEN_EMBED_MODEL.to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
            dimensions: None,
        })
    }

    /// Sets the embedding model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the matryoshka output dimension (32-4096, 0.21.0 S5.1).
    ///
    /// Only valid on Qwen3-Embedding models — construction fails otherwise
    /// (P1-2: never send a `dimensions` the model does not support).
    pub fn with_dimensions(mut self, dimensions: usize) -> Result<Self, EmbeddingError> {
        if !(32..=4096).contains(&dimensions) {
            return Err(EmbeddingError::Config(format!(
                "matryoshka dimensions must be within 32..=4096, got {dimensions}"
            )));
        }
        if !is_qwen3_embedding(&self.model) {
            return Err(EmbeddingError::Config(format!(
                "model '{}' does not support the `dimensions` parameter (Qwen3-Embedding only)",
                self.model
            )));
        }
        self.dimensions = Some(dimensions);
        Ok(self)
    }
}

impl CompatConfigAccess for QwenEmbeddingsConfig {
    fn api_key(&self) -> &str {
        &self.api_key
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }
}

impl CompatSpec for QwenEmbeddingsConfig {
    fn api_key_env() -> &'static str {
        "QWEN_API_KEY"
    }
    fn batch_size() -> usize {
        64
    }
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        match model {
            QWEN_EMBED_MODEL => Ok(1536),
            QWEN3_EMBEDDING_0_6B => Ok(1024),
            QWEN3_EMBEDDING_4B => Ok(2560),
            QWEN3_EMBEDDING_8B => Ok(4096),
            _ => Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for Qwen model '{model}' (supported: '{QWEN_EMBED_MODEL}', '{QWEN3_EMBEDDING_0_6B}', '{QWEN3_EMBEDDING_4B}', '{QWEN3_EMBEDDING_8B}')"
            ))),
        }
    }
    fn validate(config: &Self) -> Result<(), EmbeddingError> {
        // P1-2: a `dimensions` override on a model that does not support the
        // parameter must fail at construction, never at request time.
        if let Some(d) = config.dimensions {
            if !is_qwen3_embedding(&config.model) {
                return Err(EmbeddingError::Config(format!(
                    "model '{}' does not support the `dimensions` parameter (Qwen3-Embedding only)",
                    config.model
                )));
            }
            if !(32..=4096).contains(&d) {
                return Err(EmbeddingError::Config(format!(
                    "matryoshka dimensions must be within 32..=4096, got {d}"
                )));
            }
        }
        Ok(())
    }
    fn from_env_result() -> Result<Self, EmbeddingError> {
        Self::from_env_result()
    }
}

/// Qwen embeddings client for generating vector embeddings.
///
/// Reuses the OpenAI-compatible-protocol shared base class (P1-5): fails fast at construction
/// validating a non-empty API key and known model dimension (P1-2/P1-3), batch alignment errors
/// explicitly (P0-1), and error bodies are not swallowed (P1-4).
pub type QwenEmbeddings = OpenAICompatEmbeddings<QwenEmbeddingsConfig>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::spawn_embeddings_stub;
    use crate::Embeddings;
    use std::env;
    use std::sync::Arc;

    /// P0-1: provider returns fewer entries → explicit `EmptyVectorInBatch`, not a silent empty vector.
    #[tokio::test]
    async fn test_embed_documents_truncated_errors() {
        let base_url = spawn_embeddings_stub(Arc::new(|n| n.saturating_sub(1))).await;
        let config = QwenEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: QWEN_EMBED_MODEL.into(),
            dimensions: None,
        };
        let embeddings = QwenEmbeddings::new(config).unwrap();

        let result = embeddings.embed_documents(&["a", "b"]).await;
        assert!(
            matches!(result, Err(EmbeddingError::EmptyVectorInBatch)),
            "truncated response should report EmptyVectorInBatch, got: {:?}",
            result
        );
    }

    /// P1-3: an empty API key → `Config` error at construction (fail fast), not a delayed 401.
    #[test]
    fn test_new_rejects_empty_api_key() {
        let config = QwenEmbeddingsConfig {
            api_key: String::new(),
            base_url: QWEN_BASE_URL.into(),
            model: QWEN_EMBED_MODEL.into(),
            dimensions: None,
        };
        let err = QwenEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// P1-2: unknown model → construction-time error, never lying with a default 1536.
    #[test]
    fn test_new_rejects_unknown_model() {
        let config = QwenEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url: QWEN_BASE_URL.into(),
            model: "some-unknown-model".into(),
            dimensions: None,
        };
        let err = QwenEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// 0.21.0 S5.1: Qwen3-Embedding series dimension map (0.6B=1024, 4B=2560, 8B=4096).
    #[test]
    fn test_qwen3_dimension_map() {
        assert_eq!(
            QwenEmbeddingsConfig::dimension_for(QWEN3_EMBEDDING_0_6B).unwrap(),
            1024
        );
        assert_eq!(
            QwenEmbeddingsConfig::dimension_for(QWEN3_EMBEDDING_4B).unwrap(),
            2560
        );
        assert_eq!(
            QwenEmbeddingsConfig::dimension_for(QWEN3_EMBEDDING_8B).unwrap(),
            4096
        );
        assert_eq!(
            QwenEmbeddingsConfig::dimension_for(QWEN_EMBED_MODEL).unwrap(),
            1536
        );
    }

    /// 0.21.0 S5.1: Qwen3 construction with a matryoshka `dimensions` override.
    #[test]
    fn test_qwen3_with_dimensions() {
        let config = QwenEmbeddingsConfig::new("test-key")
            .with_model(QWEN3_EMBEDDING_0_6B)
            .with_dimensions(512)
            .unwrap();
        assert_eq!(config.dimensions, Some(512));
        let embeddings = QwenEmbeddings::new(config).unwrap();
        assert_eq!(embeddings.dimension(), 512, "configured dimension wins");
    }

    /// 0.21.0 S5.1: `dimensions` outside 32..=4096 → Config error.
    #[test]
    fn test_qwen3_with_dimensions_rejects_out_of_range() {
        let err = QwenEmbeddingsConfig::new("test-key")
            .with_model(QWEN3_EMBEDDING_0_6B)
            .with_dimensions(31)
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
        let err = QwenEmbeddingsConfig::new("test-key")
            .with_model(QWEN3_EMBEDDING_0_6B)
            .with_dimensions(4097)
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// 0.21.0 S5.1: `dimensions` on a non-Qwen3 model → Config error at
    /// construction (P1-2: never send an unsupported parameter).
    #[test]
    fn test_dimensions_rejected_on_non_qwen3_model() {
        let err = QwenEmbeddingsConfig::new("test-key")
            .with_model(QWEN_EMBED_MODEL)
            .with_dimensions(512)
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
        // And the CompatSpec-level guard (bypassing the builder) also rejects.
        let config = QwenEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url: QWEN_BASE_URL.into(),
            model: QWEN_EMBED_MODEL.into(),
            dimensions: Some(512),
        };
        let err = QwenEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// 0.21.0 S5.1: no `dimensions` → construction on Qwen3 uses the model default.
    #[test]
    fn test_qwen3_default_dimension() {
        let embeddings = QwenEmbeddings::new(
            QwenEmbeddingsConfig::new("test-key").with_model(QWEN3_EMBEDDING_8B),
        )
        .unwrap();
        assert_eq!(embeddings.dimension(), 4096);
    }

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = env::var(key).ok();
        env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    #[test]
    fn test_from_env_result_ok_when_key_set() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("QWEN_API_KEY", "test-key-123");
        let result = QwenEmbeddingsConfig::from_env_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().api_key, "test-key-123");
        restore("QWEN_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("QWEN_API_KEY").ok();
        env::remove_var("QWEN_API_KEY");
        let result = QwenEmbeddingsConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("QWEN_API_KEY"));
        restore("QWEN_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_uses_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("QWEN_API_KEY", "key");
        let old_url = save_and_set("QWEN_BASE_URL", "https://custom.api.com");
        let old_model = save_and_set("QWEN_EMBED_MODEL", "custom-model");
        let config = QwenEmbeddingsConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, "https://custom.api.com");
        assert_eq!(config.model, "custom-model");
        restore("QWEN_API_KEY", old_key);
        restore("QWEN_BASE_URL", old_url);
        restore("QWEN_EMBED_MODEL", old_model);
    }

    #[test]
    fn test_from_env_result_uses_defaults_for_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("QWEN_API_KEY", "key");
        let old_url = env::var("QWEN_BASE_URL").ok();
        env::remove_var("QWEN_BASE_URL");
        let old_model = env::var("QWEN_EMBED_MODEL").ok();
        env::remove_var("QWEN_EMBED_MODEL");
        let config = QwenEmbeddingsConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, QWEN_BASE_URL.to_string());
        assert_eq!(config.model, QWEN_EMBED_MODEL);
        restore("QWEN_API_KEY", old_key);
        restore("QWEN_BASE_URL", old_url);
        restore("QWEN_EMBED_MODEL", old_model);
    }

    #[test]
    fn test_embeddings_from_env_result_ok() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("QWEN_API_KEY", "test-key");
        assert!(QwenEmbeddings::from_env_result().is_ok());
        restore("QWEN_API_KEY", old);
    }

    #[test]
    fn test_embeddings_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("QWEN_API_KEY").ok();
        env::remove_var("QWEN_API_KEY");
        assert!(QwenEmbeddings::from_env_result().is_err());
        restore("QWEN_API_KEY", old);
    }
}
