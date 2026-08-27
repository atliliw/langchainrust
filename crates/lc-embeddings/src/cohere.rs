// lc-embeddings/src/cohere.rs
//! Cohere Embeddings — embed-english-v3.0 / embed-multilingual-v3.0.
//!
//! Uses Cohere's v2/embed endpoint for generating text embeddings.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::{EmbeddingError, Embeddings};

/// Default Cohere embedding model.
pub const COHERE_EMBED_MODEL: &str = "embed-english-v3.0";

/// Cohere API base URL.
pub const COHERE_EMBED_BASE_URL: &str = "https://api.cohere.com/v2";

/// Cohere embedding input type.
#[derive(Debug, Clone, Copy)]
pub enum CohereEmbedInputType {
    /// Search query embedding.
    SearchQuery,
    /// Search document embedding.
    SearchDocument,
    /// Classification embedding.
    Classification,
    /// Clustering embedding.
    Clustering,
}

impl CohereEmbedInputType {
    fn as_str(&self) -> &'static str {
        match self {
            CohereEmbedInputType::SearchQuery => "search_query",
            CohereEmbedInputType::SearchDocument => "search_document",
            CohereEmbedInputType::Classification => "classification",
            CohereEmbedInputType::Clustering => "clustering",
        }
    }
}

/// Cohere embedding configuration.
#[derive(Debug, Clone)]
pub struct CohereEmbeddingsConfig {
    /// Cohere API key.
    pub api_key: String,
    /// Base URL for the Cohere embeddings API.
    pub base_url: String,
    /// Embedding model name.
    pub model: String,
    /// Input type for the embedding request.
    pub input_type: CohereEmbedInputType,
}

impl Default for CohereEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: COHERE_EMBED_BASE_URL.to_string(),
            model: COHERE_EMBED_MODEL.to_string(),
            input_type: CohereEmbedInputType::SearchQuery,
        }
    }
}

impl CohereEmbeddingsConfig {
    /// Creates a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates config from environment variables.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("COHERE_API_KEY").map_err(|_| {
            EmbeddingError::Config("COHERE_API_KEY environment variable not set".to_string())
        })?;
        let base_url =
            std::env::var("COHERE_BASE_URL").unwrap_or_else(|_| COHERE_EMBED_BASE_URL.to_string());
        let model =
            std::env::var("COHERE_EMBED_MODEL").unwrap_or_else(|_| COHERE_EMBED_MODEL.to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the input type.
    pub fn with_input_type(mut self, input_type: CohereEmbedInputType) -> Self {
        self.input_type = input_type;
        self
    }
}

/// Cohere embedding response.
#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    data: Vec<CohereEmbedData>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbedData {
    embedding: Vec<f32>,
}

/// Cohere embedding provider.
pub struct CohereEmbeddings {
    config: CohereEmbeddingsConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl std::fmt::Debug for CohereEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereEmbeddings")
            .field("model", &self.config.model)
            .finish()
    }
}

impl CohereEmbeddings {
    /// Creates a new CohereEmbeddings with the given configuration.
    ///
    /// Fails fast at construction (P1-3): an empty API key errors immediately. Constructs only
    /// when the model dimension is known (P1-2): the Cohere v3.0 family (english/multilingual)
    /// is always 1024-dimensional; unknown models error rather than lying with a fixed 1024.
    pub fn new(config: CohereEmbeddingsConfig) -> Result<Self, EmbeddingError> {
        if config.api_key.trim().is_empty() {
            return Err(EmbeddingError::Config(
                "COHERE_API_KEY is empty".to_string(),
            ));
        }
        let dimension = Self::dimension_for(&config.model)?;
        Ok(Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        })
    }

    /// Dimension table for known models; the Cohere v3.0 family is always 1024 (P1-2).
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        match model {
            "embed-english-v3.0" | "embed-multilingual-v3.0" => Ok(1024),
            other => Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for Cohere model '{other}' \
                 (supported: 'embed-english-v3.0', 'embed-multilingual-v3.0')"
            ))),
        }
    }

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let config = CohereEmbeddingsConfig::from_env_result()?;
        Self::new(config)
    }
}

#[async_trait]
impl Embeddings for CohereEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // P1-1: add the empty-input check Cohere lacks, consistent with other providers' contract.
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embed", self.config.base_url);
        let body = json!({
            "model": self.config.model,
            "input_type": self.config.input_type.as_str(),
            "texts": [text],
            "embedding_types": ["float"],
        });

        // P2-5: exponential backoff retry on 429/5xx.
        let response = crate::retry::post_json_with_retry(
            &self.client,
            &url,
            &self.config.api_key,
            &body,
            &crate::retry::DEFAULT_RETRY,
        )
        .await
        .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            // P1-4: the error body must also error if reading fails; do not swallow it with unwrap_or_default().
            let error_text = response.text().await.map_err(|e| {
                EmbeddingError::HttpError(format!("failed to read error response body: {e}"))
            })?;
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let embed_response: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let mut embedding = embed_response
            .data
            .first()
            .map(|d| d.embedding.clone())
            .ok_or_else(|| EmbeddingError::ApiError("No embedding in response".to_string()))?;
        // P2-8: uniform L2 normalization, guaranteeing unit length.
        crate::l2_normalize(&mut embedding);
        Ok(embedding)
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // P1-1: an empty slice is not an error (nothing to do); only empty/all-whitespace texts error — uniform with other providers.
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embed", self.config.base_url);
        let body = json!({
            "model": self.config.model,
            "input_type": CohereEmbedInputType::SearchDocument.as_str(),
            "texts": texts,
            "embedding_types": ["float"],
        });

        // P2-5: exponential backoff retry on 429/5xx.
        let response = crate::retry::post_json_with_retry(
            &self.client,
            &url,
            &self.config.api_key,
            &body,
            &crate::retry::DEFAULT_RETRY,
        )
        .await
        .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            // P1-4: the error body must also error if reading fails; do not swallow it with unwrap_or_default().
            let error_text = response.text().await.map_err(|e| {
                EmbeddingError::HttpError(format!("failed to read error response body: {e}"))
            })?;
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let embed_response: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let mut embeddings: Vec<Vec<f32>> = embed_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect();

        // P0-1: Cohere sends all texts in one request, so the returned count must match the
        // requested count, otherwise missing vectors would silently misalign downstream.
        if embeddings.len() != texts.len() {
            return Err(EmbeddingError::BatchMismatch {
                expected: texts.len(),
                actual: embeddings.len(),
            });
        }

        // P2-8: per-item uniform L2 normalization, guaranteeing unit length.
        for v in embeddings.iter_mut() {
            crate::l2_normalize(v);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{spawn_embeddings_stub, spawn_status_stub};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// P2-5: Cohere also wires in 429 retry.
    #[tokio::test]
    async fn test_embed_query_retries_on_429() {
        let success_body = r#"{"data":[{"embedding":[0.6,0.8]}]}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let config = CohereEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::SearchQuery,
        };
        let embeddings = CohereEmbeddings::new(config).unwrap();

        let v = embeddings
            .embed_query("hello")
            .await
            .expect("should retry successfully after two 429s");
        assert_eq!(v.len(), 2);
        // P2-8: the returned vector should be normalized.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
    }

    /// P0-1: Cohere returns all texts at once; a short return must explicitly report
    /// `BatchMismatch` rather than silently giving downstream fewer vectors.
    #[tokio::test]
    async fn test_embed_documents_truncated_errors() {
        let base_url = spawn_embeddings_stub(Arc::new(|n| n.saturating_sub(1))).await;
        let config = CohereEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::SearchDocument,
        };
        let embeddings = CohereEmbeddings::new(config).unwrap();

        let result = embeddings.embed_documents(&["a", "b"]).await;
        assert!(
            matches!(
                result,
                Err(EmbeddingError::BatchMismatch {
                    expected: 2,
                    actual: 1
                })
            ),
            "truncated response should report BatchMismatch, got: {:?}",
            result
        );
    }

    #[test]
    fn test_config_new() {
        let config = CohereEmbeddingsConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, COHERE_EMBED_MODEL);
    }

    #[test]
    fn test_config_builder() {
        let config = CohereEmbeddingsConfig::new("key")
            .with_model("embed-multilingual-v3.0")
            .with_input_type(CohereEmbedInputType::SearchDocument);
        assert_eq!(config.model, "embed-multilingual-v3.0");
        assert!(matches!(
            config.input_type,
            CohereEmbedInputType::SearchDocument
        ));
    }

    #[test]
    fn test_input_type_str() {
        assert_eq!(CohereEmbedInputType::SearchQuery.as_str(), "search_query");
        assert_eq!(
            CohereEmbedInputType::SearchDocument.as_str(),
            "search_document"
        );
        assert_eq!(
            CohereEmbedInputType::Classification.as_str(),
            "classification"
        );
        assert_eq!(CohereEmbedInputType::Clustering.as_str(), "clustering");
    }

    #[test]
    fn test_embeddings_new() {
        let config = CohereEmbeddingsConfig::new("key");
        let embeddings = CohereEmbeddings::new(config).unwrap();
        assert_eq!(embeddings.model_name(), COHERE_EMBED_MODEL);
        assert_eq!(embeddings.dimension(), 1024);
    }

    /// P1-3: an empty API key → `Config` error at construction (fail fast), not a delayed 401.
    #[test]
    fn test_new_rejects_empty_api_key() {
        let config = CohereEmbeddingsConfig {
            api_key: String::new(),
            base_url: COHERE_EMBED_BASE_URL.into(),
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::SearchDocument,
        };
        let err = CohereEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// P1-2: unknown model → construction-time error, never lying with a fixed 1024.
    #[test]
    fn test_new_rejects_unknown_model() {
        let config = CohereEmbeddingsConfig::new("key").with_model("some-unknown-model");
        let err = CohereEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// P1-1: empty / all-whitespace text → `Err(EmptyInput)`; an empty slice → `Ok(vec![])`.
    #[tokio::test]
    async fn test_empty_input_contract() {
        let embeddings = CohereEmbeddings::new(CohereEmbeddingsConfig::new("key")).unwrap();
        assert!(matches!(
            embeddings.embed_query("").await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert!(matches!(
            embeddings.embed_query("   ").await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert_eq!(
            embeddings.embed_documents(&[]).await.unwrap(),
            Vec::<Vec<f32>>::new()
        );
        assert!(matches!(
            embeddings.embed_documents(&["ok", " "]).await,
            Err(EmbeddingError::EmptyInput)
        ));
    }
}
