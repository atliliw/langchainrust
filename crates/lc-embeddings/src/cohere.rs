// lc-embeddings/src/cohere.rs
//! Cohere Embeddings — embed-english-v3.0 / embed-multilingual-v3.0.
//!
//! Uses Cohere's v2/embed endpoint for generating text embeddings.

use async_trait::async_trait;
use lc_core::http::{HttpClient, RequestOptions};
use serde::Deserialize;
use serde_json::json;

use crate::{EmbeddingError, Embeddings};

/// Default Cohere embedding model.
pub const COHERE_EMBED_MODEL: &str = "embed-english-v3.0";

/// Cohere API base URL.
pub const COHERE_EMBED_BASE_URL: &str = "https://api.cohere.com/v2";

/// Maximum number of texts per v2/embed request.
///
/// Cohere documents the limit as "Maximum 96" for the `texts` array
/// (<https://docs.cohere.com/reference/embed>); larger batches are split into
/// 96-item chunks by the client.
pub const COHERE_MAX_BATCH: usize = 96;

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
    /// Wire token for the v2/embed `input_type` field.
    pub(crate) fn as_str(&self) -> &'static str {
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

/// Cohere v2 embedding response.
///
/// This client always sends `embedding_types: ["float"]`, so the v2 response
/// shape is `{"embeddings": {"float": [[...], ...]}}`. The legacy flat
/// `{"embeddings": [[...]]}` shape is accepted as a fallback.
#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    embeddings: CohereEmbeddingsBody,
}

/// The two shapes Cohere uses for the `embeddings` field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CohereEmbeddingsBody {
    /// v2 typed response: `{"float": [[...]], ...}`.
    Typed(CohereTypedEmbeddings),
    /// Legacy flat response: `[[...], ...]`.
    Flat(Vec<Vec<f32>>),
}

#[derive(Debug, Default, Deserialize)]
struct CohereTypedEmbeddings {
    #[serde(default)]
    float: Vec<Vec<f32>>,
}

impl CohereEmbeddingsBody {
    /// Extract the float vectors, erroring on an empty/unsupported payload
    /// (e.g. only quantized types were returned) instead of silently handing
    /// downstream an empty batch.
    fn into_float_vectors(self) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let vectors = match self {
            CohereEmbeddingsBody::Typed(typed) => typed.float,
            CohereEmbeddingsBody::Flat(vectors) => vectors,
        };
        if vectors.is_empty() {
            return Err(EmbeddingError::ParseError(
                "Cohere response contains no float embeddings (expected embeddings.float)"
                    .to_string(),
            ));
        }
        Ok(vectors)
    }
}

/// Cohere embedding provider.
pub struct CohereEmbeddings {
    config: CohereEmbeddingsConfig,
    http: HttpClient,
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
        let http = HttpClient::api()
            .build()
            .map_err(|e| EmbeddingError::Config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            config,
            http,
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

    /// Per-request options: bearer auth (the unified client defaults to
    /// `no_proxy()`, matching this client's historical behavior).
    fn request_options(&self) -> RequestOptions {
        RequestOptions::new().bearer(self.config.api_key.clone())
    }

    /// Maps unified-layer errors onto [`EmbeddingError`]: an error status keeps
    /// its status code and body text; transport/timeout/build errors collapse
    /// to [`EmbeddingError::HttpError`].
    fn map_http_error(err: lc_core::http::HttpError) -> EmbeddingError {
        match err {
            lc_core::http::HttpError::Status { status, body } => {
                EmbeddingError::ApiError(format!("HTTP {status}: {body}"))
            }
            other => EmbeddingError::HttpError(other.to_string()),
        }
    }

    /// Posts one `/embed` request for a chunk of at most [`COHERE_MAX_BATCH`]
    /// texts and returns one float vector per text, order preserved.
    async fn embed_chunk(
        &self,
        texts: &[&str],
        input_type: CohereEmbedInputType,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let url = format!("{}/embed", self.config.base_url);
        let body = json!({
            "model": self.config.model,
            "input_type": input_type.as_str(),
            "texts": texts,
            "embedding_types": ["float"],
        });

        // The unified layer retries the closed retryable-status set (429/5xx);
        // POST bodies are only re-sent per the pre-dispatch safety default.
        let response = self
            .http
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(Self::map_http_error)?;

        let embed_response: CohereEmbedResponse =
            serde_json::from_str(&response.body).map_err(|e| {
                let preview: String = response.body.chars().take(200).collect();
                EmbeddingError::ParseError(format!("{e} - body: {preview}"))
            })?;

        let vectors = embed_response.embeddings.into_float_vectors()?;
        // Every requested text must come back; a short/long batch would
        // silently misalign downstream consumers (P0-1 BatchMismatch).
        if vectors.len() != texts.len() {
            return Err(EmbeddingError::BatchMismatch {
                expected: texts.len(),
                actual: vectors.len(),
            });
        }
        Ok(vectors)
    }
}

#[async_trait]
impl Embeddings for CohereEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // P1-1: add the empty-input check Cohere lacks, consistent with other providers' contract.
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let mut vectors = self.embed_chunk(&[text], self.config.input_type).await?;
        let mut embedding = vectors.swap_remove(0);
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

        // 0.25.0: the v2/embed `texts` array is capped at COHERE_MAX_BATCH (96).
        // Split larger inputs into chunks and concatenate in request order;
        // each chunk keeps its own BatchMismatch check. The configured
        // `input_type` is honored verbatim (it used to be hardcoded to
        // search_document here, silently overriding explicit configuration).
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(COHERE_MAX_BATCH) {
            let mut chunk_vectors = self.embed_chunk(chunk, self.config.input_type).await?;
            for v in chunk_vectors.iter_mut() {
                crate::l2_normalize(v);
            }
            embeddings.append(&mut chunk_vectors);
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
    use crate::test_support::{
        spawn_json_handler_stub, spawn_json_recording_stub, spawn_status_stub,
    };
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// P2-5: Cohere also wires in 429 retry.
    #[tokio::test]
    async fn test_embed_query_retries_on_429() {
        let success_body = r#"{"embeddings":{"float":[[0.6,0.8]]}}"#;
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
        // Real Cohere v2 shape: embeddings.float, one vector per requested text.
        let base_url = spawn_json_handler_stub(Arc::new(move |body: serde_json::Value| {
            let wanted = body
                .get("texts")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let n = wanted.saturating_sub(1);
            let vectors: Vec<serde_json::Value> =
                (0..n).map(|_| serde_json::json!([0.1, 0.2])).collect();
            serde_json::json!({ "embeddings": { "float": vectors } })
        }))
        .await
        .0;
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

    /// 0.25.0 B3: 200 texts must be split 96/96/8 across three requests, and
    /// the concatenated vectors must stay aligned with the input order.
    #[tokio::test]
    async fn test_embed_documents_chunks_at_96_and_preserves_order() {
        use std::sync::atomic::AtomicUsize;
        let next_global_index = Arc::new(AtomicUsize::new(0));
        let handler_index = next_global_index.clone();
        let (base_url, bodies) =
            spawn_json_handler_stub(Arc::new(move |body: serde_json::Value| {
                let wanted = body
                    .get("texts")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                // Encode each vector's global index so order alignment survives
                // chunking and L2 normalization (ratio of the two components).
                let vectors: Vec<serde_json::Value> = (0..wanted)
                    .map(|_| {
                        let global = handler_index.fetch_add(1, Ordering::SeqCst) as f32;
                        serde_json::json!([global, 1.0])
                    })
                    .collect();
                serde_json::json!({ "embeddings": { "float": vectors } })
            }))
            .await;
        let config = CohereEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::SearchDocument,
        };
        let embeddings = CohereEmbeddings::new(config).unwrap();

        let texts: Vec<String> = (0..200).map(|i| format!("text-{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let result = embeddings.embed_documents(&refs).await.unwrap();
        assert_eq!(result.len(), 200);

        let chunk_sizes: Vec<usize> = bodies
            .lock()
            .unwrap()
            .iter()
            .map(|b| {
                b.get("texts")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0)
            })
            .collect();
        assert_eq!(chunk_sizes, vec![96, 96, 8]);

        // Normalized [i, 1.0] keeps component ratio i: check chunk boundaries.
        for i in [0usize, 95, 96, 199] {
            let ratio = result[i][0] / result[i][1];
            assert!(
                (ratio - i as f32).abs() < 0.05,
                "vector {i} misaligned: ratio {ratio}"
            );
        }
    }

    /// 0.25.0 B3: `embed_documents` must send the configured `input_type`
    /// instead of hardcoding `search_document`.
    #[tokio::test]
    async fn test_embed_documents_honors_configured_input_type() {
        let (base_url, bodies) = spawn_json_recording_stub(serde_json::json!({
            "embeddings": { "float": [[0.6, 0.8]] }
        }))
        .await;
        let config = CohereEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::Classification,
        };
        let embeddings = CohereEmbeddings::new(config).unwrap();

        embeddings.embed_documents(&["a"]).await.unwrap();
        assert_eq!(bodies.lock().unwrap()[0]["input_type"], "classification");
    }

    /// 0.25.0 B3: a non-2xx response surfaces as an API error carrying the
    /// status code and the server body text.
    #[tokio::test]
    async fn test_embed_documents_non_2xx_surfaces_status_and_body() {
        // First request gets 400 "transient"; 400 is not in the retryable set.
        let (base_url, requests) =
            spawn_status_stub(400, 1, 200, r#"{"embeddings":{"float":[]}}"#).await;
        let config = CohereEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_EMBED_MODEL.into(),
            input_type: CohereEmbedInputType::SearchDocument,
        };
        let embeddings = CohereEmbeddings::new(config).unwrap();

        let err = embeddings.embed_documents(&["a"]).await.unwrap_err();
        match err {
            EmbeddingError::ApiError(msg) => {
                assert!(msg.contains("HTTP 400"), "message: {msg}");
                assert!(msg.contains("transient"), "message: {msg}");
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "400 must not be retried"
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
