// lc-embeddings/src/vision_qwen.rs
//! Qwen / DashScope **native** multimodal embeddings — `multimodal-embedding-v1`.
//!
//! Unlike [`crate::QwenEmbeddings`] (the OpenAI-compatible `/embeddings`
//! alias over text models), multimodal embeddings use DashScope's native
//! endpoint with a different request/response dialect:
//!
//! ```text
//! POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding-v1
//! {
//!   "model": "multimodal-embedding-v1",
//!   "input": {
//!     "contents": [
//!       [{"image": "https://example.com/cat.png"}],
//!       [{"text": "一只猫"}]
//!     ]
//!   }
//! }
//! ```
//!
//! Both modalities land in one shared 1024-dimensional space. Images may be
//! public URLs (DashScope fetches them) or base64 data URIs; text and image
//! contents may be mixed within a single request. Responses carry an `index`
//! per embedding, which we always sort back to request order.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::retry::{post_json_with_retry, DEFAULT_RETRY};
use crate::vision::{ImageInput, VisionEmbeddings};
use crate::{l2_normalize, EmbeddingError};

/// DashScope native multimodal embedding service base URL.
pub const QWEN_VISION_BASE_URL: &str =
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding";

/// Default multimodal embedding model.
pub const QWEN_VISION_EMBED_MODEL: &str = "multimodal-embedding-v1";

/// Output dimension of `multimodal-embedding-v1`.
pub const QWEN_VISION_DIMENSION: usize = 1024;

/// DashScope accepts at most 10 `contents` entries per multimodal request.
const MAX_CONTENTS_PER_REQUEST: usize = 10;

/// Configuration for the DashScope native multimodal embeddings API.
#[derive(Debug, Clone)]
pub struct QwenVisionEmbeddingsConfig {
    /// DashScope API key.
    pub api_key: String,
    /// Multimodal embedding service base URL.
    pub base_url: String,
    /// Model name (model segment of the service path).
    pub model: String,
}

impl Default for QwenVisionEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("QWEN_API_KEY").unwrap_or_default(),
            base_url: QWEN_VISION_BASE_URL.to_string(),
            model: QWEN_VISION_EMBED_MODEL.to_string(),
        }
    }
}

impl QwenVisionEmbeddingsConfig {
    /// Creates a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates config from environment variables.
    ///
    /// Reads `QWEN_API_KEY` (required) and the optional
    /// `QWEN_VISION_BASE_URL` / `QWEN_VISION_EMBED_MODEL`.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("QWEN_API_KEY").map_err(|_| {
            EmbeddingError::Config("QWEN_API_KEY environment variable not set".to_string())
        })?;
        let base_url = std::env::var("QWEN_VISION_BASE_URL")
            .unwrap_or_else(|_| QWEN_VISION_BASE_URL.to_string());
        let model = std::env::var("QWEN_VISION_EMBED_MODEL")
            .unwrap_or_else(|_| QWEN_VISION_EMBED_MODEL.to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
        })
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// DashScope multimodal embeddings provider.
pub struct QwenVisionEmbeddings {
    config: QwenVisionEmbeddingsConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl std::fmt::Debug for QwenVisionEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenVisionEmbeddings")
            .field("model", &self.config.model)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct DashScopeMmResponse {
    output: Option<DashScopeMmOutput>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DashScopeMmOutput {
    embeddings: Vec<DashScopeMmEmbedding>,
}

#[derive(Debug, Deserialize)]
struct DashScopeMmEmbedding {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

impl QwenVisionEmbeddings {
    /// Creates the provider, failing fast on an empty API key or an unknown
    /// model dimension.
    pub fn new(config: QwenVisionEmbeddingsConfig) -> Result<Self, EmbeddingError> {
        if config.api_key.trim().is_empty() {
            return Err(EmbeddingError::Config("QWEN_API_KEY is empty".to_string()));
        }
        let dimension = Self::dimension_for(&config.model)?;
        Ok(Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        })
    }

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        Self::new(QwenVisionEmbeddingsConfig::from_env_result()?)
    }

    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        match model {
            QWEN_VISION_EMBED_MODEL => Ok(QWEN_VISION_DIMENSION),
            other => Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for Qwen multimodal model '{other}' \
                 (supported: '{QWEN_VISION_EMBED_MODEL}')"
            ))),
        }
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            self.config.model
        )
    }

    /// One request for a slice of already-built `contents` entries; returns
    /// vectors reordered by the provider-supplied `index`.
    async fn post_contents(
        &self,
        contents: &[serde_json::Value],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let body = json!({
            "model": self.config.model,
            "input": {"contents": contents},
        });

        let response = post_json_with_retry(
            &self.client,
            &self.endpoint(),
            &self.config.api_key,
            &body,
            &DEFAULT_RETRY,
        )
        .await
        .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.map_err(|e| {
                EmbeddingError::HttpError(format!("failed to read error response body: {e}"))
            })?;
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let parsed: DashScopeMmResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let output = match parsed.output {
            Some(output) if !output.embeddings.is_empty() => output,
            _ => {
                return Err(EmbeddingError::ApiError(
                    parsed
                        .code
                        .zip(parsed.message)
                        .map(|(code, message)| format!("{code}: {message}"))
                        .unwrap_or_else(|| "No embeddings in DashScope response".to_string()),
                ));
            }
        };

        // Providers may reorder entries; restore request order via `index`.
        let mut indexed = output.embeddings;
        indexed.sort_by_key(|e| e.index);
        Ok(indexed.into_iter().map(|e| e.embedding).collect())
    }
}

#[async_trait]
impl VisionEmbeddings for QwenVisionEmbeddings {
    async fn embed_image(&self, image: &ImageInput) -> Result<Vec<f32>, EmbeddingError> {
        let mut batch = self.embed_images(std::slice::from_ref(image)).await?;
        batch
            .pop()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding in response".to_string()))
    }

    async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // Build all content entries first so a bad input fails before any request.
        let entries: Result<Vec<serde_json::Value>, EmbeddingError> = images
            .iter()
            .map(|image| {
                let reference = image.reference()?;
                Ok(json!([{"image": reference}]))
            })
            .collect();
        let entries = entries?;

        let mut all = Vec::with_capacity(images.len());
        for chunk in entries.chunks(MAX_CONTENTS_PER_REQUEST) {
            all.extend(self.post_contents(chunk).await?);
        }

        if all.len() != images.len() {
            return Err(EmbeddingError::BatchMismatch {
                expected: images.len(),
                actual: all.len(),
            });
        }
        for vector in all.iter_mut() {
            l2_normalize(vector);
        }
        Ok(all)
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let contents = [json!([{"text": text}])];
        let mut batch = self.post_contents(&contents).await?;
        if batch.len() != 1 {
            return Err(EmbeddingError::BatchMismatch {
                expected: 1,
                actual: batch.len(),
            });
        }
        let mut embedding = batch.remove(0);
        l2_normalize(&mut embedding);
        Ok(embedding)
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

    fn config_with(base_url: String) -> QwenVisionEmbeddingsConfig {
        QwenVisionEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: QWEN_VISION_EMBED_MODEL.into(),
        }
    }

    fn two_vector_response() -> serde_json::Value {
        json!({
            "output": {
                "embeddings": [
                    // Deliberately reversed indices to verify reordering.
                    {"embedding": [0.8, -0.6], "index": 1, "type": 0},
                    {"embedding": [0.6, 0.8], "index": 0, "type": 1}
                ]
            },
            "request_id": "abc"
        })
    }

    #[test]
    fn construction_validates_key_and_model() {
        let mut bad_key = config_with("http://127.0.0.1:1".into());
        bad_key.api_key = String::new();
        assert!(matches!(
            QwenVisionEmbeddings::new(bad_key),
            Err(EmbeddingError::Config(_))
        ));

        let mut bad_model = config_with("http://127.0.0.1:1".into());
        bad_model.model = "some-other-model".into();
        assert!(matches!(
            QwenVisionEmbeddings::new(bad_model),
            Err(EmbeddingError::Config(_))
        ));

        let ok = QwenVisionEmbeddings::new(config_with("http://127.0.0.1:1".into())).unwrap();
        assert_eq!(ok.dimension(), QWEN_VISION_DIMENSION);
        assert_eq!(ok.model_name(), QWEN_VISION_EMBED_MODEL);
    }

    /// B7 snapshot: native DashScope request shape (`input.contents`).
    #[tokio::test]
    async fn image_request_body_matches_native_snapshot() {
        let (base_url, bodies) = spawn_json_recording_stub(two_vector_response()).await;
        let embeddings = QwenVisionEmbeddings::new(config_with(base_url)).unwrap();

        let images = vec![
            ImageInput::from_url("https://example.com/cat.png"),
            ImageInput::from_base64("aW1n", "image/png"),
        ];
        let vectors = embeddings.embed_images(&images).await.unwrap();
        assert_eq!(vectors.len(), 2);
        // Reordered to request order via `index`.
        assert_eq!(vectors[0], vec![0.6, 0.8]);
        assert_eq!(vectors[1], vec![0.8, -0.6]);

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0]["model"], QWEN_VISION_EMBED_MODEL);
        assert_eq!(
            recorded[0]["input"]["contents"],
            json!([
                [{"image": "https://example.com/cat.png"}],
                [{"image": "data:image/png;base64,aW1n"}],
            ])
        );
    }

    #[tokio::test]
    async fn text_request_uses_text_content() {
        let response = json!({"output": {"embeddings": [{"embedding": [0.6, 0.8], "index": 0}]}});
        let (base_url, bodies) = spawn_json_recording_stub(response).await;
        let embeddings = QwenVisionEmbeddings::new(config_with(base_url)).unwrap();

        let v = embeddings.embed_text("一只猫").await.unwrap();
        assert_eq!(v.len(), 2);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);

        let recorded = bodies.lock().unwrap();
        assert_eq!(
            recorded[0]["input"]["contents"],
            json!([[{"text": "一只猫"}]])
        );
    }

    #[tokio::test]
    async fn batches_larger_than_ten_contents_are_chunked_in_order() {
        // The stub mirrors each request: one vector per `contents` entry,
        // encoded with its in-request position, so chunk boundaries and
        // order preservation are both observable.
        let handler = Arc::new(|request: serde_json::Value| {
            let n = request["input"]["contents"]
                .as_array()
                .map(|a| a.len())
                .unwrap_or(0);
            json!({
                "output": {
                    "embeddings": (0..n)
                        .map(|i| json!({"embedding": [i as f32, 0.0], "index": i}))
                        .collect::<Vec<_>>()
                }
            })
        });
        let (base_url, bodies) = spawn_json_handler_stub(handler).await;
        let embeddings = QwenVisionEmbeddings::new(config_with(base_url)).unwrap();

        let images: Vec<ImageInput> = (0..21)
            .map(|i| ImageInput::from_url(format!("https://example.com/img{i}.png")))
            .collect();
        let vectors = embeddings.embed_images(&images).await.unwrap();
        assert_eq!(vectors.len(), 21, "10 + 10 + 1 across three requests");
        // Vectors are normalized, so compare positions across chunks: every
        // chunk restarts the provider-side index, proving request order.
        assert_eq!(vectors[0], vectors[10]);
        assert_eq!(vectors[10], vectors[20]);
        assert_eq!(vectors[9], vectors[19]);
        assert_ne!(vectors[9], vectors[10], "chunk boundary must not reorder");

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        assert_eq!(
            recorded[0]["input"]["contents"].as_array().unwrap().len(),
            10
        );
        assert_eq!(
            recorded[1]["input"]["contents"].as_array().unwrap().len(),
            10
        );
        assert_eq!(
            recorded[2]["input"]["contents"].as_array().unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn retries_on_429_and_enforces_empty_contract() {
        let success_body = r#"{"output":{"embeddings":[{"embedding":[0.6,0.8],"index":0}]}}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let embeddings = QwenVisionEmbeddings::new(config_with(base_url)).unwrap();

        let v = embeddings
            .embed_text("hello")
            .await
            .expect("should retry after two 429s");
        assert_eq!(v.len(), 2);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");

        assert!(matches!(
            embeddings.embed_text(" ").await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert_eq!(
            embeddings.embed_images(&[]).await.unwrap(),
            Vec::<Vec<f32>>::new()
        );
    }

    #[tokio::test]
    async fn provider_error_payload_is_surfaced() {
        let response = json!({"code": "InvalidApiKey", "message": "key invalid"});
        let (base_url, _bodies) = spawn_json_recording_stub(response).await;
        let embeddings = QwenVisionEmbeddings::new(config_with(base_url)).unwrap();

        let err = embeddings.embed_text("hello").await.unwrap_err();
        match err {
            EmbeddingError::ApiError(message) => assert!(message.contains("InvalidApiKey")),
            other => panic!("expected ApiError, got {other:?}"),
        }
    }
}
