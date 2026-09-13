// lc-embeddings/src/vision_cohere.rs
//! Cohere **vision** embeddings — Embed v4 (multimodal) over `v2/embed`.
//!
//! Embed v4 maps images and text into one shared 1536-dimensional space.
//! Images must be sent inline (base64 bytes + format); the API does not
//! fetch caller-supplied URLs, so [`ImageInput::Url`] is rejected at the
//! boundary rather than fetched server-side.
//!
//! Request (images):
//! ```json
//! {
//!   "model": "embed-v4.0",
//!   "input_type": "image",
//!   "images": [{"image_bytes": {"bytes": "<base64>"}, "format": "png"}],
//!   "embedding_types": ["float"]
//! }
//! ```
//! Response: `{"embeddings": {"float": [[...], ...]}}`.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::retry::{post_json_with_retry, DEFAULT_RETRY};
use crate::vision::{ImageInput, VisionEmbeddings};
use crate::{l2_normalize, CohereEmbedInputType, EmbeddingError};

/// Default Cohere multimodal embedding model.
pub const COHERE_VISION_EMBED_MODEL: &str = "embed-v4.0";

/// Output dimension of Embed v4 (English and multilingual variants share it).
pub const COHERE_VISION_DIMENSION: usize = 1536;

/// Cohere vision embedding configuration.
#[derive(Debug, Clone)]
pub struct CohereVisionEmbeddingsConfig {
    /// Cohere API key.
    pub api_key: String,
    /// Base URL for the Cohere API (v2).
    pub base_url: String,
    /// Multimodal model name.
    pub model: String,
    /// Input type used for [`VisionEmbeddings::embed_text`] (image batches
    /// always use `"image"` per the API).
    pub text_input_type: CohereEmbedInputType,
}

impl Default for CohereVisionEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: crate::cohere::COHERE_EMBED_BASE_URL.to_string(),
            model: COHERE_VISION_EMBED_MODEL.to_string(),
            text_input_type: CohereEmbedInputType::SearchQuery,
        }
    }
}

impl CohereVisionEmbeddingsConfig {
    /// Creates a new config with the given API key and defaults
    /// (`embed-v4.0`, `search_query` for text queries).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates config from `COHERE_API_KEY` (and optional `COHERE_BASE_URL` /
    /// `COHERE_VISION_EMBED_MODEL`) environment variables.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("COHERE_API_KEY").map_err(|_| {
            EmbeddingError::Config("COHERE_API_KEY environment variable not set".to_string())
        })?;
        let base_url = std::env::var("COHERE_BASE_URL")
            .unwrap_or_else(|_| crate::cohere::COHERE_EMBED_BASE_URL.to_string());
        let model = std::env::var("COHERE_VISION_EMBED_MODEL")
            .unwrap_or_else(|_| COHERE_VISION_EMBED_MODEL.to_string());
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

    /// Sets the input type used for text queries/documents.
    pub fn with_text_input_type(mut self, input_type: CohereEmbedInputType) -> Self {
        self.text_input_type = input_type;
        self
    }
}

/// Cohere Embed v4 multimodal embedding provider.
pub struct CohereVisionEmbeddings {
    config: CohereVisionEmbeddingsConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl std::fmt::Debug for CohereVisionEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereVisionEmbeddings")
            .field("model", &self.config.model)
            .finish()
    }
}

/// v2/embed response (with `embedding_types: ["float"]`).
#[derive(Debug, Deserialize)]
struct CohereV2EmbedResponse {
    embeddings: CohereV2FloatEmbeddings,
}

#[derive(Debug, Deserialize)]
struct CohereV2FloatEmbeddings {
    float: Vec<Vec<f32>>,
}

impl CohereVisionEmbeddings {
    /// Creates the provider, failing fast on an empty API key or an unknown
    /// model dimension.
    pub fn new(config: CohereVisionEmbeddingsConfig) -> Result<Self, EmbeddingError> {
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

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        Self::new(CohereVisionEmbeddingsConfig::from_env_result()?)
    }

    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        match model {
            // Embed v4 family is 1536-dimensional (multimodal, multilingual).
            "embed-v4.0" => Ok(COHERE_VISION_DIMENSION),
            other => Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for Cohere vision model '{other}' \
                 (supported: 'embed-v4.0')"
            ))),
        }
    }

    /// Maps an image MIME type to Cohere's `format` token.
    fn image_format(mime: &str) -> Result<&'static str, EmbeddingError> {
        match mime {
            "image/png" => Ok("png"),
            "image/jpeg" | "image/jpg" => Ok("jpeg"),
            "image/webp" => Ok("webp"),
            "image/gif" => Ok("gif"),
            other => Err(EmbeddingError::Config(format!(
                "Cohere vision embeddings support png/jpeg/webp/gif images, got {other}"
            ))),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/embed", self.config.base_url.trim_end_matches('/'))
    }

    /// POST helper shared by text and image calls: applies retry, surfaces
    /// non-2xx bodies (never silently defaulting), and extracts the float
    /// embedding matrix.
    async fn post_float_embeddings(
        &self,
        body: &serde_json::Value,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let response = post_json_with_retry(
            &self.client,
            &self.endpoint(),
            &self.config.api_key,
            body,
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

        let parsed: CohereV2EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
        Ok(parsed.embeddings.float)
    }
}

#[async_trait]
impl VisionEmbeddings for CohereVisionEmbeddings {
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

        // Inline-only: plain URLs are rejected before any request is made.
        let mut payload = Vec::with_capacity(images.len());
        for image in images {
            let (bytes, mime) = image.inline_parts()?;
            let format = Self::image_format(&mime)?;
            payload.push(json!({
                "image_bytes": {"bytes": bytes},
                "format": format,
            }));
        }

        let body = json!({
            "model": self.config.model,
            "input_type": "image",
            "images": payload,
            "embedding_types": ["float"],
        });

        let mut embeddings = self.post_float_embeddings(&body).await?;
        if embeddings.len() != images.len() {
            return Err(EmbeddingError::BatchMismatch {
                expected: images.len(),
                actual: embeddings.len(),
            });
        }
        for vector in embeddings.iter_mut() {
            l2_normalize(vector);
        }
        Ok(embeddings)
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let body = json!({
            "model": self.config.model,
            "input_type": self.config.text_input_type.as_str(),
            "texts": [text],
            "embedding_types": ["float"],
        });

        let mut embeddings = self.post_float_embeddings(&body).await?;
        let mut embedding = embeddings
            .drain(..)
            .next()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding in response".to_string()))?;
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
    use crate::test_support::{spawn_json_recording_stub, spawn_status_stub};
    use std::sync::atomic::Ordering;

    fn config_with(base_url: String) -> CohereVisionEmbeddingsConfig {
        CohereVisionEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: COHERE_VISION_EMBED_MODEL.into(),
            text_input_type: CohereEmbedInputType::SearchQuery,
        }
    }

    #[test]
    fn construction_validates_key_and_model() {
        let mut bad_key = config_with("http://127.0.0.1:1".into());
        bad_key.api_key = String::new();
        assert!(matches!(
            CohereVisionEmbeddings::new(bad_key),
            Err(EmbeddingError::Config(_))
        ));

        let mut bad_model = config_with("http://127.0.0.1:1".into());
        bad_model.model = "unknown-vision".into();
        assert!(matches!(
            CohereVisionEmbeddings::new(bad_model),
            Err(EmbeddingError::Config(_))
        ));

        let ok = CohereVisionEmbeddings::new(config_with("http://127.0.0.1:1".into())).unwrap();
        assert_eq!(ok.dimension(), COHERE_VISION_DIMENSION);
        assert_eq!(ok.model_name(), COHERE_VISION_EMBED_MODEL);
    }

    #[test]
    fn image_format_maps_known_mimes() {
        assert_eq!(
            CohereVisionEmbeddings::image_format("image/png").unwrap(),
            "png"
        );
        assert_eq!(
            CohereVisionEmbeddings::image_format("image/jpeg").unwrap(),
            "jpeg"
        );
        assert_eq!(
            CohereVisionEmbeddings::image_format("image/webp").unwrap(),
            "webp"
        );
        assert!(CohereVisionEmbeddings::image_format("image/bmp").is_err());
    }

    /// B7 snapshot: image requests use the v2 image shape with inline bytes.
    #[tokio::test]
    async fn image_request_body_matches_v2_snapshot() {
        let response = json!({"embeddings": {"float": [[0.6, 0.8], [0.8, -0.6]]}});
        let (base_url, bodies) = spawn_json_recording_stub(response).await;
        let embeddings = CohereVisionEmbeddings::new(config_with(base_url)).unwrap();

        let images = vec![
            ImageInput::from_base64("aW1n", "image/png"),
            ImageInput::from_data_uri("data:image/jpeg;base64,amZlZw"),
        ];
        let vectors = embeddings.embed_images(&images).await.unwrap();
        assert_eq!(vectors.len(), 2);
        // L2-normalized.
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let body = &recorded[0];
        assert_eq!(body["model"], COHERE_VISION_EMBED_MODEL);
        assert_eq!(body["input_type"], "image");
        assert_eq!(body["embedding_types"][0], "float");
        assert_eq!(
            body["images"],
            json!([
                {"image_bytes": {"bytes": "aW1n"}, "format": "png"},
                {"image_bytes": {"bytes": "amZlZw"}, "format": "jpeg"},
            ])
        );
    }

    #[tokio::test]
    async fn text_request_uses_configured_input_type() {
        let response = json!({"embeddings": {"float": [[0.6, 0.8]]}});
        let (base_url, bodies) = spawn_json_recording_stub(response).await;
        let cfg = config_with(base_url).with_text_input_type(CohereEmbedInputType::SearchDocument);
        let embeddings = CohereVisionEmbeddings::new(cfg).unwrap();

        let v = embeddings.embed_text("a red shoe").await.unwrap();
        assert_eq!(v.len(), 2);

        let recorded = bodies.lock().unwrap();
        assert_eq!(recorded[0]["input_type"], "search_document");
        assert_eq!(recorded[0]["texts"][0], "a red shoe");
    }

    /// Plain URLs are rejected (Cohere has no URL fetch field; fetching here
    /// would bypass the SSRF guard).
    #[tokio::test]
    async fn plain_url_images_are_rejected() {
        let response = json!({"embeddings": {"float": [[0.6, 0.8]]}});
        let (base_url, _bodies) = spawn_json_recording_stub(response).await;
        let embeddings = CohereVisionEmbeddings::new(config_with(base_url)).unwrap();

        let err = embeddings
            .embed_image(&ImageInput::from_url("https://example.com/a.png"))
            .await
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    #[tokio::test]
    async fn batch_mismatch_is_reported() {
        // Two images sent, stub returns one vector.
        let response = json!({"embeddings": {"float": [[0.6, 0.8]]}});
        let (base_url, _bodies) = spawn_json_recording_stub(response).await;
        let embeddings = CohereVisionEmbeddings::new(config_with(base_url)).unwrap();

        let images = vec![
            ImageInput::from_base64("aW1n", "image/png"),
            ImageInput::from_base64("amZlZw", "image/jpeg"),
        ];
        let err = embeddings.embed_images(&images).await.unwrap_err();
        assert!(matches!(
            err,
            EmbeddingError::BatchMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[tokio::test]
    async fn retries_on_429_and_rejects_empty_inputs() {
        let success_body = r#"{"embeddings":{"float":[[0.6,0.8]]}}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let embeddings = CohereVisionEmbeddings::new(config_with(base_url)).unwrap();

        let v = embeddings
            .embed_text("hello")
            .await
            .expect("should retry after two 429s");
        assert_eq!(v.len(), 2);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");

        assert!(matches!(
            embeddings.embed_text("  ").await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert_eq!(
            embeddings.embed_images(&[]).await.unwrap(),
            Vec::<Vec<f32>>::new()
        );
    }
}
