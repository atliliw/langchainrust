// lc-embeddings/src/vision.rs
//! Vision (multimodal) embeddings — image **and** text in one shared vector
//! space (B7, v0.22.4).
//!
//! Text-only [`Embeddings`] models map strings to vectors. Multimodal models
//! (Cohere Embed v4, Alibaba DashScope `multimodal-embedding-v1`) map images
//! *and* text into the **same** space, which is what makes cross-modal
//! retrieval possible: store vectors of product photos, query them with a
//! plain sentence such as “红色运动鞋”. This module defines:
//!
//! - [`ImageInput`] — provider-neutral image reference (URL / data URI / raw base64);
//! - [`VisionEmbeddings`] — the cross-modal embedding trait;
//! - [`MockVisionEmbeddings`] — deterministic offline backend for tests.
//!
//! As with [`Embeddings`], every returned vector is L2-normalized by the
//! concrete backends so cosine/dot-product results stay comparable across
//! providers.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{l2_normalize, EmbeddingError};

/// Provider-neutral image input for vision embedding models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImageInput {
    /// A complete data URI (`data:image/png;base64,<data>`). Sent inline.
    DataUri(String),
    /// A provider-fetchable image URL (support depends on the backend —
    /// DashScope fetches public URLs; the Cohere API requires inline bytes).
    Url(String),
    /// Raw base64-encoded image bytes with an explicit MIME type
    /// (e.g. `image/png`).
    Base64 {
        /// Base64-encoded image bytes (no data-URI header).
        data: String,
        /// Image MIME type.
        mime_type: String,
    },
}

impl ImageInput {
    /// Creates an image input from a provider-fetchable URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        ImageInput::Url(url.into())
    }

    /// Creates an image input from a complete data URI.
    pub fn from_data_uri(uri: impl Into<String>) -> Self {
        ImageInput::DataUri(uri.into())
    }

    /// Creates an image input from raw base64 bytes and a MIME type.
    pub fn from_base64(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        ImageInput::Base64 {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    /// Validates the input and returns its `(raw base64 bytes, MIME type)` for
    /// backends that require inline bytes (Cohere).
    ///
    /// Plain [`ImageInput::Url`] values are rejected: such backends cannot ask
    /// the model provider to perform a server-side fetch, and fetching here
    /// would route caller-supplied URLs through embedding hosts without the
    /// SSRF guard. Callers that need URL ingestion should resolve the image
    /// themselves (or use a backend with a reference field, like DashScope).
    pub(crate) fn inline_parts(&self) -> Result<(String, String), EmbeddingError> {
        match self {
            ImageInput::Base64 { data, mime_type } => {
                Self::validate_inline(data, mime_type)?;
                Ok((data.clone(), mime_type.clone()))
            }
            ImageInput::DataUri(uri) => {
                let (mime, data) = parse_data_uri(uri)?;
                Ok((data.to_string(), mime.to_string()))
            }
            ImageInput::Url(url) => Err(EmbeddingError::Config(format!(
                "this vision embedding backend requires inline image bytes, got a URL: {url}"
            ))),
        }
    }

    /// Returns the reference string for a backend `image` field (DashScope):
    /// URLs pass through, base64 inputs are rendered as data URIs.
    pub(crate) fn reference(&self) -> Result<String, EmbeddingError> {
        match self {
            ImageInput::Url(url) => {
                if url.trim().is_empty() {
                    return Err(EmbeddingError::EmptyInput);
                }
                Ok(url.clone())
            }
            ImageInput::DataUri(uri) => {
                let _ = parse_data_uri(uri)?;
                Ok(uri.clone())
            }
            ImageInput::Base64 { data, mime_type } => {
                Self::validate_inline(data, mime_type)?;
                Ok(format!("data:{mime_type};base64,{data}"))
            }
        }
    }

    /// Stable key for mock-vector lookup: the full data URI for inline inputs.
    pub(crate) fn mock_key(&self) -> Result<String, EmbeddingError> {
        match self {
            ImageInput::Url(url) => Ok(url.clone()),
            ImageInput::DataUri(uri) => Ok(uri.clone()),
            ImageInput::Base64 { data, mime_type } => Ok(format!("data:{mime_type};base64,{data}")),
        }
    }

    fn validate_inline(data: &str, mime: &str) -> Result<(), EmbeddingError> {
        if data.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        if !mime.starts_with("image/") {
            return Err(EmbeddingError::Config(format!(
                "vision embeddings require an image/* MIME type, got {mime:?}"
            )));
        }
        Ok(())
    }
}

/// Splits `data:<mime>;base64,<data>` into `(mime, raw base64)`.
pub(crate) fn parse_data_uri(uri: &str) -> Result<(&str, &str), EmbeddingError> {
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| EmbeddingError::Config(format!("not a data URI: {uri}")))?;
    let comma = rest.find(',').ok_or_else(|| {
        EmbeddingError::Config(format!("malformed data URI (missing comma): {uri}"))
    })?;
    let meta = &rest[..comma];
    let data = &rest[comma + 1..];
    if !meta.contains("base64") {
        return Err(EmbeddingError::Config(
            "data URI must carry base64-encoded image bytes".to_string(),
        ));
    }
    let mime = meta
        .split(';')
        .next()
        .map(str::trim)
        .filter(|m| m.starts_with("image/") && !m.is_empty())
        .ok_or_else(|| {
            EmbeddingError::Config(format!("data URI is missing an image/* MIME type: {uri}"))
        })?;
    if data.trim().is_empty() {
        return Err(EmbeddingError::EmptyInput);
    }
    Ok((mime, data))
}

/// Cross-modal embedding model: images and text mapped to one shared space.
///
/// # Contract
///
/// - `embed_text` and `embed_image` return vectors of identical dimension
///   ([`VisionEmbeddings::dimension`]) that are directly comparable
///   (cosine similarity) across modalities;
/// - all vectors are L2-normalized;
/// - empty/whitespace text or image payloads raise [`EmbeddingError::EmptyInput`];
/// - an empty image slice is `Ok(vec![])` (nothing to do is not an error).
#[async_trait]
pub trait VisionEmbeddings: Send + Sync {
    /// Embeds a single image.
    async fn embed_image(&self, image: &ImageInput) -> Result<Vec<f32>, EmbeddingError>;

    /// Embeds multiple images. The default loops one-by-one; HTTP backends
    /// override it with a single batched request and must enforce batch
    /// alignment (requested count == returned count).
    async fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(images.len());
        for image in images {
            out.push(self.embed_image(image).await?);
        }
        Ok(out)
    }

    /// Embeds text **into the same vector space as the images**.
    ///
    /// Use this (not a text-only [`crate::Embeddings`] model) when building
    /// text-to-image retrieval: the query vector and the stored image
    /// vectors must come from one model family.
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embedding dimension (identical for both modalities).
    fn dimension(&self) -> usize;

    /// Model name.
    fn model_name(&self) -> &str;
}

/// Deterministic in-memory [`VisionEmbeddings`] for offline tests.
///
/// Without overrides, vectors are derived deterministically from the input
/// key (same key → same normalized vector; different keys → different
/// vectors). Tests demonstrating cross-modal retrieval register aligned
/// vectors explicitly via [`MockVisionEmbeddings::with_text_vector`] and
/// [`MockVisionEmbeddings::with_image_vector`].
#[derive(Debug)]
pub struct MockVisionEmbeddings {
    model: String,
    dimension: usize,
    text_vectors: Mutex<HashMap<String, Vec<f32>>>,
    image_vectors: Mutex<HashMap<String, Vec<f32>>>,
}

impl MockVisionEmbeddings {
    /// Creates a mock producing `dimension`-wide vectors.
    pub fn new(dimension: usize) -> Self {
        Self {
            model: "mock-vision-embeddings".to_string(),
            dimension,
            text_vectors: Mutex::new(HashMap::new()),
            image_vectors: Mutex::new(HashMap::new()),
        }
    }

    /// Overrides the vector returned for a given text query/label.
    ///
    /// The vector is L2-normalized before being stored.
    pub fn with_text_vector(&self, text: impl Into<String>, mut vector: Vec<f32>) -> &Self {
        l2_normalize(&mut vector);
        self.text_vectors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(text.into(), vector);
        self
    }

    /// Overrides the vector returned for an image, keyed exactly by its URL,
    /// data URI, or the synthesized `data:<mime>;base64,<data>` reference.
    ///
    /// The vector is L2-normalized before being stored.
    pub fn with_image_vector(&self, image: &ImageInput, mut vector: Vec<f32>) -> &Self {
        if let Ok(key) = image.mock_key() {
            l2_normalize(&mut vector);
            self.image_vectors
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key, vector);
        }
        self
    }
}

#[async_trait]
impl VisionEmbeddings for MockVisionEmbeddings {
    async fn embed_image(&self, image: &ImageInput) -> Result<Vec<f32>, EmbeddingError> {
        // Validate through the same path as real backends: empty URLs/data
        // payloads must raise EmptyInput before mock-key lookup.
        let _ = image.reference()?;
        let key = image.mock_key()?;
        let map = self.image_vectors.lock().unwrap_or_else(|e| e.into_inner());
        Ok(map
            .get(&key)
            .cloned()
            .unwrap_or_else(|| deterministic_vector(&key, self.dimension)))
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let map = self.text_vectors.lock().unwrap_or_else(|e| e.into_inner());
        Ok(map
            .get(text)
            .cloned()
            .unwrap_or_else(|| deterministic_vector(text, self.dimension)))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Stable pseudo-random normalized vector derived from a string key.
pub(crate) fn deterministic_vector(key: &str, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let mut state = hasher.finish();
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        // LCG over u64, projected into [-1, 1).
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 33) as f32 / (1u64 << 30) as f32 - 1.0;
        vector.push(unit);
    }
    l2_normalize(&mut vector);
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_data_uri_into_mime_and_bytes() {
        let (mime, data) = parse_data_uri("data:image/png;base64,aW1n").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(data, "aW1n");
    }

    #[test]
    fn rejects_non_image_and_malformed_data_uris() {
        assert!(parse_data_uri("data:application/pdf;base64,ZG9j").is_err());
        assert!(parse_data_uri("image/png;base64,aW1n").is_err());
        assert!(parse_data_uri("data:image/png,aW1n").is_err());
        assert!(parse_data_uri("data:image/png;base64,").is_err());
    }

    #[test]
    fn inline_parts_rejects_plain_url_but_accepts_base64() {
        assert!(ImageInput::from_url("https://example.com/a.png")
            .inline_parts()
            .is_err());
        let (data, mime) = ImageInput::from_base64("aW1n", "image/png")
            .inline_parts()
            .unwrap();
        assert_eq!(data, "aW1n");
        assert_eq!(mime, "image/png");
    }

    #[test]
    fn reference_renders_base64_as_data_uri() {
        let reference = ImageInput::from_base64("aW1n", "image/png")
            .reference()
            .unwrap();
        assert_eq!(reference, "data:image/png;base64,aW1n");
        assert_eq!(
            ImageInput::from_url("https://example.com/a.png")
                .reference()
                .unwrap(),
            "https://example.com/a.png"
        );
    }

    #[tokio::test]
    async fn mock_is_deterministic_and_normalized() {
        let mock = MockVisionEmbeddings::new(16);
        let a = mock.embed_text("cat").await.unwrap();
        let b = mock.embed_text("cat").await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);

        let img = ImageInput::from_url("https://example.com/cat.png");
        let va = mock.embed_image(&img).await.unwrap();
        let vb = mock.embed_image(&img).await.unwrap();
        assert_eq!(va, vb);
        assert_ne!(va.len(), 0);
    }

    #[tokio::test]
    async fn mock_overrides_align_text_and_image_in_shared_space() {
        let mock = MockVisionEmbeddings::new(4);
        mock.with_text_vector("cat", vec![1.0, 0.0, 0.0, 0.0]);
        let img = ImageInput::from_url("https://example.com/cat.png");
        mock.with_image_vector(&img, vec![1.0, 0.0, 0.0, 0.0]);

        let text_v = mock.embed_text("cat").await.unwrap();
        let image_v = mock.embed_image(&img).await.unwrap();
        let sim = crate::cosine_similarity(&text_v, &image_v).unwrap();
        assert!((sim - 1.0).abs() < 1e-5, "aligned pair sim = {sim}");
    }

    #[tokio::test]
    async fn empty_inputs_are_rejected() {
        let mock = MockVisionEmbeddings::new(4);
        assert!(matches!(
            mock.embed_text("  ").await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert!(matches!(
            mock.embed_image(&ImageInput::from_base64(" ", "image/png"))
                .await,
            Err(EmbeddingError::EmptyInput)
        ));
        assert_eq!(
            mock.embed_images(&[]).await.unwrap(),
            Vec::<Vec<f32>>::new()
        );
    }
}
