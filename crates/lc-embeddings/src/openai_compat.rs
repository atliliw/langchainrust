// lc-embeddings/src/openai_compat.rs
//! Shared base class for OpenAI-compatible-protocol embedding clients (P1-5).
//!
//! DeepSeek and Qwen speak the same OpenAI `/embeddings` protocol (same request body,
//! same `data[index]` alignment, same Bearer auth), and their sources were almost
//! line-for-line duplicates. This module extracts the common implementation; DeepSeek/Qwen
//! only configure URL / model / dimension / batch size via [`CompatSpec`].

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use serde::Deserialize;

/// Abstraction over provider config fields — DeepSeek/Qwen config structs share field names.
pub trait CompatConfigAccess {
    /// Returns the API key
    fn api_key(&self) -> &str;
    /// Returns the Base URL
    fn base_url(&self) -> &str;
    /// Returns the model name
    fn model(&self) -> &str;
}

/// Static specification for an OpenAI-compatible-protocol provider.
///
/// Implementing this trait grants the full embedding capability provided by
/// `OpenAICompatEmbeddings`; it is the extension point for new OpenAI-compatible providers.
pub trait CompatSpec: CompatConfigAccess + Sized + Default {
    /// Environment variable name: API key (used in construction-time error messages, P1-3).
    fn api_key_env() -> &'static str;
    /// The batch limit for a single request.
    fn batch_size() -> usize;
    /// Vector dimension for a given model; unknown models must error (P1-2), never lying with a default.
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError>;
    /// Constructs config from environment variables (reuses each config's from_env_result).
    fn from_env_result() -> Result<Self, EmbeddingError>;
}

/// Generic OpenAI-compatible embedding client (P1-5).
///
/// Providers speaking the OpenAI `/embeddings` protocol (DeepSeek/Qwen, etc.) reuse this
/// implementation via the [`CompatSpec`] config spec; `C` is each provider's config type.
pub struct OpenAICompatEmbeddings<C: CompatConfigAccess + CompatSpec> {
    config: C,
    client: reqwest::Client,
    dimension: usize,
}

impl<C: CompatConfigAccess + CompatSpec> std::fmt::Debug for OpenAICompatEmbeddings<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatEmbeddings")
            .field("model", &self.config.model())
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl<C: CompatConfigAccess + CompatSpec> OpenAICompatEmbeddings<C> {
    /// Fails fast at construction (P1-3): an empty API key errors immediately instead of
    /// waiting until the request to 401; also validates the model dimension is known (P1-2).
    pub fn new(config: C) -> Result<Self, EmbeddingError> {
        if config.api_key().trim().is_empty() {
            return Err(EmbeddingError::Config(format!(
                "{} is empty",
                C::api_key_env()
            )));
        }
        let dimension = C::dimension_for(config.model())?;
        Ok(Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        })
    }

    /// Creates from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let config = C::from_env_result()?;
        Self::new(config)
    }
}

#[async_trait]
impl<C: CompatConfigAccess + CompatSpec + Send + Sync> Embeddings for OpenAICompatEmbeddings<C> {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url());

        let body = serde_json::json!({
            "model": self.config.model(),
            "input": text,
        });

        // P2-5: exponential backoff retry on 429/5xx.
        let response = crate::retry::post_json_with_retry(
            &self.client,
            &url,
            self.config.api_key(),
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

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let mut embedding = embedding_response
            .data
            .first()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding data in response".to_string()))?
            .embedding
            .clone();
        // P2-8: uniform L2 normalization, guaranteeing unit length.
        crate::l2_normalize(&mut embedding);
        Ok(embedding)
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // P1-1: an empty slice is not an error (nothing to do); only empty/all-whitespace texts error.
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url());
        let batch_size = C::batch_size().max(1);
        // P0-1: collect item-by-item into Option slots, rejecting silent empty vectors. A chunk
        // returning fewer/misaligned entries leaves None slots that error at the end, instead of
        // producing zero vectors downstream treats as "dissimilar".
        let mut all_results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut offset = 0;

        for chunk in texts.chunks(batch_size) {
            let body = serde_json::json!({
                "model": self.config.model(),
                "input": chunk,
            });

            // P2-5: exponential backoff retry on 429/5xx.
            let response = crate::retry::post_json_with_retry(
                &self.client,
                &url,
                self.config.api_key(),
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

            let embedding_response: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

            for item in embedding_response.data {
                let global_index = offset + item.index as usize;
                if global_index >= all_results.len() {
                    // Provider index beyond the requested range = batch misalignment; error out.
                    return Err(EmbeddingError::BatchMismatch {
                        expected: all_results.len(),
                        actual: global_index + 1,
                    });
                }
                all_results[global_index] = Some(item.embedding);
            }
            offset += chunk.len();
        }

        // Unwrap into Result: any empty slot errors explicitly rather than leaving a zero vector; then apply uniform L2 normalization (P2-8).
        all_results
            .into_iter()
            .map(|opt| {
                let mut v = opt.ok_or(EmbeddingError::EmptyVectorInBatch)?;
                crate::l2_normalize(&mut v);
                Ok(v)
            })
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        self.config.model()
    }
}

/// Embedding response body for the OpenAI-compatible protocol (shared by DeepSeek/Qwen).
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: i32,
}
