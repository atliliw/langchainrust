// src/embeddings/qwen.rs
//! Qwen (Alibaba Cloud) embeddings implementation.

use crate::embeddings::{Embeddings, EmbeddingError};
use crate::language_models::providers::QWEN_BASE_URL;
use async_trait::async_trait;
use serde::Deserialize;

/// Default embedding model for Qwen.
pub const QWEN_EMBED_MODEL: &str = "text-embedding-v1";

/// Configuration for Qwen embeddings API.
#[derive(Debug, Clone)]
pub struct QwenEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Default for QwenEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("QWEN_API_KEY").unwrap_or_default(),
            base_url: QWEN_BASE_URL.to_string(),
            model: QWEN_EMBED_MODEL.to_string(),
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

    /// Creates a QwenEmbeddingsConfig from environment variables.
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Sets the embedding model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// Qwen embeddings client for generating vector embeddings.
pub struct QwenEmbeddings {
    config: QwenEmbeddingsConfig,
    client: reqwest::Client,
}

impl QwenEmbeddings {
    /// Creates a QwenEmbeddings with the given configuration.
    pub fn new(config: QwenEmbeddingsConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates a QwenEmbeddings from environment variables.
    pub fn from_env() -> Self {
        Self::new(QwenEmbeddingsConfig::from_env())
    }
}

#[async_trait]
impl Embeddings for QwenEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url);

        let body = serde_json::json!({
            "model": self.config.model,
            "input": text,
        });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ApiError(format!("HTTP {}: {}", status, error_text)));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        Ok(embedding_response.data[0].embedding.clone())
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.config.base_url);
        let batch_size = 64;
        let mut all_results = vec![Vec::new(); texts.len()];
        let mut offset = 0;

        for chunk in texts.chunks(batch_size) {
            let body = serde_json::json!({
                "model": self.config.model,
                "input": chunk,
            });

            let response = self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(EmbeddingError::ApiError(format!("HTTP {}: {}", status, error_text)));
            }

            let embedding_response: EmbeddingResponse = response
                .json()
                .await
                .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

            for item in embedding_response.data {
                let global_index = offset + item.index as usize;
                if global_index < all_results.len() {
                    all_results[global_index] = item.embedding;
                }
            }
            offset += chunk.len();
        }

        Ok(all_results)
    }

    fn dimension(&self) -> usize {
        1536
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: i32,
}