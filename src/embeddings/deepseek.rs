// src/embeddings/deepseek.rs
//! DeepSeek embeddings implementation.

use crate::embeddings::{Embeddings, EmbeddingError};
use crate::language_models::providers::DEEPSEEK_BASE_URL;
use async_trait::async_trait;
use serde::Deserialize;

/// Default embedding model for DeepSeek.
pub const DEEPSEEK_EMBED_MODEL: &str = "deepseek-embedding";

/// Configuration for DeepSeek embeddings API.
#[derive(Debug, Clone)]
pub struct DeepSeekEmbeddingsConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Default for DeepSeekEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
            base_url: DEEPSEEK_BASE_URL.to_string(),
            model: DEEPSEEK_EMBED_MODEL.to_string(),
        }
    }
}

impl DeepSeekEmbeddingsConfig {
    /// Creates a new DeepSeekEmbeddingsConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a DeepSeekEmbeddingsConfig from environment variables.
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Sets the embedding model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// DeepSeek embeddings client for generating vector embeddings.
pub struct DeepSeekEmbeddings {
    config: DeepSeekEmbeddingsConfig,
    client: reqwest::Client,
}

impl DeepSeekEmbeddings {
    /// Creates a DeepSeekEmbeddings with the given configuration.
    pub fn new(config: DeepSeekEmbeddingsConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates a DeepSeekEmbeddings from environment variables.
    pub fn from_env() -> Self {
        Self::new(DeepSeekEmbeddingsConfig::from_env())
    }
}

#[async_trait]
impl Embeddings for DeepSeekEmbeddings {
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
struct EmbeddingData {
    embedding: Vec<f32>,
    index: i32,
}