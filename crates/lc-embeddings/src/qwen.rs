// lc-embeddings/src/qwen.rs
//! Qwen (Alibaba Cloud) embeddings implementation.

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use serde::Deserialize;

/// Default base URL for the Qwen (DashScope) API.
pub const QWEN_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

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
    #[deprecated(
        since = "0.7.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::default())
    }

    /// Creates a QwenEmbeddingsConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `QWEN_API_KEY`: API key (required)
    /// - `QWEN_BASE_URL`: API endpoint (optional)
    /// - `QWEN_EMBED_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = std::env::var("QWEN_API_KEY")
            .map_err(|_| "QWEN_API_KEY environment variable not set".to_string())?;
        let base_url = std::env::var("QWEN_BASE_URL").unwrap_or_else(|_| QWEN_BASE_URL.to_string());
        let model =
            std::env::var("QWEN_EMBED_MODEL").unwrap_or_else(|_| QWEN_EMBED_MODEL.to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
        })
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
    #[deprecated(
        since = "0.7.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::new(QwenEmbeddingsConfig::default()))
    }

    /// Creates a QwenEmbeddings from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, String> {
        let config = QwenEmbeddingsConfig::from_env_result()?;
        Ok(Self::new(config))
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

        let response = self
            .client
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
            return Err(EmbeddingError::ApiError(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        Ok(embedding_response
            .data
            .first()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding data in response".to_string()))?
            .embedding
            .clone())
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

            let response = self
                .client
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
struct EmbeddingData {
    embedding: Vec<f32>,
    index: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

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
        assert!(result.unwrap_err().contains("QWEN_API_KEY"));
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
