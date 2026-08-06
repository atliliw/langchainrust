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
    pub api_key: String,
    pub base_url: String,
    pub model: String,
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
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| "COHERE_API_KEY environment variable not set".to_string())?;
        let base_url =
            std::env::var("COHERE_BASE_URL").unwrap_or_else(|_| COHERE_EMBED_BASE_URL.to_string());
        let model = std::env::var("COHERE_EMBED_MODEL")
            .unwrap_or_else(|_| COHERE_EMBED_MODEL.to_string());
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
    pub fn new(config: CohereEmbeddingsConfig) -> Self {
        let dimension = 1024;
        Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        }
    }

    /// Creates from environment variables.
    pub fn from_env_result() -> Result<Self, String> {
        Ok(Self::new(CohereEmbeddingsConfig::from_env_result()?))
    }
}

#[async_trait]
impl Embeddings for CohereEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!("{}/embed", self.config.base_url);
        let body = json!({
            "model": self.config.model,
            "input_type": self.config.input_type.as_str(),
            "texts": [text],
            "embedding_types": ["float"],
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

        let embed_response: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        embed_response
            .data
            .first()
            .map(|d| d.embedding.clone())
            .ok_or_else(|| EmbeddingError::ApiError("No embedding in response".to_string()))
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embed", self.config.base_url);
        let body = json!({
            "model": self.config.model,
            "input_type": CohereEmbedInputType::SearchDocument.as_str(),
            "texts": texts,
            "embedding_types": ["float"],
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

        let embed_response: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        Ok(embed_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect())
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
        assert_eq!(CohereEmbedInputType::Classification.as_str(), "classification");
        assert_eq!(CohereEmbedInputType::Clustering.as_str(), "clustering");
    }

    #[test]
    fn test_embeddings_new() {
        let config = CohereEmbeddingsConfig::new("key");
        let embeddings = CohereEmbeddings::new(config);
        assert_eq!(embeddings.model_name(), COHERE_EMBED_MODEL);
        assert_eq!(embeddings.dimension(), 1024);
    }

    #[test]
    fn test_embeddings_multilingual_dimension() {
        let config = CohereEmbeddingsConfig::new("key").with_model("embed-multilingual-v3.0");
        let embeddings = CohereEmbeddings::new(config);
        assert_eq!(embeddings.dimension(), 1024);
    }
}
