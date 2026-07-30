// src/embeddings/openai.rs
//! OpenAI Embeddings 实现
//!
//! 使用 OpenAI 的 text-embedding-ada-002 或其他嵌入模型。

use super::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use serde::Deserialize;

/// OpenAI Embeddings 配置
#[derive(Debug, Clone)]
pub struct OpenAIEmbeddingsConfig {
    /// API 密钥
    pub api_key: String,

    /// API 基础 URL
    pub base_url: String,

    /// 模型名称（默认: text-embedding-ada-002）
    pub model: String,

    /// 批量大小（默认: 2048）
    pub batch_size: usize,
}

impl Default for OpenAIEmbeddingsConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-ada-002".to_string(),
            batch_size: 2048,
        }
    }
}

impl OpenAIEmbeddingsConfig {
    /// 创建新配置
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// 设置模型
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// 设置基础 URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// OpenAI Embeddings 客户端
pub struct OpenAIEmbeddings {
    config: OpenAIEmbeddingsConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl OpenAIEmbeddings {
    /// 创建新的 OpenAI Embeddings 客户端
    pub fn new(config: OpenAIEmbeddingsConfig) -> Self {
        // 根据模型确定维度
        let dimension = match config.model.as_str() {
            "text-embedding-ada-002" => 1536,
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            _ => 1536, // 默认维度
        };

        Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        }
    }

    /// Creates OpenAIEmbeddings from environment variables.
    #[deprecated(
        since = "0.7.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::new(OpenAIEmbeddingsConfig::default()))
    }

    /// Creates OpenAIEmbeddings from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY`: API key (required)
    /// - `OPENAI_BASE_URL`: API endpoint (optional)
    /// - `OPENAI_EMBED_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY environment variable not set".to_string())?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("OPENAI_EMBED_MODEL")
            .unwrap_or_else(|_| "text-embedding-ada-002".to_string());
        Ok(Self::new(OpenAIEmbeddingsConfig {
            api_key,
            base_url,
            model,
            batch_size: 2048,
        }))
    }
}

#[async_trait]
impl Embeddings for OpenAIEmbeddings {
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

        let embedding_response: OpenAIEmbeddingResponse = response
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
        let batch_size = self.config.batch_size.max(1);
        let mut all_results = vec![Vec::new(); texts.len()];
        let mut offset = 0;

        // 按 batch_size 分批调用 API
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

            let embedding_response: OpenAIEmbeddingResponse = response
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
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

/// OpenAI Embedding API 响应
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    model: String,
    usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: i32,
    object: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIEmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[cfg(test)]
mod tests_env {
    use super::*;
    use crate::ENV_TEST_LOCK;
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
        let _lock = crate::ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("OPENAI_API_KEY", "test-key-123");
        let result = OpenAIEmbeddings::from_env_result();
        assert!(result.is_ok());
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");
        let result = OpenAIEmbeddings::from_env_result();
        match result {
            Err(msg) => assert!(msg.contains("OPENAI_API_KEY")),
            Ok(_) => panic!("expected error when OPENAI_API_KEY is missing"),
        }
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_uses_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("OPENAI_API_KEY", "key");
        let old_url = save_and_set("OPENAI_BASE_URL", "https://custom.api.com/v1");
        let old_model = save_and_set("OPENAI_EMBED_MODEL", "text-embedding-3-small");
        let embeddings = OpenAIEmbeddings::from_env_result().unwrap();
        assert_eq!(embeddings.model_name(), "text-embedding-3-small");
        restore("OPENAI_API_KEY", old_key);
        restore("OPENAI_BASE_URL", old_url);
        restore("OPENAI_EMBED_MODEL", old_model);
    }

    #[test]
    fn test_from_env_result_uses_defaults_for_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("OPENAI_API_KEY", "key");
        let old_url = env::var("OPENAI_BASE_URL").ok();
        env::remove_var("OPENAI_BASE_URL");
        let old_model = env::var("OPENAI_EMBED_MODEL").ok();
        env::remove_var("OPENAI_EMBED_MODEL");
        let embeddings = OpenAIEmbeddings::from_env_result().unwrap();
        assert_eq!(embeddings.model_name(), "text-embedding-ada-002");
        restore("OPENAI_API_KEY", old_key);
        restore("OPENAI_BASE_URL", old_url);
        restore("OPENAI_EMBED_MODEL", old_model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenAIEmbeddingsConfig::default();
        assert_eq!(config.model, "text-embedding-ada-002");
        assert_eq!(config.batch_size, 2048);
    }

    #[test]
    fn test_config_builder() {
        let config = OpenAIEmbeddingsConfig::new("test-key")
            .with_model("text-embedding-3-large")
            .with_base_url("https://custom.api.com/v1");

        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.model, "text-embedding-3-large");
        assert_eq!(config.base_url, "https://custom.api.com/v1");
    }

    #[tokio::test]
    #[ignore = "需要真实 API 调用"]
    async fn test_real_embedding() {
        let config = OpenAIEmbeddingsConfig {
            api_key: "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string(),
            base_url:
                "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                    .to_string(),
            model: "text-embedding-ada-002".to_string(),
            batch_size: 2048,
        };

        let embeddings = OpenAIEmbeddings::new(config);

        let result = embeddings.embed_query("Hello, world!").await;
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 1536);
    }
}
