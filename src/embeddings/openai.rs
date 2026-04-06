// src/embeddings/openai.rs
//! OpenAI Embeddings 实现
//!
//! 使用 OpenAI 的 text-embedding-ada-002 或其他嵌入模型。

use super::{Embeddings, EmbeddingError};
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
    
    /// 从环境变量创建
    pub fn from_env() -> Self {
        Self::new(OpenAIEmbeddingsConfig::default())
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
        
        let embedding_response: OpenAIEmbeddingResponse = response
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
        
        let body = serde_json::json!({
            "model": self.config.model,
            "input": texts,
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
        
        let embedding_response: OpenAIEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;
        
        // 按索引排序
        let mut results = vec![Vec::new(); texts.len()];
        for item in embedding_response.data {
            results[item.index as usize] = item.embedding;
        }
        
        Ok(results)
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
            api_key: "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6".to_string(),
            base_url: "https://api.openai-proxy.org/v1".to_string(),
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