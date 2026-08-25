// lc-embeddings/src/deepseek.rs
//! DeepSeek embeddings implementation.
//!
//! DeepSeek 走 OpenAI 兼容 `/embeddings` 协议，复用
//! [`crate::openai_compat`] 公共基类（P1-5），本文件只配置规格
//! （URL / 模型 / 维度 / 批量大小）。

use crate::openai_compat::{CompatConfigAccess, CompatSpec, OpenAICompatEmbeddings};
use crate::EmbeddingError;

/// Default base URL for the DeepSeek API.
pub const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

/// Default embedding model for DeepSeek.
pub const DEEPSEEK_EMBED_MODEL: &str = "deepseek-embedding";

/// Configuration for DeepSeek embeddings API.
#[derive(Debug, Clone)]
pub struct DeepSeekEmbeddingsConfig {
    /// DeepSeek API key.
    pub api_key: String,
    /// Base URL for the DeepSeek embeddings API.
    pub base_url: String,
    /// Embedding model name.
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

    /// Creates a DeepSeekEmbeddingsConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `DEEPSEEK_API_KEY`: API key (required)
    /// - `DEEPSEEK_BASE_URL`: API endpoint (optional)
    /// - `DEEPSEEK_EMBED_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
            EmbeddingError::Config("DEEPSEEK_API_KEY environment variable not set".to_string())
        })?;
        let base_url =
            std::env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| DEEPSEEK_BASE_URL.to_string());
        let model = std::env::var("DEEPSEEK_EMBED_MODEL")
            .unwrap_or_else(|_| DEEPSEEK_EMBED_MODEL.to_string());
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

impl CompatConfigAccess for DeepSeekEmbeddingsConfig {
    fn api_key(&self) -> &str {
        &self.api_key
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
}

impl CompatSpec for DeepSeekEmbeddingsConfig {
    fn api_key_env() -> &'static str {
        "DEEPSEEK_API_KEY"
    }
    fn batch_size() -> usize {
        64
    }
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        if model == DEEPSEEK_EMBED_MODEL {
            Ok(1536)
        } else {
            Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for DeepSeek model '{model}' (supported: '{DEEPSEEK_EMBED_MODEL}')"
            )))
        }
    }
    fn from_env_result() -> Result<Self, EmbeddingError> {
        Self::from_env_result()
    }
}

/// DeepSeek embeddings client for generating vector embeddings.
///
/// 复用 OpenAI 兼容协议公共基类（P1-5）：构造时 fail fast 校验 API key 非空
/// 与模型维度已知（P1-2/P1-3），批量对齐显式报错（P0-1），错误体不吞错（P1-4）。
pub type DeepSeekEmbeddings = OpenAICompatEmbeddings<DeepSeekEmbeddingsConfig>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{spawn_embeddings_stub, spawn_status_stub};
    use crate::Embeddings;
    use std::env;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// P0-1: 服务端少返回 → 显式 `EmptyVectorInBatch`，而非静默空向量。
    #[tokio::test]
    async fn test_embed_documents_truncated_errors() {
        let base_url = spawn_embeddings_stub(Arc::new(|n| n.saturating_sub(1))).await;
        let config = DeepSeekEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: DEEPSEEK_EMBED_MODEL.into(),
        };
        let embeddings = DeepSeekEmbeddings::new(config).unwrap();

        let result = embeddings.embed_documents(&["a", "b"]).await;
        assert!(
            matches!(result, Err(EmbeddingError::EmptyVectorInBatch)),
            "truncated response should report EmptyVectorInBatch, got: {:?}",
            result
        );
    }

    /// P2-5: DeepSeek（复用 OpenAI 兼容基类）同样接线 429 重试。
    #[tokio::test]
    async fn test_embed_query_retries_on_429() {
        let success_body = r#"{"data":[{"embedding":[0.6,0.8],"index":0}],"model":"stub","usage":{"prompt_tokens":0,"total_tokens":0}}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let config = DeepSeekEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: DEEPSEEK_EMBED_MODEL.into(),
        };
        let embeddings = DeepSeekEmbeddings::new(config).unwrap();

        let v = embeddings
            .embed_query("hello")
            .await
            .expect("should retry successfully after two 429s");
        assert_eq!(v.len(), 2);
        // P2-8: 返回向量应已归一化。
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
        assert_eq!(requests.load(Ordering::SeqCst), 3, "1 initial + 2 retries");
    }

    /// P1-3: API key 为空 → 构造期 fail fast 报 `Config`，而非拖到发请求才 401。
    #[test]
    fn test_new_rejects_empty_api_key() {
        let config = DeepSeekEmbeddingsConfig {
            api_key: String::new(),
            base_url: DEEPSEEK_BASE_URL.into(),
            model: DEEPSEEK_EMBED_MODEL.into(),
        };
        let err = DeepSeekEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// P1-2: 未知模型 → 构造期报错，不得回落默认 1536 撒谎。
    #[test]
    fn test_new_rejects_unknown_model() {
        let config = DeepSeekEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url: DEEPSEEK_BASE_URL.into(),
            model: "some-unknown-model".into(),
        };
        let err = DeepSeekEmbeddings::new(config).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

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
        let old = save_and_set("DEEPSEEK_API_KEY", "test-key-123");
        let result = DeepSeekEmbeddingsConfig::from_env_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().api_key, "test-key-123");
        restore("DEEPSEEK_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("DEEPSEEK_API_KEY").ok();
        env::remove_var("DEEPSEEK_API_KEY");
        let result = DeepSeekEmbeddingsConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("DEEPSEEK_API_KEY"));
        restore("DEEPSEEK_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_uses_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("DEEPSEEK_API_KEY", "key");
        let old_url = save_and_set("DEEPSEEK_BASE_URL", "https://custom.api.com");
        let old_model = save_and_set("DEEPSEEK_EMBED_MODEL", "custom-model");
        let config = DeepSeekEmbeddingsConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, "https://custom.api.com");
        assert_eq!(config.model, "custom-model");
        restore("DEEPSEEK_API_KEY", old_key);
        restore("DEEPSEEK_BASE_URL", old_url);
        restore("DEEPSEEK_EMBED_MODEL", old_model);
    }

    #[test]
    fn test_from_env_result_uses_defaults_for_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_key = save_and_set("DEEPSEEK_API_KEY", "key");
        let old_url = env::var("DEEPSEEK_BASE_URL").ok();
        env::remove_var("DEEPSEEK_BASE_URL");
        let old_model = env::var("DEEPSEEK_EMBED_MODEL").ok();
        env::remove_var("DEEPSEEK_EMBED_MODEL");
        let config = DeepSeekEmbeddingsConfig::from_env_result().unwrap();
        assert_eq!(config.base_url, DEEPSEEK_BASE_URL.to_string());
        assert_eq!(config.model, DEEPSEEK_EMBED_MODEL);
        restore("DEEPSEEK_API_KEY", old_key);
        restore("DEEPSEEK_BASE_URL", old_url);
        restore("DEEPSEEK_EMBED_MODEL", old_model);
    }

    #[test]
    fn test_embeddings_from_env_result_ok() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("DEEPSEEK_API_KEY", "test-key");
        assert!(DeepSeekEmbeddings::from_env_result().is_ok());
        restore("DEEPSEEK_API_KEY", old);
    }

    #[test]
    fn test_embeddings_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("DEEPSEEK_API_KEY").ok();
        env::remove_var("DEEPSEEK_API_KEY");
        assert!(DeepSeekEmbeddings::from_env_result().is_err());
        restore("DEEPSEEK_API_KEY", old);
    }
}
