//! 测试公共配置
//!
//! API Key 配置方式：
//! 1. 环境变量: export OPENAI_API_KEY="your-key"
//! 2. 直接修改下方 API_KEY 常量

use langchainrust::{OpenAIChat, OpenAIConfig, OpenAIEmbeddings, OpenAIEmbeddingsConfig};
use std::sync::OnceLock;

// ============================================================================
// 🔑 在这里配置你的 API Key
// ============================================================================

/// API Key - 修改这里或设置环境变量 OPENAI_API_KEY
const API_KEY: &str = "sk-l0YYMX65mCYRlTJYH0ptf4BFpqJwm8Xo9Z5IMqSZD0yOafl6";

/// Base URL - 可选修改
const BASE_URL: &str = "https://api.openai-proxy.org/v1";

/// 默认模型
const DEFAULT_MODEL: &str = "gpt-3.5-turbo";

/// Embedding 模型
const EMBEDDING_MODEL: &str = "text-embedding-ada-002";

// ============================================================================

static CONFIG: OnceLock<TestConfig> = OnceLock::new();

pub struct TestConfig {
    pub api_key: String,
    pub base_url: String,
}

impl TestConfig {
    pub fn get() -> &'static Self {
        CONFIG.get_or_init(|| {
            let api_key = if API_KEY.is_empty() {
                std::env::var("OPENAI_API_KEY").expect(
                    "请设置 OPENAI_API_KEY 环境变量，或在 tests/common/mod.rs 中配置 API_KEY",
                )
            } else {
                API_KEY.to_string()
            };

            let base_url =
                std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| BASE_URL.to_string());

            TestConfig { api_key, base_url }
        })
    }

    pub fn openai_chat_config(&self) -> OpenAIConfig {
        OpenAIConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: DEFAULT_MODEL.to_string(),
            streaming: false,
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        }
    }

    pub fn openai_chat(&self) -> OpenAIChat {
        OpenAIChat::new(self.openai_chat_config())
    }

    pub fn embeddings_config(&self) -> OpenAIEmbeddingsConfig {
        OpenAIEmbeddingsConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            model: EMBEDDING_MODEL.to_string(),
            ..Default::default()
        }
    }

    pub fn embeddings(&self) -> OpenAIEmbeddings {
        OpenAIEmbeddings::new(self.embeddings_config())
    }
}
