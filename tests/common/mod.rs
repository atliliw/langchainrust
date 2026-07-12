//! 测试公共配置
//!
//! API Key 配置方式：
//! 1. 环境变量: export OPENAI_API_KEY="your-key"
//! 2. 直接修改下方 API_KEY 常量
//!
//! MongoDB 配置方式：
//! 1. 环境变量: export MONGO_URI="mongodb://localhost:27017"
//! 2. 直接修改下方 MONGO_URI 常量

#![allow(dead_code)]

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
// 🗄️ MongoDB 配置 - 修改这里或设置环境变量
// ============================================================================

/// MongoDB 连接 URI
///
/// 示例:
/// - 本地: "mongodb://localhost:27017"
/// - Atlas: "mongodb+srv://username:password@cluster.mongodb.net"
const MONGO_URI: &str = "";

/// MongoDB 数据库名称
const MONGO_DATABASE: &str = "langgraph_test";

/// MongoDB 集合名称
const MONGO_COLLECTION: &str = "graph_definitions";

// ============================================================================

static CONFIG: OnceLock<TestConfig> = OnceLock::new();

#[cfg(feature = "mongodb-persistence")]
static MONGO_CONFIG: OnceLock<MongoTestConfig> = OnceLock::new();

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

/// MongoDB 测试配置
#[cfg(feature = "mongodb-persistence")]
pub struct MongoTestConfig {
    pub uri: String,
    pub database: String,
    pub collection: String,
}

#[cfg(feature = "mongodb-persistence")]
impl MongoTestConfig {
    pub fn get() -> &'static Self {
        MONGO_CONFIG.get_or_init(|| {
            let uri = if MONGO_URI.is_empty() {
                std::env::var("MONGO_URI")
                    .expect("请设置 MONGO_URI 环境变量，或在 tests/common/mod.rs 中配置 MONGO_URI")
            } else {
                MONGO_URI.to_string()
            };

            let database =
                std::env::var("MONGO_DATABASE").unwrap_or_else(|_| MONGO_DATABASE.to_string());

            let collection =
                std::env::var("MONGO_COLLECTION").unwrap_or_else(|_| MONGO_COLLECTION.to_string());

            MongoTestConfig {
                uri,
                database,
                collection,
            }
        })
    }

    pub fn to_mongo_config(&self) -> langchainrust::langgraph::MongoConfig {
        langchainrust::langgraph::MongoConfig::new(
            self.uri.clone(),
            self.database.clone(),
            self.collection.clone(),
        )
    }
}
