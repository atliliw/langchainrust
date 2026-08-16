// lc-embeddings/src/openai_compat.rs
//! OpenAI 兼容协议 embedding 客户端公共基类（P1-5）。
//!
//! DeepSeek 与 Qwen 走同一套 OpenAI `/embeddings` 协议（同样的请求体、同样的
//! `data[index]` 对齐、同样的 Bearer 认证），二者源码几乎逐行重复。本模块抽出
//! 通用实现，DeepSeek/Qwen 仅通过 [`CompatSpec`] 配置 URL / 模型 / 维度 / 批量大小。

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use serde::Deserialize;

/// 访问 provider 配置字段的抽象——DeepSeek/Qwen 的 config 结构体字段名相同。
pub trait CompatConfigAccess {
    fn api_key(&self) -> &str;
    fn base_url(&self) -> &str;
    fn model(&self) -> &str;
}

/// OpenAI 兼容协议 provider 的静态规格。
///
/// 实现该 trait 即可获得 `OpenAICompatEmbeddings` 提供的完整 embedding 能力，
/// 是接入新增 OpenAI 兼容 provider 的扩展点。
pub trait CompatSpec: CompatConfigAccess + Sized + Default {
    /// 环境变量名：API key（用于构造期错误信息，P1-3）。
    fn api_key_env() -> &'static str;
    /// 单次请求的批量上限。
    fn batch_size() -> usize;
    /// 给定模型的向量维度；未知模型必须报错（P1-2），不得回落默认值撒谎。
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError>;
    /// 从环境变量构造 config（复用各 config 已实现的 from_env_result）。
    fn from_env_result() -> Result<Self, String>;
}

/// 通用 OpenAI 兼容 embedding 客户端（P1-5）。
///
/// DeepSeek/Qwen 等走 OpenAI `/embeddings` 协议的 provider 通过
/// [`CompatSpec`] 配置规格复用本实现；`C` 即各自的 config 类型。
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
    /// 构造时 fail fast（P1-3）：API key 为空立即报错，而不是拖到发请求才 401；
    /// 同时校验模型维度已知（P1-2）。
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
    pub fn from_env_result() -> Result<Self, String> {
        let config = C::from_env_result()?;
        Self::new(config).map_err(|e| e.to_string())
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

        // P2-5: 429/5xx 指数退避重试。
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
            // P1-4: 读失败的错误体也要报错，不能 unwrap_or_default() 吞掉。
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
        // P2-8: 统一 L2 归一化,保证单位长度。
        crate::l2_normalize(&mut embedding);
        Ok(embedding)
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // P1-1: 空切片不是错误（无事可做），含空/全空白文本才报错。
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url());
        let batch_size = C::batch_size().max(1);
        // P0-1: 用 Option 槽位逐项收集,拒绝静默空向量。某 chunk 少返回/错位会
        // 留下 None 槽位并在收尾时报错,而不是产出零向量被下游当成"不相似"。
        let mut all_results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut offset = 0;

        for chunk in texts.chunks(batch_size) {
            let body = serde_json::json!({
                "model": self.config.model(),
                "input": chunk,
            });

            // P2-5: 429/5xx 指数退避重试。
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
                // P1-4: 读失败的错误体也要报错，不能 unwrap_or_default() 吞掉。
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
                    // 服务端 index 超出请求范围 = 批次错位,直接报错。
                    return Err(EmbeddingError::BatchMismatch {
                        expected: all_results.len(),
                        actual: global_index + 1,
                    });
                }
                all_results[global_index] = Some(item.embedding);
            }
            offset += chunk.len();
        }

        // 展开为 Result:任一槽位空缺即显式报错,而非留下零向量;并统一 L2 归一化(P2-8)。
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

/// OpenAI 兼容协议的 embedding 响应体（DeepSeek/Qwen 共用）。
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: i32,
}
