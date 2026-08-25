// lc-embeddings/src/openai.rs
//! OpenAI Embeddings implementation
//!
//! Uses OpenAI's text-embedding-ada-002 or other embedding models.

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;

/// P2-6: 批量切块后的并发上限——避免一次性打爆 provider 限流。
const MAX_CONCURRENT_CHUNKS: usize = 8;

/// OpenAI Embeddings configuration
#[derive(Debug, Clone)]
pub struct OpenAIEmbeddingsConfig {
    /// API key
    pub api_key: String,

    /// API base URL
    pub base_url: String,

    /// Model name (default: text-embedding-ada-002)
    pub model: String,

    /// Batch size (default: 2048)
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
    /// Create a new configuration
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

/// OpenAI Embeddings client
pub struct OpenAIEmbeddings {
    config: OpenAIEmbeddingsConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl OpenAIEmbeddings {
    /// Create a new OpenAI Embeddings client.
    ///
    /// 构造时 fail fast（P1-3）：API key 为空立即报错，而不是拖到发请求才 401。
    /// 模型维度已知才构造（P1-2）：未知模型返回 `Err`，不得回落默认 1536 撒谎。
    pub fn new(config: OpenAIEmbeddingsConfig) -> Result<Self, EmbeddingError> {
        if config.api_key.trim().is_empty() {
            return Err(EmbeddingError::Config(
                "OPENAI_API_KEY is empty".to_string(),
            ));
        }
        let dimension = Self::dimension_for(&config.model)?;

        Ok(Self {
            config,
            client: reqwest::Client::new(),
            dimension,
        })
    }

    /// 已知模型的 embedding 维度表；未知模型返回 `Err`（P1-2）。
    fn dimension_for(model: &str) -> Result<usize, EmbeddingError> {
        match model {
            "text-embedding-ada-002" => Ok(1536),
            "text-embedding-3-small" => Ok(1536),
            "text-embedding-3-large" => Ok(3072),
            other => Err(EmbeddingError::Config(format!(
                "unknown embedding dimension for OpenAI model '{other}' \
                 (supported: 'text-embedding-ada-002', 'text-embedding-3-small', \
                 'text-embedding-3-large')"
            ))),
        }
    }

    /// Creates OpenAIEmbeddings from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `OPENAI_API_KEY`: API key (required)
    /// - `OPENAI_BASE_URL`: API endpoint (optional)
    /// - `OPENAI_EMBED_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, EmbeddingError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            EmbeddingError::Config("OPENAI_API_KEY environment variable not set".to_string())
        })?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("OPENAI_EMBED_MODEL")
            .unwrap_or_else(|_| "text-embedding-ada-002".to_string());
        Self::new(OpenAIEmbeddingsConfig {
            api_key,
            base_url,
            model,
            batch_size: 2048,
        })
    }
}

#[async_trait]
impl Embeddings for OpenAIEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url);

        let body = serde_json::json!({
            "model": self.config.model,
            "input": text,
        });

        // P2-5: 429/5xx 指数退避重试,瞬时故障不再一次失败即抛错。
        let response = crate::retry::post_json_with_retry(
            &self.client,
            &url,
            &self.config.api_key,
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

        let embedding_response: OpenAIEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let mut embedding = embedding_response
            .data
            .first()
            .ok_or_else(|| EmbeddingError::ApiError("No embedding data in response".to_string()))?
            .embedding
            .clone();
        // P2-8: 统一 L2 归一化,保证单位长度,消除 provider 漂移。
        crate::l2_normalize(&mut embedding);
        Ok(embedding)
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // P1-1: 任一空/全空白文本都报错，与 trait 默认契约一致。
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let url = format!("{}/embeddings", self.config.base_url);
        let batch_size = self.config.batch_size.max(1);
        // P2-6: 各 chunk 并发请求(buffer_unordered + 并发上限),而非串行 await,
        // 提升大批量吞吐;并发上限避免一次性打爆 provider 限流。
        let concurrency = texts.len().div_ceil(batch_size).min(MAX_CONCURRENT_CHUNKS);

        // 每个 future 返回 (chunk_idx, data);并发完成顺序不定,由收集端按 chunk_idx
        // 放回全局槽位。P0-1: 任一槽位空缺即显式报错,绝不把缺失向量当成"不相似"。
        // 参考 faithfulness.rs 的并发写法:先把各 chunk 转成 owned Vec<String> 再
        // stream::iter,map 闭包输入无生命周期 → 闭包天然可泛化;async move 只捕获
        // owned chunk + Copy 引用(client/api_key/model/url),map(FnMut) 才编译得过。
        let chunks: Vec<(usize, Vec<String>)> = texts
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.iter().map(|s| s.to_string()).collect()))
            .collect();
        let client = &self.client;
        let api_key = self.config.api_key.as_str();
        let model = self.config.model.as_str();
        let url = url.as_str();
        let mut all_results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut stream = futures_util::stream::iter(chunks)
            .map(|(chunk_idx, chunk)| async move {
                let body = serde_json::json!({
                    "model": model,
                    "input": chunk,
                });
                // P2-5: 429/5xx 指数退避重试。
                let response = crate::retry::post_json_with_retry(
                    client,
                    url,
                    api_key,
                    &body,
                    &crate::retry::DEFAULT_RETRY,
                )
                .await
                .map_err(|e| EmbeddingError::HttpError(e.to_string()))?;

                let status = response.status();
                if !status.is_success() {
                    // P1-4: 读失败的错误体也要报错，不能 unwrap_or_default() 吞掉。
                    let error_text = response.text().await.map_err(|e| {
                        EmbeddingError::HttpError(format!(
                            "failed to read error response body: {e}"
                        ))
                    })?;
                    return Err(EmbeddingError::ApiError(format!(
                        "HTTP {}: {}",
                        status, error_text
                    )));
                }

                let embedding_response: OpenAIEmbeddingResponse = response
                    .json()
                    .await
                    .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

                Ok::<_, EmbeddingError>((chunk_idx, embedding_response.data))
            })
            .buffer_unordered(concurrency);
        while let Some(result) = stream.next().await {
            let (chunk_idx, data) = result?;
            let base = chunk_idx * batch_size;
            for item in data {
                let global_index = base + item.index as usize;
                if global_index >= all_results.len() {
                    // 服务端 index 超出请求范围 = 批次错位,直接报错。
                    return Err(EmbeddingError::BatchMismatch {
                        expected: all_results.len(),
                        actual: global_index + 1,
                    });
                }
                all_results[global_index] = Some(item.embedding);
            }
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
        &self.config.model
    }
}

/// OpenAI Embedding API response
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
        let old = save_and_set("OPENAI_API_KEY", "test-key-123");
        let result = OpenAIEmbeddings::from_env_result();
        assert!(result.is_ok());
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_err_when_key_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old = env::var("OPENAI_API_KEY").ok();
        env::remove_var("OPENAI_API_KEY");
        let result = OpenAIEmbeddings::from_env_result();
        match result {
            Err(msg) => assert!(msg.to_string().contains("OPENAI_API_KEY")),
            Ok(_) => panic!("expected error when OPENAI_API_KEY is missing"),
        }
        restore("OPENAI_API_KEY", old);
    }

    #[test]
    fn test_from_env_result_uses_optional_vars() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
    use crate::test_support::{spawn_embeddings_stub, spawn_status_stub};
    use std::sync::Arc;

    /// P0-1: 跨 chunk 边界批量对齐——顺序与输入一致，无空向量、无错位。
    #[tokio::test]
    async fn test_embed_documents_batch_alignment() {
        let base_url = spawn_embeddings_stub(Arc::new(|n| n)).await;
        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 2,
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

        // 5 条文本 → 3 个 chunk（2/2/1），验证跨 chunk 顺序正确。
        // stub 以文本字节和编码向量,因此每条文本必须落到自己的槽位,
        // 任何错位/重复都会让对应槽位的向量与文本不匹配。
        let texts = ["a", "b", "c", "d", "e"];
        let results = embeddings
            .embed_documents(&texts)
            .await
            .expect("batch embedding should succeed");
        assert_eq!(results.len(), 5);
        for (i, text) in texts.iter().enumerate() {
            // stub 返回 [sum, 1.0];P2-8 归一化后与逐条归一化的期望值一致,
            // 且每条向量仍互不相同,可验证跨 chunk 顺序对齐。
            let raw = text.bytes().map(|b| b as f32).sum::<f32>();
            let mut expected = vec![raw, 1.0];
            crate::l2_normalize(&mut expected);
            assert_eq!(results[i], expected, "text #{} out of alignment", i);
        }
    }

    /// P0-1: 服务端某 chunk 少返回 → 显式 `EmptyVectorInBatch`，而非静默空向量。
    #[tokio::test]
    async fn test_embed_documents_truncated_response_errors() {
        let base_url = spawn_embeddings_stub(Arc::new(|n| n.saturating_sub(1))).await;
        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 2,
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

        let result = embeddings.embed_documents(&["a", "b"]).await;
        assert!(
            matches!(result, Err(EmbeddingError::EmptyVectorInBatch)),
            "truncated response should report EmptyVectorInBatch, got: {:?}",
            result
        );
    }

    /// P0-1: 服务端返回 index 超出请求范围 → 显式 `BatchMismatch`。
    #[tokio::test]
    async fn test_embed_documents_overrun_returns_batch_mismatch() {
        let base_url = spawn_embeddings_stub(Arc::new(|_| 100)).await;
        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 2,
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

        let result = embeddings.embed_documents(&["a", "b"]).await;
        assert!(
            matches!(result, Err(EmbeddingError::BatchMismatch { .. })),
            "out-of-range index should report BatchMismatch, got: {:?}",
            result
        );
    }

    /// P2-5: `embed_query` 接线重试——429 两次后 200，请求共 3 次且返回成功。
    #[tokio::test]
    async fn test_embed_query_retries_on_429() {
        use std::sync::atomic::Ordering;

        let success_body = r#"{"data":[{"object":"embedding","index":0,"embedding":[0.6,0.8]}],"model":"stub","usage":{"prompt_tokens":0,"total_tokens":0}}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 2048,
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

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

    /// P2-5: `embed_documents` 的每个 chunk 同样走重试——429 后成功。
    #[tokio::test]
    async fn test_embed_documents_retries_on_429() {
        use std::sync::atomic::Ordering;

        let success_body = r#"{"data":[{"object":"embedding","index":0,"embedding":[1.0,0.0]},{"object":"embedding","index":1,"embedding":[0.0,1.0]}],"model":"stub","usage":{"prompt_tokens":0,"total_tokens":0}}"#;
        let (base_url, requests) = spawn_status_stub(429, 2, 200, success_body).await;
        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 2,
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

        let results = embeddings
            .embed_documents(&["a", "b"])
            .await
            .expect("should retry successfully after 429");
        assert_eq!(results.len(), 2);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            3,
            "single chunk: 1 initial + 2 retries"
        );
    }

    /// P2-6: 多 chunk 并发请求——并发度受 `MAX_CONCURRENT_CHUNKS` 约束，
    /// 且确实并行（最大在途请求数 > 1），而非串行。
    #[tokio::test]
    async fn test_embed_documents_chunks_run_concurrently() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{addr}");

        let in_flight_server = in_flight.clone();
        let max_in_flight_server = max_in_flight.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let in_flight = in_flight_server.clone();
                let max_in_flight = max_in_flight_server.clone();
                tokio::spawn(async move {
                    let mut header = Vec::new();
                    let mut byte = [0u8; 1];
                    while header.len() < 64 * 1024 {
                        if socket.read_exact(&mut byte).await.is_err() {
                            return;
                        }
                        header.push(byte[0]);
                        if header.ends_with(b"\r\n\r\n") {
                            break;
                        }
                    }
                    let header_str = String::from_utf8_lossy(&header).to_lowercase();
                    let content_length: usize = header_str
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse().ok())
                        .unwrap_or(0);
                    let mut body = vec![0u8; content_length];
                    if content_length > 0 && socket.read_exact(&mut body).await.is_err() {
                        return;
                    }
                    let body_str = String::from_utf8_lossy(&body);
                    let inputs: Vec<String> = serde_json::from_str::<serde_json::Value>(&body_str)
                        .ok()
                        .and_then(|v| v.get("input").cloned())
                        .and_then(|input| match input {
                            serde_json::Value::String(s) => Some(vec![s]),
                            serde_json::Value::Array(a) => Some(
                                a.iter()
                                    .filter_map(|x| x.as_str().map(String::from))
                                    .collect(),
                            ),
                            _ => None,
                        })
                        .unwrap_or_default();

                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    max_in_flight.fetch_max(now, Ordering::SeqCst);
                    // 50ms 让各 chunk 有重叠窗口，验证确实并行。
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);

                    let data: Vec<serde_json::Value> = inputs
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let raw = s.bytes().map(|b| b as f32).sum::<f32>();
                            let mut v = vec![raw, 1.0];
                            crate::l2_normalize(&mut v);
                            serde_json::json!({
                                "object": "embedding",
                                "index": i,
                                "embedding": v,
                            })
                        })
                        .collect();
                    let json = serde_json::json!({
                        "data": data,
                        "model": "stub",
                        "usage": { "prompt_tokens": 0, "total_tokens": 0 },
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        json.len(),
                        json
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });

        let config = OpenAIEmbeddingsConfig {
            api_key: "test-key".into(),
            base_url,
            model: "text-embedding-ada-002".into(),
            batch_size: 1, // 每条文本一个 chunk → 5 个并发 future
        };
        let embeddings = OpenAIEmbeddings::new(config).unwrap();

        let results = embeddings
            .embed_documents(&["a", "b", "c", "d", "e"])
            .await
            .expect("concurrent batch should succeed");
        assert_eq!(results.len(), 5);
        let peak = max_in_flight.load(Ordering::SeqCst);
        assert!(
            peak >= 2,
            "multiple chunks should run concurrently (max in-flight = {peak}), not serially"
        );
        assert!(
            peak <= super::MAX_CONCURRENT_CHUNKS,
            "concurrency must not exceed MAX_CONCURRENT_CHUNKS (actual {peak})"
        );
    }

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
}
