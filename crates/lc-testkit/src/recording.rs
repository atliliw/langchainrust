//! `RecordingProvider`:真实调用一次,把请求/响应对追加到 JSONL 录制文件。
//!
//! 录制是**旁路**:真实调用失败就返回失败、不写录播;真实调用成功但写盘失败
//! 仅 `log::warn!`,不阻断真实结果。

use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_providers::ProviderError;
use lc_schema::Message;
use serde::{Deserialize, Serialize};

use crate::error::TestkitError;

/// 一次录制的请求/响应对,序列化为 JSONL 一行。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedExchange {
    /// 请求(含 system/user/assistant/tool 历史)。
    pub messages: Vec<Message>,
    /// 完整响应。
    pub response: LLMResult,
}

/// 追加写录制文件的共享句柄(append 模式,std Mutex 保护)。
pub struct Recorder {
    file: Mutex<std::fs::File>,
}

impl Recorder {
    /// 打开/创建录制文件。打不开 → 构造期直接 `Err`(fail fast)。
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Best-effort 追加一条录播:失败只 `log::warn!`,绝不向上传播。
    pub fn record(&self, exchange: &RecordedExchange) {
        let line = match serde_json::to_string(exchange) {
            Ok(line) => line,
            Err(e) => {
                log::warn!("lc-testkit: failed to serialize recording: {e}");
                return;
            }
        };
        let Ok(mut file) = self.file.lock() else {
            log::warn!("lc-testkit: recording lock poisoned");
            return;
        };
        if let Err(e) = writeln!(file, "{line}") {
            log::warn!("lc-testkit: failed to append recording: {e}");
        }
    }
}

/// 把内层模型错误映射为 `TestkitError`(经 `ProviderError` 无损透传)。
fn to_testkit<E: Into<ProviderError>>(e: E) -> TestkitError {
    TestkitError::Inner(e.into())
}

/// 包裹任意 `BaseChatModel`:成功响应后把请求/响应对追加到 JSONL。
pub struct RecordingProvider<M> {
    inner: M,
    recorder: Arc<Recorder>,
    model_name: String,
}

impl<M> RecordingProvider<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: Into<ProviderError>,
{
    /// 用内层模型 + 录制文件构造。文件打不开 → `Err`。
    pub fn new(inner: M, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let model_name = format!("{}-recorded", inner.model_name());
        let recorder = Arc::new(Recorder::new(path)?);
        Ok(Self {
            inner,
            recorder,
            model_name,
        })
    }

    /// 访问内层模型。
    pub fn inner(&self) -> &M {
        &self.inner
    }
}

#[async_trait]
impl<M> Runnable<Vec<Message>, LLMResult> for RecordingProvider<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: Into<ProviderError>,
{
    type Error = TestkitError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }
}

impl<M> BaseLanguageModel<Vec<Message>, LLMResult> for RecordingProvider<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: Into<ProviderError>,
{
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        self.inner.get_num_tokens(text)
    }

    fn temperature(&self) -> Option<f32> {
        self.inner.temperature()
    }

    fn max_tokens(&self) -> Option<usize> {
        self.inner.max_tokens()
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.inner = self.inner.with_temperature(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.inner = self.inner.with_max_tokens(max);
        self
    }
}

#[async_trait]
impl<M> BaseChatModel for RecordingProvider<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: Into<ProviderError>,
{
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let response = self
            .inner
            .chat(messages.clone(), config)
            .await
            .map_err(to_testkit)?;
        self.recorder.record(&RecordedExchange {
            messages,
            response: response.clone(),
        });
        Ok(response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let mut stream = self
            .inner
            .stream_chat(messages.clone(), config)
            .await
            .map_err(to_testkit)?;
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.map_err(to_testkit)?);
        }
        let full = chunks.concat();
        let response = LLMResult {
            content: full.clone(),
            model: self.model_name.clone(),
            ..Default::default()
        };
        self.recorder
            .record(&RecordedExchange { messages, response });
        let stream = futures_util::stream::iter(vec![Ok(full)]);
        Ok(Box::pin(stream))
    }
}
