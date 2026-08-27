//! `RecordingProvider`: makes one real call, appending the request/response pair to the JSONL recording file.
//!
//! Recording is **pass-through**: a failed real call returns the failure without writing;
//! a successful call whose write fails only `log::warn!`s and does not block the real result.

use std::io::Write;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::tools::ToolDefinition;
use lc_providers::ProviderError;
use lc_schema::Message;
use serde::{Deserialize, Serialize};

use crate::error::TestkitError;

/// One recorded request/response pair, serialized as a single JSONL line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedExchange {
    /// The request (including system/user/assistant/tool history).
    pub messages: Vec<Message>,
    /// The full response.
    pub response: LLMResult,
    /// Tool definitions bound on the request side (non-empty after `bind_tools`).
    ///
    /// `#[serde(default)]` lets old fixtures (without a `tools` field) read as `None`, compatible
    /// with zero changes; `skip_serializing_if` keeps recordings without bound tools in the old format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Shared handle for append-mode writing to the recording file (protected by a std Mutex).
pub struct Recorder {
    file: Mutex<std::fs::File>,
}

impl Recorder {
    /// Opens/creates the recording file. If it cannot be opened, construction fails fast with `Err`.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Best-effort append of one recording: failure only `log::warn!`s and never propagates up.
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

/// Maps an inner model error into `TestkitError` (losslessly passed through `ProviderError`).
fn to_testkit<E: Into<ProviderError>>(e: E) -> TestkitError {
    TestkitError::Inner(e.into())
}

/// Wraps any `BaseChatModel`: after a successful response, appends the request/response pair to JSONL.
pub struct RecordingProvider<M> {
    inner: M,
    recorder: Arc<Recorder>,
    model_name: String,
    /// Currently bound tool definitions (set by `bind_tools`, recorded into the exchange on chat).
    tools: Option<Vec<ToolDefinition>>,
}

impl<M> RecordingProvider<M>
where
    M: BaseChatModel + Send + Sync + 'static,
    M::Error: Into<ProviderError>,
{
    /// Constructs from an inner model + recording file. An unopenable file → `Err`.
    pub fn new(inner: M, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let model_name = format!("{}-recorded", inner.model_name());
        let recorder = Arc::new(Recorder::new(path)?);
        Ok(Self {
            inner,
            recorder,
            model_name,
            tools: None,
        })
    }

    /// Accesses the inner model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Binds tools: returns a new instance that records that tool set.
    ///
    /// Later `chat` calls record the tool definitions into `RecordedExchange.tools`, letting the
    /// replay side route by tool name (see [`crate::ReplayStrategy::ByToolName`]).
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self
    where
        M: Clone,
    {
        Self {
            inner: self.inner.clone(),
            recorder: self.recorder.clone(),
            model_name: self.model_name.clone(),
            tools: Some(tools),
        }
    }
}

#[async_trait]
impl<M> Runnable<Vec<Message>, LLMResult> for RecordingProvider<M>
where
    M: BaseChatModel + Clone + Send + Sync + 'static,
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
    M: BaseChatModel + Clone + Send + Sync + 'static,
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
    M: BaseChatModel + Clone + Send + Sync + 'static,
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
            tools: self.tools.clone(),
        });
        Ok(response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let mut stream = self
            .inner
            .stream_chat(messages.clone(), config)
            .await
            .map_err(to_testkit)?;
        let mut full = String::new();
        let mut usage: Option<TokenUsage> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(to_testkit)?;
            full.push_str(&chunk.text);
            if chunk.token_usage.is_some() {
                usage = chunk.token_usage;
            }
        }
        let response = LLMResult {
            content: full.clone(),
            model: self.model_name.clone(),
            token_usage: usage.clone(),
            ..Default::default()
        };
        self.recorder.record(&RecordedExchange {
            messages,
            response,
            tools: self.tools.clone(),
        });
        let stream = futures_util::stream::iter(vec![Ok(StreamChunk {
            text: full,
            token_usage: usage,
        })]);
        Ok(Box::pin(stream))
    }

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Delegate to the inherent `bind_tools`: clone the inner model, record the tool set, return a new instance.
        Some(Box::new(self.bind_tools(tools)))
    }
}
