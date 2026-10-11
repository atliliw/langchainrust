//! `RecordingProvider`: makes one real call, appending the request/response pair to the JSONL recording file.
//!
//! Recording is **pass-through**: a failed real call returns the failure without writing;
//! a successful call whose write fails only `log::warn!`s and does not block the real result.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_core::tools::ToolCall;
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
    /// `#[serde(default)]` lets old fixtures (without a `tools` field) read as empty, compatible
    /// with zero changes; `skip_serializing_if` keeps recordings without bound tools in the old
    /// format (no `tools` key emitted). A plain `Vec` collapses the redundant `None`/empty
    /// distinction — "no tools bound" and "an empty tool list" are the same state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
}

/// Minimal replay-qualification check: the invariants replay *relies* on that plain JSON
/// deserialization does not guarantee. Replay routes by tool name (`ByToolName`) and
/// Exact-matches on whole messages, so a tool call with an empty `id`/`name` is not a valid
/// recording (it could never be routed/matched faithfully). Returns the first problem, or
/// `None` when the exchange is replay-qualified.
fn validate_exchange(exchange: &RecordedExchange) -> Option<String> {
    let check_calls = |calls: &[ToolCall], ctx: &str| -> Option<String> {
        for (j, c) in calls.iter().enumerate() {
            if c.id.is_empty() {
                return Some(format!("{ctx}[{j}] has an empty id"));
            }
            if c.name().is_empty() {
                return Some(format!("{ctx}[{j}] has an empty name"));
            }
        }
        None
    };
    for (i, m) in exchange.messages.iter().enumerate() {
        if let Some(calls) = &m.tool_calls {
            if let Some(detail) = check_calls(calls, &format!("message[{i}].tool_calls")) {
                return Some(detail);
            }
        }
    }
    if let Some(calls) = &exchange.response.tool_calls {
        if let Some(detail) = check_calls(calls, "response.tool_calls") {
            return Some(detail);
        }
    }
    None
}

/// Reads every JSONL exchange from a recording file into memory (blank lines skipped),
/// validating each line against the minimal replay schema (`read_exchanges` is the single
/// parse site for the recording format).
///
/// Shared by [`ReplayProvider::from_file`](crate::ReplayProvider::from_file) and the
/// golden-dataset builder (N2) so the file format is parsed in exactly one place.
pub fn read_exchanges(path: impl AsRef<Path>) -> Result<Vec<RecordedExchange>, TestkitError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut exchanges = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let exchange: RecordedExchange = serde_json::from_str(line).map_err(|e| {
            TestkitError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid recording line: {e}"),
            ))
        })?;
        if let Some(detail) = validate_exchange(&exchange) {
            return Err(TestkitError::InvalidRecording {
                index: exchanges.len(),
                detail,
            });
        }
        exchanges.push(exchange);
    }
    Ok(exchanges)
}

/// Shared handle for append-mode writing to the recording file (protected by a std Mutex).
///
/// 0.27.0 L316: the raw `File` is wrapped in a `BufWriter` so sequential
/// appends coalesce into one underlying write instead of one `write(2)` syscall
/// per exchange (recording paths run on network-synced dirs where per-call sync
/// is the actual cost). Data is therefore **buffered** until an explicit
/// [`Recorder::flush`], a completed stream, or the recorder is dropped — so
/// callers that want a line durable *now* (e.g. after a stream finishes) must
/// flush, mirroring the buffered-writer contract `write_golden_jsonl` already uses.
pub struct Recorder {
    file: Mutex<std::io::BufWriter<std::fs::File>>,
}

impl Recorder {
    /// Opens/creates the recording file. If it cannot be opened, construction fails fast with `Err`.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(std::io::BufWriter::new(file)),
        })
    }

    /// Best-effort append of one recording: failure only `log::warn!`s and never propagates up.
    ///
    /// The line is written into the internal buffer; it is not necessarily on
    /// disk until [`flush`](Self::flush) or drop (L316 write buffering).
    pub fn record(&self, exchange: &RecordedExchange) {
        let line = match serde_json::to_string(exchange) {
            Ok(line) => line,
            Err(e) => {
                log::warn!("lc-testkit: failed to serialize recording: {e}");
                return;
            }
        };
        let Ok(mut writer) = self.file.lock() else {
            log::warn!("lc-testkit: recording lock poisoned");
            return;
        };
        if let Err(e) = writeln!(writer, "{line}") {
            log::warn!("lc-testkit: failed to append recording: {e}");
        }
    }

    /// Forces any buffered recording lines to disk. Idempotent.
    ///
    /// A recorded-but-unflushed exchange can still be lost on a crash; call this
    /// at a boundary where durability matters (end of a test, after a stream).
    pub fn flush(&self) -> std::io::Result<()> {
        let mut writer = self.file.lock().unwrap_or_else(|e| e.into_inner());
        writer.flush()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // Runs only when the last `Arc<Recorder>` is dropped, so no other thread
        // can hold the lock. Guarantees buffered lines are not silently lost when
        // a recording provider ends scope without an explicit flush.
        if let Ok(writer) = self.file.get_mut() {
            if let Err(e) = writer.flush() {
                log::warn!("lc-testkit: failed to flush recording on drop: {e}");
            }
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
    tools: Vec<ToolDefinition>,
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
            tools: Vec::new(),
        })
    }

    /// Accesses the inner model.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Forces the buffered recording to disk (L316). Use at a test boundary where
    /// durability matters — a completed exchange buffered by [`Recorder::record`]
    /// can otherwise be lost on a crash before the recorder is dropped.
    pub fn flush(&self) -> std::io::Result<()> {
        self.recorder.flush()
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
            tools,
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
        // L316: the recording closed-loop guarantee — once the call returns,
        // the exchange must be durable on disk (replay reads it back without a
        // separate flush). Flush the buffered line so a completed turn never
        // waits on drop or an explicit flush.
        if let Err(e) = self.recorder.flush() {
            log::warn!("lc-testkit: failed to flush recording after chat: {e}");
        }
        Ok(response)
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        let inner = self
            .inner
            .stream_chat(messages.clone(), config)
            .await
            .map_err(to_testkit)?;
        let recorder = self.recorder.clone();
        let model_name = self.model_name.clone();
        let tools = self.tools.clone();
        let record_request = messages;

        // I1: true streaming + recording. Forward every chunk as it arrives (so
        // consumer-side `on_new_token` / backpressure apply), accumulating the
        // full text / usage / tool calls / thinking along the way. The assembled
        // response is written to the recording only once the stream completes —
        // and `tool_calls` are captured so replay can route tool exchanges. A
        // mid-stream error is forwarded and nothing is recorded (matching the
        // failure contract of the non-streaming `chat`).
        let seed = (
            inner,
            String::new(),
            None::<TokenUsage>,
            None::<Vec<ToolCall>>,
            String::new(),
            false, // `errored`: a mid-stream error already forwarded → suppress recording.
        );
        let stream = futures_util::stream::unfold(
            seed,
            move |(mut inner, mut full, mut usage, mut tcs, mut thinking, errored)| {
                let recorder = recorder.clone();
                let model_name = model_name.clone();
                let tools = tools.clone();
                let record_request = record_request.clone();
                async move {
                    if errored {
                        // A mid-stream error ended the stream last iteration; terminate
                        // without recording the truncated partial response (matches the
                        // failure contract of the non-streaming `chat`).
                        return None;
                    }
                    match inner.next().await {
                        Some(Ok(chunk)) => {
                            full.push_str(&chunk.text);
                            if chunk.token_usage.is_some() {
                                usage = chunk.token_usage.clone();
                            }
                            if chunk.tool_calls.is_some() {
                                tcs = chunk.tool_calls.clone();
                            }
                            if let Some(t) = &chunk.thinking_content {
                                thinking.push_str(t);
                            }
                            Some((Ok(chunk), (inner, full, usage, tcs, thinking, errored)))
                        }
                        Some(Err(e)) => {
                            // Forward the error and mark the stream errored, so the next
                            // iteration terminates without recording anything.
                            Some((
                                Err(to_testkit(e)),
                                (inner, full, usage, tcs, thinking, true),
                            ))
                        }
                        None => {
                            let response = LLMResult {
                                content: full,
                                model: model_name,
                                token_usage: usage,
                                tool_calls: tcs,
                                thinking_content: if thinking.is_empty() {
                                    None
                                } else {
                                    Some(thinking)
                                },
                            };
                            recorder.record(&RecordedExchange {
                                messages: record_request,
                                response,
                                tools,
                            });
                            // L316: a completed stream is a natural durable
                            // boundary — flush so the assembled exchange is
                            // actually on disk rather than left in the buffer.
                            if let Err(e) = recorder.flush() {
                                log::warn!(
                                    "lc-testkit: failed to flush recording after stream: {e}"
                                );
                            }
                            None
                        }
                    }
                }
            },
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_exchange() -> RecordedExchange {
        RecordedExchange {
            messages: vec![Message::human("hi")],
            response: LLMResult {
                content: "world".into(),
                model: "test-be".into(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            },
            tools: Vec::new(),
        }
    }

    fn scratch_dir(nonce: &str) -> (std::path::PathBuf, String) {
        let tag = format!("lc-testkit-rec-{nonce}-{}", std::process::id());
        let dir = std::env::temp_dir().join(&tag);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rec.jsonl");
        (path, tag)
    }

    #[test]
    fn records_stay_buffered_until_flush() {
        let (path, tag) = scratch_dir("buf");
        let recorder = Recorder::new(&path).unwrap();
        recorder.record(&sample_exchange());

        // L316 write buffering: a small recorded line sits in the BufWriter and
        // is not yet on disk until flush (deterministic under the 8 KiB default).
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);

        recorder.flush().unwrap();
        assert_eq!(read_exchanges(&path).unwrap().len(), 1);
        std::fs::remove_dir_all(std::env::temp_dir().join(tag)).ok();
    }

    #[test]
    fn drop_flushes_remaining_lines() {
        let (path, tag) = scratch_dir("drop");
        {
            let recorder = Recorder::new(&path).unwrap();
            recorder.record(&sample_exchange());
            recorder.record(&sample_exchange());
            // No explicit flush: buffered lines must land when the last Arc
            // drops, so an unconsumed recording provider is not silently lost.
        }
        assert_eq!(read_exchanges(&path).unwrap().len(), 2);
        std::fs::remove_dir_all(std::env::temp_dir().join(tag)).ok();
    }
}
