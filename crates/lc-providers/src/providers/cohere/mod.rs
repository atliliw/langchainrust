// lc-providers/src/providers/cohere/mod.rs
//! Cohere API implementation.
//!
//! Supports Cohere's Command R+ models with chat, streaming, and tool calling.
//! Cohere uses its own API format (not OpenAI-compatible).
//!
//! # Supported Models
//!
//! - `command-r-plus` — flagship model with RAG capabilities
//! - `command-r` — balanced performance
//! - `command` — fast and cost-effective
//! - `command-light` — lightweight model
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_providers::providers::{CohereChat, CohereConfig};
//!
//! let llm = CohereChat::new(CohereConfig::new("your-api-key"));
//! let result = llm.chat(messages, None).await?;
//! ```

mod config;
mod error;
#[cfg(test)]
mod tests;
mod types;

pub use config::{CohereConfig, COHERE_BASE_URL, COHERE_MODELS};
pub use error::CohereError;

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;
use std::pin::Pin;

use self::types::*;
use crate::provider_http::{provider_api_client, provider_request_options, provider_sse_client};
use crate::ProviderError;
use lc_callbacks::RunType;
use lc_core::http::HttpClient;
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{run_tree_from_config, Runnable};
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Cohere chat client.
///
/// Native implementation using Cohere's v2 chat API.
#[derive(Clone)]
pub struct CohereChat {
    config: CohereConfig,
    /// Buffered client for non-streaming calls (0.25.0: unified HTTP layer).
    http_api: HttpClient,
    /// SSE-profile client for streaming calls (establishment retries only).
    http_sse: HttpClient,
}

impl std::fmt::Debug for CohereChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereChat").finish_non_exhaustive()
    }
}

impl CohereChat {
    /// Creates a new CohereChat with the given configuration.
    pub fn new(config: CohereConfig) -> Self {
        Self {
            config,
            // 0.25.0: unified HTTP layer — bounded body, closed retriable
            // status set, Retry-After support, method-aware POST retries.
            http_api: provider_api_client(),
            http_sse: provider_sse_client(),
        }
    }

    /// Per-request auth/headers assembled from the provider config.
    fn request_options(&self) -> lc_core::http::RequestOptions {
        provider_request_options(true, &self.config.api_key, &[])
    }

    /// Maps a unified-layer error onto the Cohere error enum while keeping the
    /// `HTTP {status}: {body}` message shape.
    fn map_http_error(err: lc_core::http::HttpError) -> CohereError {
        match err {
            lc_core::http::HttpError::Status { status, body } => {
                CohereError::Api(format!("HTTP {status}: {body}"))
            }
            other => CohereError::Http(other.to_string()),
        }
    }

    /// Creates a CohereChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(CohereConfig::from_env_result()?))
    }

    /// Converts langchain Message to Cohere chat message format.
    ///
    /// Cohere v2 chat API uses:
    /// - `role`: "system" | "user" | "assistant" | "tool"
    /// - `content`: string or array of content parts
    fn message_to_cohere_format(message: &Message) -> serde_json::Value {
        match &message.message_type {
            lc_schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            lc_schema::MessageType::Human => json!({
                "role": "user",
                "content": message.content,
            }),
            lc_schema::MessageType::AI => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": message.content,
                });
                if let Some(tool_calls) = &message.tool_calls {
                    msg["tool_calls"] = serde_json::to_value(
                        tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    }
                                })
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap_or(serde_json::Value::Null);
                }
                msg
            }
            lc_schema::MessageType::Tool { tool_call_id } => json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": message.content,
            }),
        }
    }

    /// Builds the API request body for Cohere v2 chat.
    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        let cohere_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(Self::message_to_cohere_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "messages": cohere_messages,
            "stream": stream,
        });

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        if let Some(max) = self.config.max_tokens {
            body["max_tokens"] = json!(max);
        }

        if let Some(ref preamble) = self.config.preamble {
            body["preamble"] = json!(preamble);
        }

        // 0.25.0: v2 function calling. ToolDefinition serializes to Cohere's
        // OpenAI-compatible tool shape {"type":"function","function":{...}},
        // which v2 accepts natively and returns in the same call shape.
        if let Some(tools) = &self.config.tools {
            body["tools"] = serde_json::to_value(tools).unwrap_or(serde_json::Value::Null);
        }

        if let Some(tool_choice) = &self.config.tool_choice {
            body["tool_choice"] = json!(tool_choice);
        }

        body
    }

    /// Binds tool definitions for function calling (0.25.0: Cohere previously
    /// had no tool support at all).
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        let mut config = self.config.clone();
        config.tools = Some(tools);
        Self {
            config,
            http_api: self.http_api.clone(),
            http_sse: self.http_sse.clone(),
        }
    }

    /// Sets the tool choice strategy (`NONE`/`AUTO`/`ANY` on Cohere v2).
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.config.tool_choice = Some(choice.into());
        self
    }

    /// Internal chat implementation.
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, CohereError> {
        let url = format!("{}/chat", self.config.base_url);
        let body = self.build_request_body(messages, false);

        // 0.25.0: unified HTTP layer — retriable status set, Retry-After and
        // POST pre-dispatch-only retry semantics live in lc_core::http.
        let response = self
            .http_api
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(Self::map_http_error)?;

        let chat_response: CohereChatResponse =
            serde_json::from_str(&response.body).map_err(|e| CohereError::Parse(e.to_string()))?;

        let message = chat_response
            .message
            .ok_or_else(|| CohereError::Api("No message in response".to_string()))?;

        // Concatenate the text parts; non-text parts (e.g. tool results in
        // history) have no `text` and are skipped (0.25.0: text is Optional).
        let content = message
            .content
            .iter()
            .filter_map(|part| part.text.clone())
            .collect::<Vec<_>>()
            .join("");

        let tool_calls = if message.tool_calls.is_empty() {
            None
        } else {
            Some(
                message
                    .tool_calls
                    .into_iter()
                    .map(|tc| lc_core::tools::ToolCall {
                        id: tc.id,
                        tool_type: "function".to_string(),
                        function: lc_core::tools::FunctionCall {
                            name: tc.function.name,
                            arguments: normalize_arguments(tc.function.arguments),
                        },
                    })
                    .collect(),
            )
        };

        Ok(LLMResult {
            content,
            model: chat_response.model,
            token_usage: chat_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.tokens.input_tokens,
                completion_tokens: u.tokens.output_tokens,
                total_tokens: u.tokens.input_tokens + u.tokens.output_tokens,
            }),
            tool_calls,
            thinking_content: None,
        })
    }

    /// Internal streaming implementation.
    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, CohereError>> + Send>>, CohereError>
    {
        use crate::openai::sse::{SSEParser, SseByteFramer};
        use std::sync::{Arc, Mutex};

        let url = format!("{}/chat", self.config.base_url);
        let body = self.build_request_body(messages, true);

        // 0.25.0: unified HTTP layer. Retries cover establishment only; once
        // the response head arrives the stream runs without reconnecting.
        let byte_stream = self
            .http_sse
            .open_sse(&url, Some(&body), self.request_options())
            .await
            .map_err(Self::map_http_error)?;
        let parser = Arc::new(Mutex::new((SSEParser::new(), SseByteFramer::new())));
        let parser_clone = parser.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, CohereError>>(64);

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut byte_stream = byte_stream;
            // 0.25.0: accumulate fragmented tool-call deltas by index; the
            // complete calls are flushed on `message-end`.
            let mut tool_acc = CohereToolCallAccumulator::default();
            let mut tool_calls_emitted = false;
            let mut saw_terminal = false;
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(CohereError::Http(e.to_string()))).await;
                        return;
                    }
                };

                let events = {
                    // 0.22.0 C1: byte-layer framing; only complete events are decoded
                    let mut guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    let mut out = Vec::new();
                    for text in guard.1.push(&chunk_bytes) {
                        out.extend(guard.0.parse(&text));
                    }
                    out
                };

                for event in events {
                    if event.is_done() {
                        // Tolerate OpenAI-compatible proxies: `[DONE]` also counts
                        // as a terminal. Real Cohere never sends it.
                        saw_terminal = true;
                        break;
                    }
                    // 0.20.0 P4: Cohere v2 SSE 是**自己的**事件格式,不是 OpenAI
                    // 兼容格式——没有 choices[0].delta.content。文本增量在
                    // content-delta 事件的 delta.message.content.text,usage 在
                    // message-end 事件的 delta.usage.tokens。旧实现复用
                    // parse_openai_chunk,因缺 id/object/created/model/choices
                    // 每条事件反序列化都失败,流式静默产出空文本。此处改用专用解析。
                    // 解析失败的 SSE chunk 不静默丢弃:记 error 日志,避免流式回复
                    // 因单条坏数据被截断却毫无提示。
                    match parse_cohere_event(&event.data) {
                        Ok(Some(ev)) => {
                            // M-8: Cohere's real terminal is a `message-end` event
                            // (it never sends `[DONE]`); track it so a truncated
                            // connection is surfaced as an error rather than a
                            // "complete" partial reply.
                            if ev.event_type == "message-end" {
                                saw_terminal = true;
                            }
                            if let Some(chunk) = cohere_event_to_chunk(&mut tool_acc, &ev) {
                                if chunk.tool_calls.is_some() {
                                    tool_calls_emitted = true;
                                }
                                if tx.send(Ok(chunk)).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            log::error!(
                                "Failed to parse streaming SSE chunk (skipping this token): {}",
                                e
                            );
                        }
                    }
                }
                if saw_terminal {
                    break;
                }
            }
            // M-8: the byte stream ended without a terminal `message-end` (and no
            // `[DONE]`): the connection was truncated. Surface the interruption
            // instead of handing the partial reply back as if it were complete —
            // aligns with OpenAI/Anthropic/Gemini/Azure/Ollama.
            if !saw_terminal {
                let _ = tx
                    .send(Err(CohereError::StreamInterrupted(
                        "connection closed before message-end".to_string(),
                    )))
                    .await;
                return;
            }

            // Defensive flush for streams that close without `message-end`:
            // tool calls must not vanish just because the terminal event did.
            if !tool_calls_emitted {
                if let Some(calls) = tool_acc.build() {
                    let _ = tx
                        .send(Ok(StreamChunk {
                            text: String::new(),
                            thinking_content: None,
                            token_usage: None,
                            tool_calls: Some(calls),
                        }))
                        .await;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

/// Parses one Cohere v2 SSE event's `data:` payload into a typed stream event.
///
/// Returns `Ok(None)` for the OpenAI-style `[DONE]` terminator (Cohere never
/// sends it — it closes the connection after `message-end` — but this tolerates
/// OpenAI-compatible proxies).
///
/// 0.20.0 P4: the replacement for `SSEEvent::parse_openai_chunk`, which
/// deserializes OpenAI's `id/object/created/model/choices` shape and therefore
/// fails on every real Cohere event.
fn parse_cohere_event(data: &str) -> Result<Option<CohereStreamEvent>, serde_json::Error> {
    if data == "[DONE]" {
        return Ok(None);
    }
    let parsed = serde_json::from_str(data)?;
    Ok(Some(parsed))
}

/// Normalizes a Cohere v2 `function.arguments` payload to the string shape the
/// rest of the workspace expects: non-streaming responses carry a JSON object
/// (returned compactly serialized), while streaming histories may carry a
/// pre-serialized JSON string (returned verbatim).
fn normalize_arguments(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    }
}

/// One in-flight tool call assembled from Cohere streaming deltas.
#[derive(Default)]
struct CohereToolCallState {
    id: String,
    name: String,
    arguments: String,
}

/// Accumulates fragmented Cohere v2 streaming tool calls. `tool-call-start`
/// seeds `id`/`name`, subsequent `tool-call-delta` events append JSON
/// arguments fragments, all correlated by the wire `index`.
#[derive(Default)]
struct CohereToolCallAccumulator {
    // BTreeMap so the finished calls come back in deterministic index order.
    calls: std::collections::BTreeMap<usize, CohereToolCallState>,
}

impl CohereToolCallAccumulator {
    fn push(&mut self, delta: &CohereStreamToolCallDelta) {
        let entry = self.calls.entry(delta.index).or_default();
        if let Some(id) = &delta.id {
            if !id.is_empty() {
                entry.id.clone_from(id);
            }
        }
        if let Some(function) = &delta.function {
            if let Some(name) = &function.name {
                if !name.is_empty() {
                    entry.name.clone_from(name);
                }
            }
            if let Some(arguments) = &function.arguments {
                entry.arguments.push_str(arguments);
            }
        }
    }

    /// Builds the completed calls. Entries missing `id` or `name` (a truncated
    /// stream) are dropped rather than emitting an un-routable tool call.
    fn build(&self) -> Option<Vec<lc_core::tools::ToolCall>> {
        let calls: Vec<_> = self
            .calls
            .values()
            .filter(|state| !state.id.is_empty() && !state.name.is_empty())
            .map(|state| lc_core::tools::ToolCall {
                id: state.id.clone(),
                tool_type: "function".to_string(),
                function: lc_core::tools::FunctionCall {
                    name: state.name.clone(),
                    arguments: state.arguments.clone(),
                },
            })
            .collect();
        (!calls.is_empty()).then_some(calls)
    }
}

/// Maps one parsed Cohere streaming event to an optional `StreamChunk`,
/// feeding tool-call fragments into `acc`.
///
/// - `content-delta` → a text chunk (empty deltas are dropped)
/// - `tool-plan-delta` → a `thinking_content` chunk (the model's plan text)
/// - `tool-call-start` / `tool-call-delta` → accumulate by index, no chunk
/// - `message-end` → terminal chunk carrying usage and/or completed tool calls
/// - any other event → `None` (framing events produce no output)
fn cohere_event_to_chunk(
    acc: &mut CohereToolCallAccumulator,
    ev: &CohereStreamEvent,
) -> Option<StreamChunk> {
    match ev.event_type.as_str() {
        "content-delta" => ev
            .delta
            .as_ref()
            .and_then(|d| d.message.as_ref())
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.text.clone())
            .filter(|t| !t.is_empty())
            .map(StreamChunk::new),
        "tool-plan-delta" => ev
            .delta
            .as_ref()
            .and_then(|d| d.message.as_ref())
            .and_then(|m| m.tool_plan.clone())
            .filter(|t| !t.is_empty())
            .map(|plan| StreamChunk {
                text: String::new(),
                thinking_content: Some(plan),
                token_usage: None,
                tool_calls: None,
            }),
        "tool-call-start" | "tool-call-delta" => {
            if let Some(delta) = &ev.delta {
                if let Some(message) = &delta.message {
                    for call in &message.tool_calls {
                        acc.push(call);
                    }
                }
            }
            None
        }
        "message-end" => {
            let tool_calls = acc.build();
            let token_usage = ev
                .delta
                .as_ref()
                .and_then(|d| d.usage.as_ref())
                .map(|usage| TokenUsage {
                    prompt_tokens: usage.tokens.input_tokens,
                    completion_tokens: usage.tokens.output_tokens,
                    total_tokens: usage.tokens.input_tokens + usage.tokens.output_tokens,
                });
            (token_usage.is_some() || tool_calls.is_some()).then(|| StreamChunk {
                text: String::new(),
                thinking_content: None,
                token_usage,
                tool_calls,
            })
        }
        _ => None,
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for CohereChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        lc_core::token_counter::count_tokens(text).unwrap_or_else(|e| {
            // 编码器加载失败时按字节数高估(宁可略高,不静默按 0 算导致路由/截断误判)
            log::warn!("Token counting failed, falling back to byte-length estimation: {e}");
            text.len()
        })
    }

    fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    fn max_tokens(&self) -> Option<usize> {
        self.config.max_tokens
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.config.max_tokens = Some(max);
        self
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for CohereChat {
    type Error = CohereError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, config).await
    }

    async fn stream(
        &self,
        input: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let model = self.config.model.clone();
        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = Some(m);
        }
        let token_stream = effective.stream_chat_internal(input).await?;

        let stream = token_stream.map(move |token_result| match token_result {
            Ok(chunk) => Ok(LLMResult {
                content: chunk.text,
                model: model.clone(),
                token_usage: chunk.token_usage,
                tool_calls: chunk.tool_calls,
                thinking_content: chunk.thinking_content,
            }),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseChatModel for CohereChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:chat", self.config.model));

        let mut run = run_tree_from_config(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
                "model": self.config.model,
            }),
            config.as_ref(),
        );

        if let Some(ref cfg) = config {
            if let Some(ref callbacks) = cfg.callbacks {
                for handler in callbacks.handlers() {
                    handler.on_llm_start(&run, &messages).await;
                }
            }
        }

        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = Some(m);
        }
        let result = effective.chat_internal(messages.clone()).await;

        match result {
            Ok(response) => {
                run.end(json!({
                    "content": &response.content,
                    "model": &response.model,
                    "token_usage": &response.token_usage,
                }));

                if let Some(ref cfg) = config {
                    if let Some(ref callbacks) = cfg.callbacks {
                        for handler in callbacks.handlers() {
                            handler.on_llm_end(&run, &response.content).await;
                        }
                    }
                }

                Ok(response)
            }
            Err(e) => {
                run.end_with_error(e.to_string());

                if let Some(ref cfg) = config {
                    if let Some(ref callbacks) = cfg.callbacks {
                        for handler in callbacks.handlers() {
                            handler.on_llm_error(&run, &e.to_string()).await;
                        }
                    }
                }

                Err(e)
            }
        }
    }

    async fn stream_chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:stream", self.config.model));

        let run = run_tree_from_config(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.len(),
                "model": self.config.model,
            }),
            config.as_ref(),
        );

        if let Some(ref cfg) = config {
            if let Some(ref callbacks) = cfg.callbacks {
                for handler in callbacks.handlers() {
                    handler.on_llm_start(&run, &messages).await;
                }
            }
        }

        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = Some(m);
        }
        let stream = effective.stream_chat_internal(messages).await?;

        let callbacks = config.and_then(|c| c.callbacks);
        let stream = stream.then(move |token_result| {
            let cbs = callbacks.clone();
            let run = run.clone();
            async move {
                if let Some(ref cbs) = cbs {
                    if let Ok(ref token) = token_result {
                        for handler in cbs.handlers() {
                            handler.on_llm_new_token(&run, &token.text).await;
                        }
                    }
                }
                token_result
            }
        });

        Ok(Box::pin(stream))
    }
}
