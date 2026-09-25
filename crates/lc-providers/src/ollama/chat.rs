// src/language_models/ollama/chat.rs
//! Ollama chat model implementation for local LLM deployment.
//!
//! Ollama allows running open-source LLMs locally (Llama, Mistral, CodeLlama, etc.)
//! with an OpenAI-compatible API interface.

use async_trait::async_trait;
use futures_util::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::json;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::OllamaConfig;
use crate::openai::sse::{SSEParser, SseByteFramer, StreamToolCallAccumulator};
use crate::ProviderError;
use lc_callbacks::RunType;
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{run_tree_from_config, Runnable};
use lc_core::tools::{StructuredOutput, ToolCall, ToolDefinition};
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Ollama chat model client for local LLM deployment.
///
/// Provides an OpenAI-compatible interface to interact with Ollama server
/// running local models like Llama, Mistral, etc.
///
/// # Example
/// ```rust,ignore
/// use langchainrust::{OllamaChat, Message};
///
/// let llm = OllamaChat::new("llama3.2");
/// let response = llm.chat(vec![
///     Message::human("What is Rust?"),
/// ], None).await?;
/// ```
#[derive(Clone, Debug)]
pub struct OllamaChat {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaChat {
    /// Creates a new OllamaChat client with the specified model.
    ///
    /// Uses default localhost:11434 as the server URL.
    ///
    /// # Arguments
    /// * `model` - The model name (e.g., "llama3.2", "mistral").
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: OllamaConfig::new(model),
            // 0.22.0 audit fix (H-P1): shared client with a connect timeout.
            client: crate::retry::default_client(),
        }
    }

    /// Creates a new OllamaChat with a custom configuration.
    ///
    /// # Arguments
    /// * `config` - A pre-configured OllamaConfig instance.
    pub fn with_config(config: OllamaConfig) -> Self {
        Self {
            config,
            // 0.22.0 audit fix (H-P1): shared client with a connect timeout.
            client: crate::retry::default_client(),
        }
    }

    /// Creates an OllamaChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let config = OllamaConfig::from_env_result()?;
        Ok(Self::with_config(config))
    }

    fn message_to_openai_format(message: &Message) -> serde_json::Value {
        match &message.message_type {
            lc_schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            lc_schema::MessageType::Human => {
                // B7: shared multimodal block builder. The Ollama shim accepts
                // image blocks; non-image attachments are rejected up front by
                // resolve_message_media(Ollama).
                if let Some(blocks) = crate::media::openai_user_blocks(message) {
                    json!({"role": "user", "content": blocks})
                } else {
                    json!({"role": "user", "content": &message.content})
                }
            }
            lc_schema::MessageType::AI => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": message.content,
                });
                if let Some(tool_calls) = &message.tool_calls {
                    msg["tool_calls"] =
                        serde_json::to_value(tool_calls).unwrap_or_else(|_| serde_json::json!([]));
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

    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        let formatted_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(Self::message_to_openai_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "messages": formatted_messages,
            "stream": stream,
        });

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        if let Some(max) = self.config.max_tokens {
            body["max_tokens"] = json!(max);
        }

        if let Some(top_p) = self.config.top_p {
            body["top_p"] = json!(top_p);
        }

        if let Some(tools) = &self.config.tools {
            body["tools"] = serde_json::to_value(tools).unwrap_or(serde_json::Value::Null);
        }

        if let Some(tool_choice) = &self.config.tool_choice {
            body["tool_choice"] = json!(tool_choice);
        }

        body
    }

    /// Binds tool definitions to the model for function calling.
    ///
    /// # Arguments
    /// * `tools` - List of tool definitions available to the model.
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        let config = OllamaConfig {
            tools: Some(tools),
            ..self.config.clone()
        };
        Self {
            config,
            client: self.client.clone(),
        }
    }

    /// Sets the tool choice strategy.
    ///
    /// # Arguments
    /// * `choice` - "auto", "none", or specific tool name.
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.config.tool_choice = Some(choice.into());
        self
    }

    /// Enables structured JSON output with a specific schema.
    ///
    /// # Type Parameters
    /// * `T` - The output type implementing Deserialize and JsonSchema.
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> OllamaStructuredOutput<T> {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T)).unwrap_or_else(|_| {
            // H64: Schema generation should not silently produce null
            serde_json::json!({"type": "object", "properties": {}})
        });

        let tool = ToolDefinition::new("structured_output", "Return structured JSON output")
            .with_parameters(schema);

        let config = OllamaConfig {
            tools: Some(vec![tool]),
            tool_choice: Some("auto".to_string()),
            ..self.config.clone()
        };

        OllamaStructuredOutput {
            config,
            client: self.client.clone(),
            _phantom: PhantomData,
        }
    }

    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, OllamaError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut messages = messages;
        crate::media::resolve_message_media(&mut messages, crate::media::MediaPolicy::Ollama)
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;
        let body = self.build_request_body(messages, false);

        // 0.22.0 audit fix (H-P2): retry transient failures (429/5xx/network).
        // A14: non-idempotent POST — see retry::TransportRetryMode; use
        // retry::SAFE_RETRY to forbid replaying a possibly-dispatched request.
        let response = crate::retry::send_with_retry(
            || {
                self.client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .json(&body)
            },
            &crate::retry::DEFAULT_RETRY,
        )
        .await
        .map_err(|e| OllamaError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OllamaError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let chat_response: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| OllamaError::Parse(e.to_string()))?;

        let choice = chat_response
            .choices
            .first()
            .ok_or_else(|| OllamaError::Api("No choices in response".to_string()))?;
        let message = &choice.message;

        Ok(LLMResult {
            content: message.content.clone().unwrap_or_default(),
            model: chat_response.model,
            token_usage: chat_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            tool_calls: message.tool_calls.clone(),
            thinking_content: None,
        })
    }

    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, OllamaError>> + Send>>, OllamaError>
    {
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut messages = messages;
        crate::media::resolve_message_media(&mut messages, crate::media::MediaPolicy::Ollama)
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| OllamaError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OllamaError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let byte_stream = response.bytes_stream();
        let parser = Arc::new(Mutex::new((SSEParser::new(), SseByteFramer::new())));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, OllamaError>>(64);

        let parser_clone = parser.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            // 0.22.0 C2: accumulate streaming tool_calls deltas (Ollama speaks
            // the OpenAI-compatible delta format but the loop previously only
            // forwarded `delta.content`, silently dropping tool calls).
            let mut tool_acc = StreamToolCallAccumulator::default();
            // 0.22.0 audit fix (Medium): `[DONE]` must break the outer loop
            // (instead of `return`) so the terminal tool-call flush below
            // still runs.
            let mut done = false;
            // A12 parity: track whether a terminal marker was seen before the
            // stream ended. (OpenAI/Azure/Anthropic already guard this; before
            // this fix Ollama silently returned truncated output as if complete.)
            let mut saw_terminal = false;
            while let Some(chunk_result) = byte_stream.next().await {
                // H2 fix: propagate network errors outside the mutex scope
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(OllamaError::Http(e.to_string()))).await;
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
                // parser_guard is dropped here, before any await

                for event in events {
                    if event.is_done() {
                        done = true;
                        saw_terminal = true;
                        break;
                    }
                    // 解析失败的 SSE chunk 不再静默丢弃:记 error 日志,
                    // 避免流式回复因单条坏数据被截断却毫无提示
                    match event.parse_openai_chunk() {
                        Ok(Some(chunk)) => {
                            if let Some(choice) = chunk.choices.first() {
                                // Some OpenAI-compatible servers (incl. Ollama's
                                // /v1/chat/completions) may drop [DONE] and signal
                                // completion only via a non-null finish_reason.
                                if choice.finish_reason.is_some() {
                                    saw_terminal = true;
                                }
                                if let Some(content) = &choice.delta.content {
                                    if tx.send(Ok(StreamChunk::new(content))).await.is_err() {
                                        return;
                                    }
                                }
                                // C2: fold tool-call fragments into the accumulator
                                if let Some(deltas) = &choice.delta.tool_calls {
                                    for delta in deltas {
                                        tool_acc.push(delta);
                                    }
                                }
                            }
                            // Ollama 本地服务不保证回传 usage,维持 token_usage: None
                            // (与 wrapper/client 代理路径一致,文档注明)。
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
                if done {
                    break;
                }
            }
            // A12 parity: the byte stream ended without any terminal marker —
            // the connection was truncated, so surface the interruption instead
            // of handing the partial reply back as if it were complete.
            if !saw_terminal {
                let _ = tx
                    .send(Err(OllamaError::StreamInterrupted(
                        "connection closed before [DONE] or finish_reason".to_string(),
                    )))
                    .await;
                return;
            }

            // C2: stream exhausted — flush accumulated tool calls as a terminal chunk
            let calls = tool_acc.build();
            if !calls.is_empty() {
                let _ = tx
                    .send(Ok(StreamChunk {
                        thinking_content: None,
                        text: String::new(),
                        token_usage: None,
                        tool_calls: Some(calls),
                    }))
                    .await;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod stream_truncation_tests {
    use super::*;
    use futures_util::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Body of one `data:` SSE event carrying the given content delta, terminated
    /// with `\r\n\r\n` so `SseByteFramer` completes one event. Built with `json!`
    /// so the chunk is always well-formed (openai/sse.rs requires
    /// id/object/created/model/choices to deserialize).
    fn delta_event(content: &str) -> String {
        let chunk = json!({
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [{ "index": 0, "delta": { "content": content } }],
        });
        format!("data: {chunk}\r\n\r\n")
    }

    /// F2 regression: a byte stream that ends *cleanly* (chunked terminator, no
    /// `[DONE]`, no `finish_reason`) must surface `StreamInterrupted` instead of
    /// silently handing the truncated partial output back as if it were complete.
    #[tokio::test]
    async fn truncated_stream_without_terminal_marker_is_reported() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await; // consume the POST + body

            let mut body = String::new();
            for e in [delta_event("Hel"), delta_event("lo")] {
                body.push_str(&format!("{:x}\r\n{}\r\n", e.len(), e));
            }
            body.push_str("0\r\n\r\n"); // clean end-of-body — but no [DONE]

            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{body}"
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        });

        let chat = OllamaChat::with_config(
            OllamaConfig::new("test").with_base_url(format!("http://{}", addr)),
        );
        let mut stream = chat
            .stream_chat_internal(vec![Message::human("hi")])
            .await
            .unwrap();

        let mut got: Vec<Result<StreamChunk, OllamaError>> = Vec::new();
        while let Some(item) = stream.next().await {
            got.push(item);
        }

        let last = got
            .pop()
            .expect("truncated stream must yield at least the terminal error");
        assert!(
            matches!(&last, Err(OllamaError::StreamInterrupted(_))),
            "truncated stream must end with StreamInterrupted, got {last:?}"
        );
        // the partial tokens DID stream out first, then the interruption surfaced
        assert!(
            got.iter().any(|i| i.is_ok()),
            "partial tokens should have streamed before the interruption"
        );
        server.await.unwrap();
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for OllamaChat {
    type Error = OllamaError;

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

        // H4: True streaming — emit one LLMResult per token
        let stream = token_stream.map(move |token_result| match token_result {
            Ok(chunk) => Ok(LLMResult {
                content: chunk.text,
                model: model.clone(),
                token_usage: chunk.token_usage,
                // 0.22.0 C2: accumulated tool calls surface on the streaming path
                tool_calls: chunk.tool_calls,
                thinking_content: None,
            }),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for OllamaChat {
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
impl BaseChatModel for OllamaChat {
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

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Expose the inherent tool-binding capability at the trait level so it
        // survives being wrapped by `ChatModelWrapper` / `LLMClient` (Q1).
        Some(Box::new(self.bind_tools(tools)))
    }
}

/// Errors that can occur when interacting with the Ollama chat API.
#[derive(Debug)]
#[non_exhaustive]
pub enum OllamaError {
    /// HTTP request error.
    Http(String),
    /// API returned an error.
    Api(String),
    /// Response parsing error.
    Parse(String),
    /// The SSE stream ended before a terminal marker (`[DONE]`, or a chunk
    /// carrying a non-null `finish_reason`) — the connection was truncated.
    StreamInterrupted(String),
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaError::Http(msg) => write!(f, "HTTP error: {}", msg),
            OllamaError::Api(msg) => write!(f, "API error: {}", msg),
            OllamaError::Parse(msg) => write!(f, "Parse error: {}", msg),
            OllamaError::StreamInterrupted(msg) => {
                write!(f, "Stream interrupted: {}", msg)
            }
        }
    }
}

impl std::error::Error for OllamaError {}

// L2 fix: add From<String> for OllamaError, matching OpenAIError pattern
impl From<String> for OllamaError {
    fn from(s: String) -> Self {
        OllamaError::Api(s)
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OllamaChoice>,
    usage: Option<OllamaUsage>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OllamaChoice {
    index: i32,
    message: OllamaMessage,
    finish_reason: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    // 0.25.0: Ollama emits `content: null` (or omits it) on a tool-call turn;
    // a required String made every such response fail to deserialize.
    #[serde(default)]
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Structured output wrapper for Ollama chat models, parsing chat responses into a typed value.
pub struct OllamaStructuredOutput<T: DeserializeOwned + JsonSchema> {
    config: OllamaConfig,
    client: reqwest::Client,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> OllamaStructuredOutput<T> {
    /// Invokes the chat API and parses the response into the structured type `T`.
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<T, OllamaError> {
        let chat = OllamaChat {
            config: self.config.clone(),
            client: self.client.clone(),
        };

        let result = chat.chat_internal(messages).await?;
        let structured = StructuredOutput::<T>::new(result);
        structured
            .parse()
            .map_err(|e| OllamaError::Parse(e.to_string()))
    }
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
    fn test_from_env_result_ok_when_vars_set() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_url = save_and_set("OLLAMA_BASE_URL", "http://custom:11434/v1");
        let old_model = save_and_set("OLLAMA_MODEL", "llama3.2");
        let result = OllamaChat::from_env_result();
        assert!(result.is_ok());
        restore("OLLAMA_BASE_URL", old_url);
        restore("OLLAMA_MODEL", old_model);
    }

    #[test]
    fn test_from_env_result_uses_defaults_when_vars_missing() {
        let _lock = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let old_url = env::var("OLLAMA_BASE_URL").ok();
        let old_model = env::var("OLLAMA_MODEL").ok();
        env::remove_var("OLLAMA_BASE_URL");
        env::remove_var("OLLAMA_MODEL");
        let chat = OllamaChat::from_env_result().unwrap();
        assert_eq!(chat.model_name(), "");
        restore("OLLAMA_BASE_URL", old_url);
        restore("OLLAMA_MODEL", old_model);
    }
}

// 0.25.0 B2: a tool-call turn carries `content: null`; it must deserialize and
// surface as an empty string with the tool call intact.
#[cfg(test)]
mod null_content_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn non_stream_null_content_with_tool_call_parses() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let _ = socket.read(&mut buf).await; // consume the POST + body
            let response = "{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":1,\
\"model\":\"llama3.2\",\"choices\":[{\"index\":0,\"finish_reason\":\"tool_calls\",\
\"message\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\
\"id\":\"call_1\",\"type\":\"function\",\"function\":{\
\"name\":\"get_weather\",\"arguments\":\"{}\"}}]}}],\
\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{response}"
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        });

        let chat = OllamaChat::with_config(
            OllamaConfig::new("test").with_base_url(format!("http://{addr}")),
        );
        let result = chat
            .chat_internal(vec![Message::human("weather?")])
            .await
            .expect("null content must not fail");

        assert_eq!(result.content, "");
        let calls = result.tool_calls.expect("tool call preserved");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(result.token_usage.unwrap().total_tokens, 6);
    }

    #[test]
    fn missing_content_field_defaults_to_none() {
        let msg: OllamaMessage =
            serde_json::from_str(r#"{"role":"assistant","tool_calls":[]}"#).unwrap();
        assert!(msg.content.is_none());
    }
}
