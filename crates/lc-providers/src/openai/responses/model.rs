// src/language_models/openai/responses/model.rs
//! ResponsesModel struct and its trait implementations.
//!
//! Contains the core model type, constructors, message conversion,
//! request building, and all trait impls (Runnable, BaseLanguageModel,
//! BaseChatModel).

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;
use std::pin::Pin;

use lc_callbacks::RunType;
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{run_tree_from_config, Runnable};
use lc_core::tools::ToolCall;
use lc_core::RunnableConfig;
use lc_schema::Message;

use super::types::{
    BuiltinTool, ResponsesApiResponse, ResponsesConfig, ResponsesContentPart, ResponsesError,
    ResponsesOutputItem, ResponsesStreamEvent,
};
use crate::provider_http::{provider_api_client, provider_request_options, provider_sse_client};
use lc_core::http::HttpClient;

// ---------------------------------------------------------------------------
// ResponsesModel
// ---------------------------------------------------------------------------

/// OpenAI Responses API model.
///
/// Uses the `/v1/responses` endpoint which provides access to built-in
/// tools such as web search, file search, code interpreter, and computer
/// use alongside standard chat capabilities.
#[derive(Clone)]
pub struct ResponsesModel {
    pub(crate) config: ResponsesConfig,
    /// Buffered client for non-streaming calls (0.25.0: unified HTTP layer).
    http_api: HttpClient,
    /// SSE-profile client for streaming calls (establishment retries only).
    http_sse: HttpClient,
}

impl ResponsesModel {
    /// Create a new model with the given configuration.
    pub fn new(config: ResponsesConfig) -> Self {
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

    /// Maps a unified-layer error onto the Responses error enum while keeping
    /// the `HTTP {status}: {body}` message shape.
    fn map_http_error(err: lc_core::http::HttpError) -> ResponsesError {
        match err {
            lc_core::http::HttpError::Status { status, body } => {
                ResponsesError::Api(format!("HTTP {status}: {body}"))
            }
            other => ResponsesError::Http(other.to_string()),
        }
    }

    /// Create a model from environment variables.
    pub fn from_env() -> Result<Self, ResponsesError> {
        Ok(Self::new(ResponsesConfig::from_env()?))
    }

    /// Add a built-in tool, returning a new model instance.
    pub fn with_builtin_tool(mut self, tool: BuiltinTool) -> Self {
        self.config.builtin_tools.push(tool);
        self
    }

    // -- Message conversion --------------------------------------------------

    /// Convert a `Message` to the Responses API input format.
    ///
    /// The Responses API accepts an `input` array where each element is
    /// either a simple message object or a more structured item.  For
    /// standard chat we use the simple message format.
    pub(crate) fn message_to_input(message: &Message) -> serde_json::Value {
        match &message.message_type {
            lc_schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            lc_schema::MessageType::Human => {
                if message.has_images() {
                    let mut content = vec![json!({"type": "input_text", "text": &message.content})];
                    for img in &message.images {
                        content.push(json!({
                            "type": "input_image",
                            "image_url": &img.url,
                        }));
                    }
                    json!({"role": "user", "content": content})
                } else {
                    json!({"role": "user", "content": message.content})
                }
            }
            lc_schema::MessageType::AI => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": message.content,
                });
                if let Some(tool_calls) = &message.tool_calls {
                    msg["tool_calls"] =
                        serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
                }
                msg
            }
            lc_schema::MessageType::Tool { tool_call_id } => json!({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": message.content,
            }),
        }
    }

    // -- Request body --------------------------------------------------------

    /// Build the JSON request body for the Responses API.
    pub(crate) fn build_request_body(
        &self,
        messages: Vec<Message>,
        stream: bool,
    ) -> serde_json::Value {
        let input: Vec<serde_json::Value> = messages.iter().map(Self::message_to_input).collect();

        let mut body = json!({
            "model": self.config.model,
            "input": input,
            "stream": stream,
        });

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        if let Some(max) = self.config.max_tokens {
            body["max_output_tokens"] = json!(max);
        }

        if let Some(top_p) = self.config.top_p {
            body["top_p"] = json!(top_p);
        }

        if !self.config.builtin_tools.is_empty() {
            let tools: Vec<serde_json::Value> = self
                .config
                .builtin_tools
                .iter()
                .map(|t| t.to_api_value())
                .collect();
            body["tools"] = json!(tools);
        }

        body
    }

    // -- Internal chat -------------------------------------------------------

    pub(crate) async fn chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<LLMResult, ResponsesError> {
        let url = format!("{}/responses", self.config.base_url);
        let body = self.build_request_body(messages, false);

        // 0.25.0: unified HTTP layer — retriable status set, Retry-After and
        // POST pre-dispatch-only retry semantics live in lc_core::http.
        let response = self
            .http_api
            .post_json_with(&url, &body, self.request_options())
            .await
            .map_err(Self::map_http_error)?;

        let api_response: ResponsesApiResponse =
            serde_json::from_str(&response.body).map_err(|e| {
                let preview: String = response.body.chars().take(200).collect();
                ResponsesError::Parse(format!("{e} - body: {preview}"))
            })?;

        Self::parse_response(api_response)
    }

    /// Extracts every callable item from a completed response: custom
    /// function calls plus the built-in hosted-tool calls.
    fn extract_tool_calls(output: &[ResponsesOutputItem]) -> Option<Vec<ToolCall>> {
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for item in output {
            match item {
                ResponsesOutputItem::FunctionCall(call) => {
                    tool_calls.push(
                        ToolCall::builder(&call.call_id)
                            .name(&call.name)
                            .arguments(&call.arguments)
                            .build(),
                    );
                }
                ResponsesOutputItem::WebSearchCall(call) => {
                    tool_calls.push(
                        ToolCall::builder(&call.id)
                            .name("web_search")
                            .arguments(
                                json!({
                                    "query": call.query,
                                    "status": &call.status,
                                })
                                .to_string(),
                            )
                            .build(),
                    );
                }
                ResponsesOutputItem::FileSearchCall(call) => {
                    tool_calls.push(
                        ToolCall::builder(&call.id)
                            .name("file_search")
                            .arguments(
                                json!({
                                    "query": call.query,
                                    "status": &call.status,
                                })
                                .to_string(),
                            )
                            .build(),
                    );
                }
                ResponsesOutputItem::CodeInterpreterCall(call) => {
                    tool_calls.push(
                        ToolCall::builder(&call.id)
                            .name("code_interpreter")
                            .arguments(
                                json!({
                                    "code": call.code,
                                    "results": call.results,
                                    "status": call.status,
                                })
                                .to_string(),
                            )
                            .build(),
                    );
                }
                ResponsesOutputItem::ComputerCall(call) => {
                    tool_calls.push(
                        ToolCall::builder(&call.id)
                            .name("computer_use")
                            .arguments(
                                json!({
                                    "action": call.action,
                                    "status": call.status,
                                })
                                .to_string(),
                            )
                            .build(),
                    );
                }
                // Text and reasoning items are not tool calls.
                ResponsesOutputItem::Message(_) | ResponsesOutputItem::Reasoning(_) => {}
            }
        }
        (!tool_calls.is_empty()).then_some(tool_calls)
    }

    /// Parse a completed Responses API response into an `LLMResult`.
    pub(crate) fn parse_response(
        api_response: ResponsesApiResponse,
    ) -> Result<LLMResult, ResponsesError> {
        let mut content = String::new();

        for item in &api_response.output {
            if let ResponsesOutputItem::Message(msg) = item {
                for part in &msg.content {
                    match part {
                        ResponsesContentPart::OutputText(text_part) => {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(&text_part.text);
                        }
                        ResponsesContentPart::Refusal(refusal) => {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(&format!("[Refusal: {}]", refusal.refusal));
                        }
                    }
                }
            }
        }

        let model = api_response.model.unwrap_or_else(|| "gpt-4o".to_string());

        let token_usage = api_response.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LLMResult {
            content,
            model,
            token_usage,
            tool_calls: Self::extract_tool_calls(&api_response.output),
            thinking_content: None,
        })
    }

    // -- Internal stream -----------------------------------------------------

    pub(crate) async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, ResponsesError>> + Send>>,
        ResponsesError,
    > {
        use crate::openai::sse::{SSEParser, SseByteFramer};
        use std::sync::{Arc, Mutex};

        let url = format!("{}/responses", self.config.base_url);
        let body = self.build_request_body(messages, true);

        // 0.25.0: unified HTTP layer. Retries cover establishment only; once
        // the response head arrives the stream runs without reconnecting.
        let byte_stream = self
            .http_sse
            .open_sse(&url, Some(&body), self.request_options())
            .await
            .map_err(Self::map_http_error)?;

        let parser = Arc::new(Mutex::new((SSEParser::new(), SseByteFramer::new())));
        // M18: Use bounded channel to prevent OOM with slow consumers
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, ResponsesError>>(64);

        let parser_clone = parser.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            // 0.22.0 audit fix (Medium): `[DONE]` must exit the outer loop.
            let mut done = false;
            // 0.25.0: require a terminal marker — response.completed,
            // response.failed, response.incomplete, error, or `[DONE]`. A
            // connection that closes first delivered a truncated answer;
            // reporting it as complete silently dropped model output.
            let mut saw_terminal = false;
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(ResponsesError::Http(e.to_string()))).await;
                        return;
                    }
                };

                // H41: unwrap_or_else recovers from a poisoned mutex.
                // 0.22.0 C1: byte-layer framing; only complete events decode.
                let events = {
                    let mut guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    let mut out = Vec::new();
                    for text in guard.1.push(&chunk_bytes) {
                        out.extend(guard.0.parse(&text));
                    }
                    out
                };
                // guard is dropped here, before any await

                for event in events {
                    if event.is_done() {
                        done = true;
                        saw_terminal = true;
                        break;
                    }
                    match serde_json::from_str::<ResponsesStreamEvent>(&event.data) {
                        Ok(stream_event) => match stream_event {
                            ResponsesStreamEvent::OutputTextDelta(delta) => {
                                if tx.send(Ok(StreamChunk::new(delta.delta))).await.is_err() {
                                    return;
                                }
                            }
                            ResponsesStreamEvent::OutputItemDone(item_done) => {
                                // Completed output items include custom
                                // function calls (previously rejected by the
                                // enum) and hosted-tool calls; emit them as
                                // their own chunk so streaming tool use is
                                // not lost the way the old fold did.
                                if let Some(tool_calls) =
                                    Self::extract_tool_calls(std::slice::from_ref(&item_done.item))
                                {
                                    if tx
                                        .send(Ok(StreamChunk {
                                            thinking_content: None,
                                            text: String::new(),
                                            token_usage: None,
                                            tool_calls: Some(tool_calls),
                                        }))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            ResponsesStreamEvent::Completed(completed) => {
                                // Terminal event; emit token usage when carried.
                                // (returns below, so the terminal guard is moot.)
                                if let Some(usage) = completed.response.usage {
                                    let token_usage = TokenUsage {
                                        prompt_tokens: usage.input_tokens,
                                        completion_tokens: usage.output_tokens,
                                        total_tokens: usage.total_tokens,
                                    };
                                    if tx
                                        .send(Ok(StreamChunk {
                                            thinking_content: None,
                                            text: String::new(),
                                            token_usage: Some(token_usage),
                                            tool_calls: None,
                                        }))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                return;
                            }
                            ResponsesStreamEvent::Failed(payload) => {
                                let _ = tx
                                    .send(Err(ResponsesError::Api(format!(
                                        "Response failed: {}",
                                        failed_event_message(&payload)
                                    ))))
                                    .await;
                                return;
                            }
                            ResponsesStreamEvent::Incomplete(payload) => {
                                let _ = tx
                                    .send(Err(ResponsesError::Api(format!(
                                        "Response incomplete: {}",
                                        incomplete_event_reason(&payload)
                                    ))))
                                    .await;
                                return;
                            }
                            ResponsesStreamEvent::Error(err) => {
                                // Top-level error events must never be
                                // swallowed; the model output is unusable.
                                let _ = tx
                                    .send(Err(ResponsesError::Api(format!(
                                        "Stream error: {}",
                                        err.message()
                                    ))))
                                    .await;
                                return;
                            }
                            // Other events are informational.
                            _ => {}
                        },
                        // Forward-compat: a future event type fails the tagged
                        // enum parse. Skip it rather than killing the stream;
                        // the terminal guard still detects truncation.
                        Err(e) => {
                            log::debug!("skipping unparsable Responses SSE event: {e}");
                        }
                    }
                }
                if done {
                    break;
                }
            }
            if !saw_terminal {
                let _ = tx
                    .send(Err(ResponsesError::StreamInterrupted(
                        "connection closed before a terminal response event".to_string(),
                    )))
                    .await;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

/// Best-effort message extraction from a `response.failed` event payload.
///
/// Real shape is `{"response":{"error":{"code","message"}}}`; tolerate a flat
/// `{"error":{"message"}}` from gateways.
fn failed_event_message(payload: &serde_json::Value) -> String {
    payload
        .get("response")
        .and_then(|r| r.get("error"))
        .or_else(|| payload.get("error"))
        .and_then(|e| e.get("message").and_then(|m| m.as_str()))
        .map(str::to_string)
        .unwrap_or_else(|| "unknown failure".to_string())
}

/// Best-effort reason extraction from a `response.incomplete` event payload:
/// `{"response":{"status":"incomplete","incomplete_details":{"reason":...}}}`.
fn incomplete_event_reason(payload: &serde_json::Value) -> String {
    payload
        .get("response")
        .and_then(|r| r.get("incomplete_details"))
        .and_then(|d| d.get("reason").and_then(|r| r.as_str()))
        .map(str::to_string)
        .unwrap_or_else(|| "unknown reason".to_string())
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for ResponsesModel {
    type Error = ResponsesError;

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
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let model = self.config.model.clone();
        let token_stream = self.stream_chat_internal(input).await?;

        // 0.25.0: pass chunks through as they arrive instead of folding the
        // whole stream into one buffered LLMResult (the old fold also dropped
        // usage and tool calls from every chunk).
        let stream = token_stream.map(move |result| {
            result.map(|chunk| LLMResult {
                content: chunk.text,
                model: model.clone(),
                token_usage: chunk.token_usage,
                tool_calls: chunk.tool_calls,
                thinking_content: chunk.thinking_content,
            })
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for ResponsesModel {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        lc_core::token_counter::count_tokens(text).unwrap_or_else(|e| {
            // If the encoder fails to load, overestimate by byte length (better slightly high than silently counting 0, which would mislead routing/truncation)
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
impl BaseChatModel for ResponsesModel {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:responses:chat", self.config.model));

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

        let result = self.chat_internal(messages.clone()).await;

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
            .unwrap_or_else(|| format!("{}:responses:stream", self.config.model));

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

        let stream = self.stream_chat_internal(messages).await?;

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
