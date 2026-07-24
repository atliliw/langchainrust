// src/language_models/openai/responses.rs
//! OpenAI Responses API model implementation.
//!
//! This module implements the `/v1/responses` endpoint, which provides
//! access to OpenAI's built-in tools (web search, file search, code
//! interpreter, computer use) alongside standard chat capabilities.

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use std::pin::Pin;

use crate::callbacks::{RunTree, RunType};
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::core::tools::ToolCall;
use crate::schema::Message;
use crate::RunnableConfig;

// ---------------------------------------------------------------------------
// Builtin tools
// ---------------------------------------------------------------------------

/// Built-in tools available through the Responses API.
#[derive(Debug, Clone)]
pub enum BuiltinTool {
    /// Web search via OpenAI.
    WebSearch,
    /// File search over vector stores.
    FileSearch { vector_store_ids: Vec<String> },
    /// Code interpreter sandbox.
    CodeInterpreter,
    /// Computer use (GUI automation).
    ComputerUse {
        display_width: Option<u32>,
        display_height: Option<u32>,
    },
}

impl BuiltinTool {
    /// Convert to the JSON value expected by the Responses API `tools` field.
    fn to_api_value(&self) -> serde_json::Value {
        match self {
            BuiltinTool::WebSearch => json!({"type": "web_search"}),
            BuiltinTool::FileSearch { vector_store_ids } => json!({
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }),
            BuiltinTool::CodeInterpreter => json!({"type": "code_interpreter"}),
            BuiltinTool::ComputerUse {
                display_width,
                display_height,
            } => {
                let mut val = json!({"type": "computer_use"});
                if let Some(w) = display_width {
                    val["display_width"] = json!(w);
                }
                if let Some(h) = display_height {
                    val["display_height"] = json!(h);
                }
                val
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Responses API model.
#[derive(Debug, Clone)]
pub struct ResponsesConfig {
    /// API key.
    pub api_key: String,
    /// Model name (e.g. "gpt-4o").
    pub model: String,
    /// Base URL (defaults to `https://api.openai.com/v1`).
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<usize>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f32>,
    /// Built-in tools to include in every request.
    pub builtin_tools: Vec<BuiltinTool>,
}

impl Default for ResponsesConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            builtin_tools: Vec::new(),
        }
    }
}

impl ResponsesConfig {
    /// Create a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` and
    /// `OPENAI_MODEL` (optional).
    pub fn from_env() -> Result<Self, ResponsesError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            ResponsesError::Api("OPENAI_API_KEY environment variable must be set".to_string())
        })?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the max output tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Add a built-in tool.
    pub fn with_builtin_tool(mut self, tool: BuiltinTool) -> Self {
        self.builtin_tools.push(tool);
        self
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the Responses API model.
#[derive(Debug)]
pub enum ResponsesError {
    /// HTTP transport error.
    Http(String),
    /// API returned a non-success status.
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for ResponsesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponsesError::Http(msg) => write!(f, "HTTP error: {}", msg),
            ResponsesError::Api(msg) => write!(f, "API error: {}", msg),
            ResponsesError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ResponsesError {}

// ---------------------------------------------------------------------------
// Response structures
// ---------------------------------------------------------------------------

/// Top-level response from the `/v1/responses` endpoint.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponsesApiResponse {
    id: String,
    object: Option<String>,
    model: Option<String>,
    output: Vec<ResponsesOutputItem>,
    usage: Option<ResponsesUsage>,
}

/// A single item in the `output` array.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesOutputItem {
    /// A text message from the model.
    #[serde(rename = "message")]
    Message(ResponsesMessage),
    /// A web search call.
    #[serde(rename = "web_search_call")]
    WebSearchCall(ResponsesWebSearchCall),
    /// A file search call.
    #[serde(rename = "file_search_call")]
    FileSearchCall(ResponsesFileSearchCall),
    /// A code interpreter call.
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall(ResponsesCodeInterpreterCall),
    /// A computer use call.
    #[serde(rename = "computer_call")]
    ComputerCall(ResponsesComputerCall),
}

/// Message output item.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponsesMessage {
    id: Option<String>,
    role: Option<String>,
    content: Vec<ResponsesContentPart>,
    status: Option<String>,
}

/// Content part inside a message.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponsesContentPart {
    #[serde(rename = "output_text")]
    OutputText(ResponsesOutputText),
    #[serde(rename = "refusal")]
    Refusal(ResponsesRefusal),
}

/// Text content part.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponsesOutputText {
    text: String,
    #[serde(default)]
    annotations: Vec<serde_json::Value>,
}

/// Refusal content part.
#[derive(Debug, Deserialize)]
struct ResponsesRefusal {
    refusal: String,
}

/// Web search call output.
#[derive(Debug, Deserialize)]
struct ResponsesWebSearchCall {
    id: String,
    status: String,
    query: Option<String>,
}

/// File search call output.
#[derive(Debug, Deserialize)]
struct ResponsesFileSearchCall {
    id: String,
    status: String,
    query: Option<String>,
}

/// Code interpreter call output.
#[derive(Debug, Deserialize)]
struct ResponsesCodeInterpreterCall {
    id: String,
    code: Option<String>,
    results: Option<Vec<serde_json::Value>>,
    status: Option<String>,
}

/// Computer use call output.
#[derive(Debug, Deserialize)]
struct ResponsesComputerCall {
    id: String,
    action: Option<serde_json::Value>,
    status: Option<String>,
}

/// Token usage from the Responses API.
#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    input_tokens: usize,
    output_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Stream structures
// ---------------------------------------------------------------------------

/// SSE event types for the Responses API streaming.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(tag = "type")]
enum ResponsesStreamEvent {
    /// Emitted when the response object is created.
    #[serde(rename = "response.created")]
    Created(serde_json::Value),
    /// In-progress output item added.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(serde_json::Value),
    /// Content part added to an output item.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(serde_json::Value),
    /// Text delta for output_text.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(ResponsesTextDelta),
    /// Text content part completed.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(serde_json::Value),
    /// Content part completed.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(serde_json::Value),
    /// Output item completed.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(serde_json::Value),
    /// Web search call in progress.
    #[serde(rename = "response.web_search_call.in_progress")]
    WebSearchInProgress(serde_json::Value),
    /// Web search call searching.
    #[serde(rename = "response.web_search_call.searching")]
    WebSearchSearching(serde_json::Value),
    /// Web search call completed.
    #[serde(rename = "response.web_search_call.completed")]
    WebSearchCompleted(serde_json::Value),
    /// Code interpreter call in progress.
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    CodeInterpreterInProgress(serde_json::Value),
    /// Code interpreter call code delta.
    #[serde(rename = "response.code_interpreter_call.code_delta")]
    CodeInterpreterCodeDelta(serde_json::Value),
    /// Code interpreter call completed.
    #[serde(rename = "response.code_interpreter_call.completed")]
    CodeInterpreterCompleted(serde_json::Value),
    /// File search call in progress.
    #[serde(rename = "response.file_search_call.in_progress")]
    FileSearchInProgress(serde_json::Value),
    /// File search call completed.
    #[serde(rename = "response.file_search_call.completed")]
    FileSearchCompleted(serde_json::Value),
    /// Response completed.
    #[serde(rename = "response.completed")]
    Completed(ResponsesCompletedEvent),
    /// Response failed.
    #[serde(rename = "response.failed")]
    Failed(serde_json::Value),
}

/// Text delta payload.
#[derive(Debug, Deserialize)]
struct ResponsesTextDelta {
    delta: String,
}

/// Completed event payload (contains full response).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ResponsesCompletedEvent {
    response: ResponsesApiResponse,
}

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
    config: ResponsesConfig,
    client: reqwest::Client,
}

impl ResponsesModel {
    /// Create a new model with the given configuration.
    pub fn new(config: ResponsesConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
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
    fn message_to_input(message: &Message) -> serde_json::Value {
        match &message.message_type {
            crate::schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            crate::schema::MessageType::Human => {
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
            crate::schema::MessageType::AI => {
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
            crate::schema::MessageType::Tool { tool_call_id } => json!({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": message.content,
            }),
        }
    }

    // -- Request body --------------------------------------------------------

    /// Build the JSON request body for the Responses API.
    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
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

    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, ResponsesError> {
        let url = format!("{}/responses", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ResponsesError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ResponsesError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let api_response: ResponsesApiResponse = response
            .json()
            .await
            .map_err(|e| ResponsesError::Parse(e.to_string()))?;

        Self::parse_response(api_response)
    }

    /// Parse a completed Responses API response into an `LLMResult`.
    fn parse_response(api_response: ResponsesApiResponse) -> Result<LLMResult, ResponsesError> {
        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for item in &api_response.output {
            match item {
                ResponsesOutputItem::Message(msg) => {
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
                ResponsesOutputItem::WebSearchCall(call) => {
                    tool_calls.push(ToolCall::new(
                        &call.id,
                        "web_search",
                        json!({
                            "query": call.query,
                            "status": &call.status,
                        })
                        .to_string(),
                    ));
                }
                ResponsesOutputItem::FileSearchCall(call) => {
                    tool_calls.push(ToolCall::new(
                        &call.id,
                        "file_search",
                        json!({
                            "query": call.query,
                            "status": &call.status,
                        })
                        .to_string(),
                    ));
                }
                ResponsesOutputItem::CodeInterpreterCall(call) => {
                    tool_calls.push(ToolCall::new(
                        &call.id,
                        "code_interpreter",
                        json!({
                            "code": call.code,
                            "results": call.results,
                            "status": call.status,
                        })
                        .to_string(),
                    ));
                }
                ResponsesOutputItem::ComputerCall(call) => {
                    tool_calls.push(ToolCall::new(
                        &call.id,
                        "computer_use",
                        json!({
                            "action": call.action,
                            "status": call.status,
                        })
                        .to_string(),
                    ));
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
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking_content: None,
        })
    }

    // -- Internal stream -----------------------------------------------------

    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, ResponsesError>> + Send>>, ResponsesError>
    {
        use super::sse::SSEParser;
        use std::sync::{Arc, Mutex};

        let url = format!("{}/responses", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ResponsesError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ResponsesError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let byte_stream = response.bytes_stream();
        let parser = Arc::new(Mutex::new(SSEParser::new()));
        // M18: Use bounded channel to prevent OOM with slow consumers
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, ResponsesError>>(64);

        let parser_clone = parser.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                // H41: Use unwrap_or_else to recover from poisoned mutex
                let events = {
                    let mut parser_guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    if let Ok(bytes) = chunk_result {
                        let chunk_str = String::from_utf8_lossy(&bytes);

                        parser_guard.parse(&chunk_str)
                    } else {
                        Vec::new()
                    }
                };
                // parser_guard is dropped here, before any await

                for event in events {
                    if event.is_done() {
                        let _ = tx.send(Ok(String::new())).await;
                        return;
                    }
                    // Try to parse as a Responses API stream event
                    if let Ok(stream_event) =
                        serde_json::from_str::<ResponsesStreamEvent>(&event.data)
                    {
                        match stream_event {
                            ResponsesStreamEvent::OutputTextDelta(delta) => {
                                if tx.send(Ok(delta.delta)).await.is_err() {
                                    return;
                                }
                            }
                            ResponsesStreamEvent::Completed(_completed) => {
                                // Final event — nothing more to emit
                                let _ = tx.send(Ok(String::new())).await;
                                return;
                            }
                            ResponsesStreamEvent::Failed(_) => {
                                let _ = tx
                                    .send(Err(ResponsesError::Api("Response failed".to_string())))
                                    .await;
                                return;
                            }
                            // Other events are informational; skip them
                            _ => {}
                        }
                    }
                    // If the event data is not a recognized stream event
                    // (e.g. an error payload), we silently skip it.
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
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

        let stream = futures_util::stream::once(async move {
            let content = token_stream
                .fold(String::new(), |mut acc, token_result| async move {
                    if let Ok(token) = token_result {
                        acc.push_str(&token);
                    }
                    acc
                })
                .await;
            Ok(LLMResult {
                content,
                model,
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
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
        crate::core::token_counter::count_tokens(text)
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

        let mut run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
                "model": self.config.model,
            }),
        );

        if let Some(ref cfg) = config {
            for tag in &cfg.tags {
                run = run.with_tag(tag.clone());
            }
            for (key, value) in &cfg.metadata {
                run = run.with_metadata(key.clone(), value.clone());
            }
        }

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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        use futures_util::StreamExt;
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:responses:stream", self.config.model));

        let run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.len(),
                "model": self.config.model,
            }),
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
                            handler.on_llm_new_token(&run, token).await;
                        }
                    }
                }
                token_result
            }
        });

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_tool_to_api_value() {
        let web_search = BuiltinTool::WebSearch;
        assert_eq!(web_search.to_api_value(), json!({"type": "web_search"}));

        let file_search = BuiltinTool::FileSearch {
            vector_store_ids: vec!["vs_123".to_string()],
        };
        assert_eq!(
            file_search.to_api_value(),
            json!({"type": "file_search", "vector_store_ids": ["vs_123"]})
        );

        let code_interp = BuiltinTool::CodeInterpreter;
        assert_eq!(
            code_interp.to_api_value(),
            json!({"type": "code_interpreter"})
        );

        let computer = BuiltinTool::ComputerUse {
            display_width: Some(1024),
            display_height: Some(768),
        };
        let cv = computer.to_api_value();
        assert_eq!(cv["type"], "computer_use");
        assert_eq!(cv["display_width"], 1024);
        assert_eq!(cv["display_height"], 768);

        let computer_no_dims = BuiltinTool::ComputerUse {
            display_width: None,
            display_height: None,
        };
        let cv2 = computer_no_dims.to_api_value();
        assert_eq!(cv2["type"], "computer_use");
        assert!(cv2.get("display_width").is_none());
        assert!(cv2.get("display_height").is_none());
    }

    #[test]
    fn test_config_builder() {
        let config = ResponsesConfig::new("sk-test")
            .with_model("gpt-4o")
            .with_base_url("https://api.openai.com/v1")
            .with_temperature(0.7)
            .with_max_tokens(1024)
            .with_builtin_tool(BuiltinTool::WebSearch)
            .with_builtin_tool(BuiltinTool::CodeInterpreter);

        assert_eq!(config.api_key, "sk-test");
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(config.builtin_tools.len(), 2);
    }

    #[test]
    fn test_message_to_input_system() {
        let msg = Message::system("You are helpful.");
        let val = ResponsesModel::message_to_input(&msg);
        assert_eq!(val["role"], "system");
        assert_eq!(val["content"], "You are helpful.");
    }

    #[test]
    fn test_message_to_input_human() {
        let msg = Message::human("Hello!");
        let val = ResponsesModel::message_to_input(&msg);
        assert_eq!(val["role"], "user");
        assert_eq!(val["content"], "Hello!");
    }

    #[test]
    fn test_message_to_input_ai() {
        let msg = Message::ai("Hi there!");
        let val = ResponsesModel::message_to_input(&msg);
        assert_eq!(val["role"], "assistant");
        assert_eq!(val["content"], "Hi there!");
    }

    #[test]
    fn test_message_to_input_tool() {
        let msg = Message::tool("call_123", "result data");
        let val = ResponsesModel::message_to_input(&msg);
        assert_eq!(val["type"], "function_call_output");
        assert_eq!(val["call_id"], "call_123");
        assert_eq!(val["output"], "result data");
    }

    #[test]
    fn test_build_request_body_basic() {
        let config = ResponsesConfig::new("sk-test").with_model("gpt-4o");
        let model = ResponsesModel::new(config);
        let messages = vec![
            Message::system("Be helpful."),
            Message::human("What is 2+2?"),
        ];
        let body = model.build_request_body(messages, false);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "user");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let config = ResponsesConfig::new("sk-test")
            .with_model("gpt-4o")
            .with_builtin_tool(BuiltinTool::WebSearch)
            .with_builtin_tool(BuiltinTool::FileSearch {
                vector_store_ids: vec!["vs_abc".to_string()],
            });
        let model = ResponsesModel::new(config);
        let messages = vec![Message::human("Search the web")];
        let body = model.build_request_body(messages, false);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["type"], "web_search");
        assert_eq!(tools[1]["type"], "file_search");
    }

    #[test]
    fn test_build_request_body_with_options() {
        let config = ResponsesConfig::new("sk-test")
            .with_model("gpt-4o")
            .with_temperature(0.5)
            .with_max_tokens(2048);
        let model = ResponsesModel::new(config);
        let messages = vec![Message::human("Hi")];
        let body = model.build_request_body(messages, true);

        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_output_tokens"], 2048);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_parse_response_text_only() {
        let api_response = ResponsesApiResponse {
            id: "resp_001".to_string(),
            object: Some("response".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![ResponsesOutputItem::Message(ResponsesMessage {
                id: Some("msg_001".to_string()),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                    text: "Hello, world!".to_string(),
                    annotations: vec![],
                })],
                status: Some("completed".to_string()),
            })],
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            }),
        };

        let result = ResponsesModel::parse_response(api_response).unwrap();
        assert_eq!(result.content, "Hello, world!");
        assert_eq!(result.model, "gpt-4o");
        assert!(result.tool_calls.is_none());
        let usage = result.token_usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_response_with_web_search() {
        let api_response = ResponsesApiResponse {
            id: "resp_002".to_string(),
            object: Some("response".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![
                ResponsesOutputItem::WebSearchCall(ResponsesWebSearchCall {
                    id: "ws_001".to_string(),
                    status: "completed".to_string(),
                    query: Some("Rust programming".to_string()),
                }),
                ResponsesOutputItem::Message(ResponsesMessage {
                    id: Some("msg_002".to_string()),
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                        text: "Rust is a systems language.".to_string(),
                        annotations: vec![],
                    })],
                    status: Some("completed".to_string()),
                }),
            ],
            usage: None,
        };

        let result = ResponsesModel::parse_response(api_response).unwrap();
        assert_eq!(result.content, "Rust is a systems language.");
        let tc = result.tool_calls.unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "ws_001");
        assert_eq!(tc[0].function.name, "web_search");
    }

    #[test]
    fn test_parse_response_with_code_interpreter() {
        let api_response = ResponsesApiResponse {
            id: "resp_003".to_string(),
            object: Some("response".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![
                ResponsesOutputItem::CodeInterpreterCall(ResponsesCodeInterpreterCall {
                    id: "ci_001".to_string(),
                    code: Some("print(2+2)".to_string()),
                    results: Some(vec![json!({"type": "text", "text": "4"})]),
                    status: Some("completed".to_string()),
                }),
                ResponsesOutputItem::Message(ResponsesMessage {
                    id: Some("msg_003".to_string()),
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContentPart::OutputText(ResponsesOutputText {
                        text: "The answer is 4.".to_string(),
                        annotations: vec![],
                    })],
                    status: Some("completed".to_string()),
                }),
            ],
            usage: None,
        };

        let result = ResponsesModel::parse_response(api_response).unwrap();
        assert_eq!(result.content, "The answer is 4.");
        let tc = result.tool_calls.unwrap();
        assert_eq!(tc[0].function.name, "code_interpreter");
    }

    #[test]
    fn test_parse_response_refusal() {
        let api_response = ResponsesApiResponse {
            id: "resp_004".to_string(),
            object: Some("response".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![ResponsesOutputItem::Message(ResponsesMessage {
                id: Some("msg_004".to_string()),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::Refusal(ResponsesRefusal {
                    refusal: "I cannot do that.".to_string(),
                })],
                status: Some("completed".to_string()),
            })],
            usage: None,
        };

        let result = ResponsesModel::parse_response(api_response).unwrap();
        assert!(result.content.contains("[Refusal: I cannot do that.]"));
    }

    #[test]
    fn test_parse_response_multiple_content_parts() {
        let api_response = ResponsesApiResponse {
            id: "resp_005".to_string(),
            object: Some("response".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![ResponsesOutputItem::Message(ResponsesMessage {
                id: Some("msg_005".to_string()),
                role: Some("assistant".to_string()),
                content: vec![
                    ResponsesContentPart::OutputText(ResponsesOutputText {
                        text: "Part 1".to_string(),
                        annotations: vec![],
                    }),
                    ResponsesContentPart::OutputText(ResponsesOutputText {
                        text: "Part 2".to_string(),
                        annotations: vec![],
                    }),
                ],
                status: Some("completed".to_string()),
            })],
            usage: None,
        };

        let result = ResponsesModel::parse_response(api_response).unwrap();
        assert_eq!(result.content, "Part 1\nPart 2");
    }

    #[test]
    fn test_error_display() {
        let http_err = ResponsesError::Http("connection refused".to_string());
        assert_eq!(format!("{}", http_err), "HTTP error: connection refused");

        let api_err = ResponsesError::Api("rate limited".to_string());
        assert_eq!(format!("{}", api_err), "API error: rate limited");

        let parse_err = ResponsesError::Parse("invalid json".to_string());
        assert_eq!(format!("{}", parse_err), "Parse error: invalid json");
    }

    #[test]
    fn test_stream_event_deserialize_text_delta() {
        let data = r#"{"type":"response.output_text.delta","delta":"Hello"}"#;
        let event: ResponsesStreamEvent = serde_json::from_str(data).unwrap();
        match event {
            ResponsesStreamEvent::OutputTextDelta(delta) => {
                assert_eq!(delta.delta, "Hello");
            }
            _ => panic!("Expected OutputTextDelta"),
        }
    }

    #[test]
    fn test_stream_event_deserialize_completed() {
        let data = r#"{
            "type": "response.completed",
            "response": {
                "id": "resp_001",
                "model": "gpt-4o",
                "output": [],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(data).unwrap();
        match event {
            ResponsesStreamEvent::Completed(completed) => {
                assert_eq!(completed.response.id, "resp_001");
            }
            _ => panic!("Expected Completed"),
        }
    }

    #[test]
    fn test_with_builtin_tool_returns_new_instance() {
        let config = ResponsesConfig::new("sk-test");
        let model = ResponsesModel::new(config);
        assert_eq!(model.config.builtin_tools.len(), 0);

        let model2 = model.with_builtin_tool(BuiltinTool::WebSearch);
        assert_eq!(model2.config.builtin_tools.len(), 1);

        let model3 = model2.with_builtin_tool(BuiltinTool::CodeInterpreter);
        assert_eq!(model3.config.builtin_tools.len(), 2);
    }

    #[test]
    fn test_base_language_model_trait() {
        let config = ResponsesConfig::new("sk-test")
            .with_model("gpt-4o")
            .with_temperature(0.7)
            .with_max_tokens(512);
        let model = ResponsesModel::new(config);

        assert_eq!(model.model_name(), "gpt-4o");
        assert_eq!(model.temperature(), Some(0.7));
        assert_eq!(model.max_tokens(), Some(512));
        assert!(model.get_num_tokens("hello world") > 0);
    }

    #[test]
    fn test_with_temperature_and_max_tokens() {
        let config = ResponsesConfig::new("sk-test");
        let model = ResponsesModel::new(config);
        assert_eq!(model.temperature(), None);
        assert_eq!(model.max_tokens(), None);

        let model2 = model.with_temperature(0.3);
        assert_eq!(model2.temperature(), Some(0.3));

        let model3 = model2.with_max_tokens(1024);
        assert_eq!(model3.max_tokens(), Some(1024));
    }
}
