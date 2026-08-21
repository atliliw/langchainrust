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

use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolCall;
use lc_core::RunnableConfig;
use lc_schema::Message;

use super::types::{
    BuiltinTool, ResponsesApiResponse, ResponsesConfig, ResponsesContentPart, ResponsesError,
    ResponsesOutputItem, ResponsesStreamEvent,
};

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
    pub(crate) fn parse_response(
        api_response: ResponsesApiResponse,
    ) -> Result<LLMResult, ResponsesError> {
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

    pub(crate) async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, ResponsesError>> + Send>>, ResponsesError>
    {
        use crate::openai::sse::SSEParser;
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
        lc_core::token_counter::count_tokens(text).unwrap_or_else(|e| {
            // 编码器加载失败时按字节数高估(宁可略高,不静默按 0 算导致路由/截断误判)
            log::warn!("token 计数失败,回退为按字节数估算: {e}");
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
