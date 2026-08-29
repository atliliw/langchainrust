// lc-providers/src/openai/chat/mod.rs
//! OpenAI chat model implementation.

mod error;
mod structured;
#[cfg(test)]
mod tests;

pub use error::OpenAIError;
pub use structured::StructuredOutputMethod;

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use std::marker::PhantomData;
use std::pin::Pin;

use super::OpenAIConfig;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolDefinition;
use lc_core::RunnableConfig;
use lc_schema::Message;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

/// OpenAI chat client for GPT models.
#[derive(Clone)]
pub struct OpenAIChat {
    pub(crate) config: OpenAIConfig,
    pub(crate) client: reqwest::Client,
}

impl OpenAIChat {
    /// Creates a new OpenAIChat with the given configuration.
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates an OpenAIChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, OpenAIError> {
        let config = OpenAIConfig::from_env_result()?;
        Ok(Self::new(config))
    }

    /// Converts a Message to OpenAI API format.
    fn message_to_openai_format(message: &Message) -> serde_json::Value {
        match &message.message_type {
            lc_schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            lc_schema::MessageType::Human => {
                if message.has_images() {
                    let mut content = vec![json!({"type": "text", "text": &message.content})];
                    for img in &message.images {
                        content.push(json!({"type": "image_url", "image_url": {"url": &img.url}}));
                    }
                    json!({"role": "user", "content": content})
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
                        serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
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

    /// Builds the API request body.
    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(Self::message_to_openai_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "messages": openai_messages,
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

    /// Binds tool definitions for function calling.
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        let config = OpenAIConfig {
            tools: Some(tools),
            ..self.config.clone()
        };
        Self {
            config,
            client: self.client.clone(),
        }
    }

    /// Sets the tool choice strategy.
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.config.tool_choice = Some(choice.into());
        self
    }

    /// Enables structured JSON output with schema validation.
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> StructuredOutputMethod<T> {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T)).unwrap_or_else(|_| {
            // H64: Schema generation should not silently produce null
            serde_json::json!({"type": "object", "properties": {}})
        });

        let tool = ToolDefinition::new("structured_output", "Return structured JSON output")
            .with_parameters(schema)
            .with_strict(true);

        let config = OpenAIConfig {
            tools: Some(vec![tool]),
            tool_choice: Some("auto".to_string()),
            ..self.config.clone()
        };

        StructuredOutputMethod {
            config,
            client: self.client.clone(),
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for OpenAIChat {
    type Error = OpenAIError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, _config).await
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

        // H4: True streaming — emit one LLMResult per token instead of
        // collecting all tokens first and emitting a single result.
        let stream = token_stream.map(move |token_result| match token_result {
            Ok(chunk) => Ok(LLMResult {
                content: chunk.text,
                model: model.clone(),
                token_usage: chunk.token_usage,
                tool_calls: None,
                thinking_content: None,
            }),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for OpenAIChat {
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
impl BaseChatModel for OpenAIChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("{}:chat", self.config.model));

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

        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_tokens = Some(m);
        }

        // Q4: honor `config.streaming` — aggregate the streaming token stream
        // into a single LLMResult instead of ignoring the field.
        let result = if effective.config.streaming {
            let stream = effective.stream_chat_internal(messages.clone()).await?;
            let content = Self::aggregate_stream(stream).await?;
            Ok(LLMResult {
                content,
                model: effective.config.model.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        } else {
            effective.chat_internal(messages.clone()).await
        };

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

impl OpenAIChat {
    pub(crate) async fn chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<LLMResult, OpenAIError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| OpenAIError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OpenAIError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let chat_response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| OpenAIError::Parse(e.to_string()))?;

        let choice = chat_response
            .choices
            .first()
            .ok_or_else(|| OpenAIError::Api("No choices in response".to_string()))?;
        let message = &choice.message;

        Ok(Self::llm_result_from_message(
            message,
            chat_response.model,
            chat_response.usage,
        ))
    }

    /// Builds the `LLMResult` from a parsed response message (Q3).
    ///
    /// Thinking models (glm-5.2, DeepSeek-R1) may return an empty `content` with
    /// the actual reasoning in `reasoning_content`. `content` stays empty in that
    /// case — it is never filled from `reasoning_content` — and the reasoning only
    /// goes into `thinking_content`.
    fn llm_result_from_message(
        message: &OpenAIMessage,
        model: String,
        usage: Option<OpenAIUsage>,
    ) -> LLMResult {
        let content = message
            .content
            .clone()
            .filter(|c| !c.is_empty())
            .unwrap_or_default();

        let thinking_content = message.reasoning_content.clone().filter(|c| !c.is_empty());

        LLMResult {
            content,
            model,
            token_usage: usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            tool_calls: message.tool_calls.clone(),
            thinking_content,
        }
    }

    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>>, OpenAIError>
    {
        use super::sse::{SSEParser, StreamToolCallAccumulator};
        use std::sync::{Arc, Mutex};

        let url = format!("{}/chat/completions", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| OpenAIError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OpenAIError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let byte_stream = response.bytes_stream();

        let parser = Arc::new(Mutex::new(SSEParser::new()));

        let parser_clone = parser.clone();
        // M18: Use bounded channel to prevent OOM with slow consumers
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, OpenAIError>>(64);

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut byte_stream = byte_stream;
            // 0.20.0 S3.2: accumulate streaming tool_calls deltas so the terminal
            // chunk carries complete tool calls (previously dropped, which made
            // tool-call steps fall back to non-streaming plan() in lc-agents).
            let mut tool_acc = StreamToolCallAccumulator::default();
            let mut tool_calls_emitted = false;
            while let Some(chunk_result) = byte_stream.next().await {
                // H2 fix: propagate network errors to the consumer
                // Must be done OUTSIDE the mutex scope to avoid Send issue
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(OpenAIError::Http(e.to_string()))).await;
                        return;
                    }
                };

                let events = {
                    let mut parser_guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    let chunk_str = String::from_utf8_lossy(&chunk_bytes);
                    parser_guard.parse(&chunk_str)
                };
                // parser_guard is dropped here, before any await

                for event in events {
                    if event.is_done() {
                        break;
                    }
                    // Failed SSE chunks are no longer silently dropped: log an error,
                    // so a streaming reply truncated by one bad datum is not left unexplained.
                    match event.parse_openai_chunk() {
                        Ok(Some(chunk)) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    if tx.send(Ok(StreamChunk::new(content))).await.is_err() {
                                        return;
                                    }
                                }
                                if let Some(deltas) = &choice.delta.tool_calls {
                                    for delta in deltas {
                                        tool_acc.push(delta);
                                    }
                                }
                            }
                            // OpenAI carries usage at the end of the stream (usually in the
                            // last chunk before `[DONE]`). Emit it as a standalone chunk: empty
                            // text, token_usage filled, so the consumer gets the whole call's
                            // token usage from the streaming path — and, 0.20.0 S3.2, the
                            // complete tool_calls accumulated so far, so tool-call steps
                            // stream natively.
                            if let Some(usage) = chunk.usage {
                                let token_usage = TokenUsage {
                                    prompt_tokens: usage.prompt_tokens,
                                    completion_tokens: usage.completion_tokens,
                                    total_tokens: usage.total_tokens,
                                };
                                let tool_calls = tool_acc.build();
                                if !tool_calls.is_empty() {
                                    tool_calls_emitted = true;
                                }
                                let final_chunk = StreamChunk {
                                    text: String::new(),
                                    token_usage: Some(token_usage),
                                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                                };
                                if tx.send(Ok(final_chunk)).await.is_err() {
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
            }
            // Some compatible providers end the stream without a usage chunk. If tool
            // calls were accumulated but never emitted, flush them as a dedicated
            // terminal chunk so the streaming path never loses them (0.20.0 S3.2).
            if !tool_calls_emitted {
                let tool_calls = tool_acc.build();
                if !tool_calls.is_empty() {
                    let _ = tx
                        .send(Ok(StreamChunk {
                            text: String::new(),
                            token_usage: None,
                            tool_calls: Some(tool_calls),
                        }))
                        .await;
                }
            }
        });
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        Ok(Box::pin(stream))
    }

    /// Aggregates a token stream into a single string (Q4).
    ///
    /// This is the piece that makes `config.streaming` observable: the
    /// non-streaming `chat()` path consumes the token stream through here.
    async fn aggregate_stream(
        mut stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, OpenAIError>> + Send>>,
    ) -> Result<String, OpenAIError> {
        use futures_util::StreamExt;
        let mut content = String::new();
        while let Some(item) = stream.next().await {
            content.push_str(&item?.text);
        }
        Ok(content)
    }
}

/// OpenAI response structure
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIChoice {
    index: i32,
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
    /// Reasoning chain-of-thought content from reasoning models (e.g. glm-5.2, DeepSeek-R1)
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<lc_core::tools::ToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}
