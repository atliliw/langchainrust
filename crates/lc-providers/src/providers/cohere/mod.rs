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
use crate::ProviderError;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::Runnable;
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Cohere chat client.
///
/// Native implementation using Cohere's v2 chat API.
#[derive(Clone)]
pub struct CohereChat {
    config: CohereConfig,
    client: reqwest::Client,
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
            client: reqwest::Client::new(),
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

        body
    }

    /// Internal chat implementation.
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, CohereError> {
        let url = format!("{}/chat", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CohereError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CohereError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let chat_response: CohereChatResponse = response
            .json()
            .await
            .map_err(|e| CohereError::Parse(e.to_string()))?;

        let message = chat_response
            .message
            .ok_or_else(|| CohereError::Api("No message in response".to_string()))?;

        // Extract content from content array
        let content = message
            .content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

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
                            arguments: tc.function.arguments,
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
        use crate::openai::sse::SSEParser;
        use std::sync::{Arc, Mutex};

        let url = format!("{}/chat", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| CohereError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(CohereError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let byte_stream = response.bytes_stream();
        let parser = Arc::new(Mutex::new(SSEParser::new()));
        let parser_clone = parser.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, CohereError>>(64);

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(CohereError::Http(e.to_string()))).await;
                        return;
                    }
                };

                let events = {
                    let mut parser_guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    let chunk_str = String::from_utf8_lossy(&chunk_bytes);
                    parser_guard.parse(&chunk_str)
                };

                for event in events {
                    if event.is_done() {
                        break;
                    }
                    // Cohere v2 streaming uses the same SSE format as OpenAI
                    // with content deltas in choices[0].delta.content
                    // 解析失败的 SSE chunk 不再静默丢弃:记 error 日志,
                    // 避免流式回复因单条坏数据被截断却毫无提示
                    match event.parse_openai_chunk() {
                        Ok(Some(chunk)) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(content) = &choice.delta.content {
                                    if tx.send(Ok(StreamChunk::new(content))).await.is_err() {
                                        return;
                                    }
                                }
                            }
                            // Cohere 复用了 OpenAI SSE 解析:若末尾 chunk 携带 OpenAI 风格
                            // usage(prompt_tokens/completion_tokens/total_tokens),同样发出
                            // usage chunk;结构不匹配(如 Cohere 自带 tokens 嵌套)则保持 None。
                            if let Some(usage) = chunk.usage {
                                let token_usage = TokenUsage {
                                    prompt_tokens: usage.prompt_tokens,
                                    completion_tokens: usage.completion_tokens,
                                    total_tokens: usage.total_tokens,
                                };
                                let final_chunk = StreamChunk {
                                    text: String::new(),
                                    token_usage: Some(token_usage),
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
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
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
                tool_calls: None,
                thinking_content: None,
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
}
