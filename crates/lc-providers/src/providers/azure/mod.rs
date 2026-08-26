// lc-providers/src/providers/azure/mod.rs
//! Azure OpenAI API implementation.
//!
//! Azure OpenAI provides the same models as OpenAI but through Azure's infrastructure,
//! with different URL structure and authentication.
//!
//! # URL Format
//!
//! ```text
//! {endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}
//! ```
//!
//! # Authentication
//!
//! Uses `api-key` header instead of `Bearer` token.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_providers::providers::{AzureOpenAIChat, AzureOpenAIConfig};
//!
//! let config = AzureOpenAIConfig::new(
//!     "https://myresource.openai.azure.com",
//!     "my-gpt4-deployment",
//!     "your-api-key",
//! );
//! let llm = AzureOpenAIChat::new(config);
//! let result = llm.chat(messages, None).await?;
//! ```

mod config;
mod error;
#[cfg(test)]
mod tests;
mod types;

pub use config::{AzureOpenAIConfig, AZURE_DEFAULT_API_VERSION};
pub use error::AzureOpenAIError;

use async_trait::async_trait;
use futures_util::Stream;
use serde_json::json;
use std::pin::Pin;

use self::types::*;
use crate::openai::sse::SSEParser;
use crate::ProviderError;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::Runnable;
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Azure OpenAI chat client.
///
/// Unlike Mistral/DeepSeek which wrap `OpenAIChat`, Azure OpenAI requires
/// a separate implementation because:
/// - Different URL format (deployment-based)
/// - Different authentication header (`api-key` instead of `Bearer`)
/// - Model name comes from deployment, not request body
#[derive(Clone)]
pub struct AzureOpenAIChat {
    config: AzureOpenAIConfig,
    client: reqwest::Client,
}

impl std::fmt::Debug for AzureOpenAIChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAIChat")
            .field("deployment", &self.config.deployment_name)
            .finish_non_exhaustive()
    }
}

impl AzureOpenAIChat {
    /// Creates a new AzureOpenAIChat with the given configuration.
    pub fn new(config: AzureOpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates an AzureOpenAIChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(AzureOpenAIConfig::from_env_result()?))
    }

    /// Converts a Message to OpenAI API format (same as OpenAI).
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
    /// Note: Azure OpenAI does not include `model` in the request body
    /// (it's determined by the deployment URL).
    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(Self::message_to_openai_format)
            .collect();

        let mut body = json!({
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

        body
    }

    /// Internal chat implementation (no callback overhead).
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, AzureOpenAIError> {
        let url = self.config.chat_url();
        let body = self.build_request_body(messages, false);

        let response = self
            .client
            .post(&url)
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AzureOpenAIError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AzureOpenAIError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let chat_response: AzureChatResponse = response
            .json()
            .await
            .map_err(|e| AzureOpenAIError::Parse(e.to_string()))?;

        let choice = chat_response
            .choices
            .first()
            .ok_or_else(|| AzureOpenAIError::Api("No choices in response".to_string()))?;
        let message = &choice.message;

        let content = message
            .content
            .clone()
            .filter(|c| !c.is_empty())
            .unwrap_or_default();

        Ok(LLMResult {
            content,
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

    /// Internal streaming implementation.
    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamChunk, AzureOpenAIError>> + Send>>,
        AzureOpenAIError,
    > {
        use std::sync::{Arc, Mutex};

        let url = self.config.chat_url();
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AzureOpenAIError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AzureOpenAIError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let byte_stream = response.bytes_stream();
        let parser = Arc::new(Mutex::new(SSEParser::new()));
        let parser_clone = parser.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, AzureOpenAIError>>(64);

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk_bytes = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(Err(AzureOpenAIError::Http(e.to_string()))).await;
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
                            // Azure OpenAI 与 OpenAI 同构:末尾 chunk 携带 usage。
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for AzureOpenAIChat {
    fn model_name(&self) -> &str {
        self.config.effective_model()
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
impl Runnable<Vec<Message>, LLMResult> for AzureOpenAIChat {
    type Error = AzureOpenAIError;

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

        let model = self.config.effective_model().to_string();
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
impl BaseChatModel for AzureOpenAIChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config
            .as_ref()
            .and_then(|c| c.run_name.clone())
            .unwrap_or_else(|| format!("azure-{}:chat", self.config.deployment_name));

        let mut run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
                "deployment": self.config.deployment_name,
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
            .unwrap_or_else(|| format!("azure-{}:stream", self.config.deployment_name));

        let run = RunTree::new(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.len(),
                "deployment": self.config.deployment_name,
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
