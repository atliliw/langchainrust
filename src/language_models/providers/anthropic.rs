// src/language_models/providers/anthropic.rs
//! Anthropic Claude API implementation (native API format).

use async_trait::async_trait;
use futures_util::{Stream, StreamExt, FutureExt};
use std::pin::Pin;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

use crate::schema::Message;
use crate::RunnableConfig;
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::callbacks::{RunTree, RunType};

/// Anthropic API endpoint.
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Claude model list.
pub const CLAUDE_MODELS: [&str; 5] = [
    "claude-3-5-sonnet-20241022",  // Claude 3.5 Sonnet
    "claude-3-5-haiku-20241022",   // Claude 3.5 Haiku
    "claude-3-opus-20240229",      // Claude 3 Opus
    "claude-3-sonnet-20240229",    // Claude 3 Sonnet
    "claude-3-haiku-20240307",     // Claude 3 Haiku
];

/// Anthropic Claude configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: ANTHROPIC_BASE_URL.to_string(),
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: 4096,
            temperature: None,
            system_prompt: None,
        }
    }
}

impl AnthropicConfig {
    /// Creates a new AnthropicConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates an AnthropicConfig from environment variables.
    pub fn from_env() -> Self {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY environment variable not set");
        
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| ANTHROPIC_BASE_URL.to_string());
        
        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());
        
        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        Self {
            api_key,
            base_url,
            model,
            max_tokens,
            ..Default::default()
        }
    }

    /// Sets the Claude model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets the max tokens limit.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Sets the temperature parameter.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// Anthropic Claude chat client.
pub struct AnthropicChat {
    config: AnthropicConfig,
    client: reqwest::Client,
}

impl AnthropicChat {
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Self {
        Self::new(AnthropicConfig::from_env())
    }

    pub fn with_model(model: impl Into<String>) -> Self {
        Self::new(AnthropicConfig::from_env().with_model(model))
    }

    fn message_to_anthropic_format(message: &Message) -> AnthropicMessage {
        let role = match &message.message_type {
            crate::schema::MessageType::Human => "user",
            crate::schema::MessageType::AI => "assistant",
            _ => "user",
        };
        
        AnthropicMessage {
            role: role.to_string(),
            content: message.content.clone(),
        }
    }

    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| m.message_type != crate::schema::MessageType::System)
            .map(Self::message_to_anthropic_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            "stream": stream,
        });

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        if let Some(system) = &self.config.system_prompt {
            body["system"] = json!(system);
        }

        body
    }

    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, AnthropicError> {
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self.client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AnthropicError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AnthropicError::Parse(e.to_string()))?;

        let content = anthropic_response.content
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        Ok(LLMResult {
            content,
            model: anthropic_response.model,
            token_usage: anthropic_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
            tool_calls: None,
        })
    }

    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, AnthropicError>> + Send>>, AnthropicError> {
        use futures_util::StreamExt;
        
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self.client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AnthropicError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let byte_stream = response.bytes_stream();
        let stream = byte_stream
            .then(|chunk_result| async move {
                match chunk_result {
                    Ok(bytes) => {
                        let chunk_str = String::from_utf8_lossy(&bytes);
                        let mut content = String::new();
                        
                        for line in chunk_str.lines() {
                            if line.starts_with("data: ") {
                                let data = line.trim_start_matches("data: ");
                                if data == "[DONE]" {
                                    return None;
                                }
                                
                                if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                    if event.type_field == "content_block_delta" {
                                        if let Some(delta) = event.delta {
                                            if delta.type_field == "text_delta" {
                                                content.push_str(&delta.text);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if content.is_empty() {
                            None
                        } else {
                            Some(Ok(content))
                        }
                    },
                    Err(e) => Some(Err(AnthropicError::Http(e.to_string()))),
                }
            })
            .filter_map(|x| async move { x });

        Ok(Box::pin(stream))
    }
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    type_field: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    type_field: String,
    text: String,
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for AnthropicChat {
    type Error = AnthropicError;

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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error> {
        let model = self.config.model.clone();
        let token_stream = self.stream_chat_internal(input).await?;
        
        let content_future = async move {
            token_stream
                .fold(String::new(), |mut acc, token_result| async move {
                    if let Ok(token) = token_result {
                        acc.push_str(&token);
                    }
                    acc
                })
                .await
        };
        
        let stream = futures_util::stream::once(async move {
            let content = content_future.await;
            Ok(LLMResult {
                content,
                model,
                token_usage: None,
                tool_calls: None,
            })
        });
        
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for AnthropicChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }

    fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    fn max_tokens(&self) -> Option<usize> {
        Some(self.config.max_tokens)
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.config.max_tokens = max;
        self
    }
}

#[async_trait]
impl BaseChatModel for AnthropicChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let run_name = config.as_ref()
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
        let run_name = config.as_ref()
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

        let stream = self.stream_chat_internal(messages).await?;

        let callbacks = config.and_then(|c| c.callbacks);
        let stream = Box::pin(futures_util::stream::StreamExt::map(stream, move |token_result| {
            if let Some(ref cbs) = callbacks {
                if let Ok(ref token) = token_result {
                    for handler in cbs.handlers() {
                        let _ = handler.on_llm_new_token(&run, token);
                    }
                }
            }
            token_result
        }));

        Ok(stream)
    }
}

#[derive(Debug)]
pub enum AnthropicError {
    Http(String),
    Api(String),
    Parse(String),
}

impl std::fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnthropicError::Http(msg) => write!(f, "HTTP error: {}", msg),
            AnthropicError::Api(msg) => write!(f, "API error: {}", msg),
            AnthropicError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for AnthropicError {}