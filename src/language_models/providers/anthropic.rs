// src/language_models/providers/anthropic.rs
//! Anthropic Claude API implementation (native API format).
//!
//! Supports extended thinking via the `ThinkingConfig` / `with_thinking()` API.
//! When thinking is enabled, the request includes a `thinking` parameter and
//! the response may contain both "thinking" and "text" content blocks.

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::callbacks::{RunTree, RunType};
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::schema::Message;
use crate::RunnableConfig;

/// Anthropic API endpoint.
pub const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Claude model list.
pub const CLAUDE_MODELS: [&str; 5] = [
    "claude-3-5-sonnet-20241022", // Claude 3.5 Sonnet
    "claude-3-5-haiku-20241022",  // Claude 3.5 Haiku
    "claude-3-opus-20240229",     // Claude 3 Opus
    "claude-3-sonnet-20240229",   // Claude 3 Sonnet
    "claude-3-haiku-20240307",    // Claude 3 Haiku
];

/// Type of extended thinking mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThinkingType {
    /// Extended thinking enabled with a token budget.
    Enabled,
    /// Extended thinking disabled (default).
    #[default]
    Disabled,
}

/// Configuration for Anthropic extended thinking.
///
/// When enabled, the model emits a "thinking" content block before the
/// final text answer, allowing callers to observe the reasoning process.
#[derive(Debug, Clone)]
pub struct ThinkingConfig {
    /// Maximum number of tokens the model may spend on thinking.
    pub budget_tokens: usize,
    /// Whether thinking is enabled or disabled.
    pub r#type: ThinkingType,
}

impl ThinkingConfig {
    /// Create a new enabled thinking config with the given budget.
    pub fn enabled(budget_tokens: usize) -> Self {
        Self {
            budget_tokens,
            r#type: ThinkingType::Enabled,
        }
    }

    /// Create a disabled thinking config.
    pub fn disabled() -> Self {
        Self {
            budget_tokens: 0,
            r#type: ThinkingType::Disabled,
        }
    }

    /// Returns true if thinking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.r#type == ThinkingType::Enabled
    }
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Anthropic Claude configuration.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub system_prompt: Option<String>,
    /// Extended thinking configuration.
    pub thinking: ThinkingConfig,
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
            thinking: ThinkingConfig::default(),
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
    pub fn from_env() -> Result<Self, String> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY environment variable not set".to_string())?;

        let base_url =
            env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| ANTHROPIC_BASE_URL.to_string());

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());

        let max_tokens = env::var("ANTHROPIC_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        Ok(Self {
            api_key,
            base_url,
            model,
            max_tokens,
            ..Default::default()
        })
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

    /// Enables extended thinking with the given token budget.
    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = thinking;
        self
    }
}

/// Anthropic Claude chat client.
#[derive(Clone)]
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

    pub fn from_env() -> Result<Self, String> {
        Ok(Self::new(AnthropicConfig::from_env()?))
    }

    pub fn with_model(model: impl Into<String>) -> Result<Self, String> {
        Ok(Self::new(AnthropicConfig::from_env()?.with_model(model)))
    }

    /// Enables extended thinking with the given token budget.
    pub fn with_thinking(mut self, budget_tokens: usize) -> Self {
        self.config.thinking = ThinkingConfig::enabled(budget_tokens);
        self
    }

    /// Returns a reference to the thinking configuration.
    pub fn thinking_config(&self) -> &ThinkingConfig {
        &self.config.thinking
    }

    fn message_to_anthropic_format(message: &Message) -> AnthropicMessage {
        match &message.message_type {
            crate::schema::MessageType::Human => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text(message.content.clone()),
            },
            crate::schema::MessageType::AI => {
                let mut content_parts: Vec<AnthropicContentBlock> = vec![];
                if let Some(tool_calls) = &message.tool_calls {
                    for tc in tool_calls {
                        content_parts.push(AnthropicContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(json!({})),
                        });
                    }
                }
                if !message.content.is_empty() {
                    content_parts.push(AnthropicContentBlock::Text {
                        text: message.content.clone(),
                    });
                }
                if content_parts.is_empty() {
                    content_parts.push(AnthropicContentBlock::Text {
                        text: String::new(),
                    });
                }
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: AnthropicMessageContent::Blocks(content_parts),
                }
            }
            crate::schema::MessageType::Tool { tool_call_id } => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_call_id.clone(),
                    content: message.content.clone(),
                }]),
            },
            // System messages are handled separately in build_request_body
            crate::schema::MessageType::System => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text(message.content.clone()),
            },
        }
    }

    fn build_request_body(&self, messages: Vec<Message>, stream: bool) -> serde_json::Value {
        // H42: Extract system messages into top-level system field
        let mut system_text = String::new();
        let mut non_system_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if msg.message_type == crate::schema::MessageType::System {
                if !system_text.is_empty() {
                    system_text.push('\n');
                }
                system_text.push_str(&msg.content);
            } else {
                non_system_messages.push(msg);
            }
        }

        // Also include config system_prompt if set
        if let Some(ref prompt) = self.config.system_prompt {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(prompt);
        }

        let anthropic_messages: Vec<AnthropicMessage> = non_system_messages
            .iter()
            .map(Self::message_to_anthropic_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            "stream": stream,
        });

        if !system_text.is_empty() {
            body["system"] = json!(system_text);
        }

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        // M17: Validate thinking config - max_tokens must be > budget_tokens
        if self.config.thinking.is_enabled() {
            if self.config.max_tokens <= self.config.thinking.budget_tokens {
                body["max_tokens"] = json!(self.config.thinking.budget_tokens + 1024);
            }
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": self.config.thinking.budget_tokens,
            });
        }

        body
    }

    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, AnthropicError> {
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self
            .client
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
            return Err(AnthropicError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AnthropicError::Parse(e.to_string()))?;

        let mut thinking_content = String::new();
        let mut text_content = String::new();

        for block in &anthropic_response.content {
            match block.content_type.as_str() {
                "thinking" => {
                    thinking_content.push_str(&block.thinking);
                }
                "text" => {
                    text_content.push_str(&block.text);
                }
                _ => {
                    if !block.text.is_empty() {
                        text_content.push_str(&block.text);
                    }
                }
            }
        }

        if text_content.is_empty() {
            text_content = anthropic_response
                .content
                .iter()
                .filter(|c| c.content_type == "text")
                .map(|c| c.text.clone())
                .collect::<Vec<_>>()
                .join("");
        }

        if text_content.is_empty() {
            text_content = anthropic_response
                .content
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default();
        }

        Ok(LLMResult {
            content: text_content,
            model: anthropic_response.model,
            token_usage: anthropic_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
            tool_calls: None,
            thinking_content: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
        })
    }

    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<AnthropicStreamToken, AnthropicError>> + Send>>,
        AnthropicError,
    > {
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self
            .client
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
            return Err(AnthropicError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let byte_stream = response.bytes_stream();
        let sse_buffer = Arc::new(Mutex::new(String::new()));
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<AnthropicStreamToken, AnthropicError>>(64);

        let buffer_clone = sse_buffer.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                if let Ok(bytes) = chunk_result {
                    let chunk_str = String::from_utf8_lossy(&bytes);

                    // Extract complete SSE events from buffer
                    let events = {
                        let mut buffer_guard =
                            buffer_clone.lock().unwrap_or_else(|e| e.into_inner());
                        buffer_guard.push_str(&chunk_str);

                        let mut events = Vec::new();
                        while let Some(pos) = buffer_guard.find("\n\n") {
                            let event_text = buffer_guard[..pos].to_string();
                            buffer_guard.drain(..=pos + 1);
                            events.push(event_text);
                        }
                        events
                    };
                    // buffer_guard is dropped here, before any await

                    for event_text in events {
                        for line in event_text.lines() {
                            if line.starts_with("data: ") {
                                let data = line.trim_start_matches("data: ");
                                if data == "[DONE]" {
                                    continue;
                                }

                                if let Ok(event) =
                                    serde_json::from_str::<AnthropicStreamEvent>(data)
                                {
                                    if event.type_field == "content_block_delta" {
                                        if let Some(delta) = event.delta {
                                            match delta.type_field.as_str() {
                                                "text_delta" => {
                                                    if !delta.text.is_empty()
                                                        && tx
                                                            .send(Ok(AnthropicStreamToken::Text(
                                                                delta.text,
                                                            )))
                                                            .await
                                                            .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                                "thinking_delta" => {
                                                    if !delta.thinking.is_empty()
                                                        && tx
                                                            .send(Ok(
                                                                AnthropicStreamToken::Thinking(
                                                                    delta.thinking,
                                                                ),
                                                            ))
                                                            .await
                                                            .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

/// A token emitted during streaming, distinguishing between thinking and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicStreamToken {
    /// A text content token (the final answer).
    Text(String),
    /// A thinking content token (extended reasoning).
    Thinking(String),
}

/// Content for an Anthropic message, supporting both simple text and structured content arrays.
#[derive(Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content array with multiple blocks.
    Blocks(Vec<AnthropicContentBlock>),
}

/// A single content block in an Anthropic message.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Text content block.
    #[serde(rename = "text")]
    Text { text: String },
    /// Tool use content block (from assistant).
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result content block (from user, responding to tool use).
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    thinking: String,
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
    #[serde(default)]
    text: String,
    #[serde(default)]
    thinking: String,
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let model = self.config.model.clone();
        let token_stream = self.stream_chat_internal(input).await?;

        // H4: True streaming — emit one LLMResult per token
        let stream = token_stream.map(move |token_result| match token_result {
            Ok(AnthropicStreamToken::Thinking(t)) => Ok(LLMResult {
                content: String::new(),
                model: model.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: Some(t),
            }),
            Ok(AnthropicStreamToken::Text(t)) => Ok(LLMResult {
                content: t,
                model: model.clone(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
            Err(e) => Err(e),
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
        crate::core::token_counter::count_tokens(text)
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

        let result = self.chat_internal(messages.clone()).await;

        match result {
            Ok(response) => {
                run.end(json!({
                    "content": &response.content,
                    "model": &response.model,
                    "token_usage": &response.token_usage,
                    "thinking_content": &response.thinking_content,
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

        let stream = self.stream_chat_internal(messages).await?;

        let callbacks = config.and_then(|c| c.callbacks);
        let stream = stream.then(move |token_result| {
            let cbs = callbacks.clone();
            let run = run.clone();
            async move {
                match &token_result {
                    Ok(AnthropicStreamToken::Text(token)) => {
                        if let Some(ref cbs) = cbs {
                            for handler in cbs.handlers() {
                                handler.on_llm_new_token(&run, token).await;
                            }
                        }
                    }
                    Ok(AnthropicStreamToken::Thinking(thinking)) => {
                        if let Some(ref cbs) = cbs {
                            for handler in cbs.handlers() {
                                handler.on_llm_thinking(&run, thinking).await;
                            }
                        }
                    }
                    Err(_) => {}
                }
                token_result
            }
        });

        // Flatten: emit Text tokens as Ok(String), drop Thinking tokens from the stream
        let stream = stream.flat_map(|token_result| {
            futures_util::stream::iter(match token_result {
                Ok(AnthropicStreamToken::Text(token)) => vec![Ok(token)],
                Ok(AnthropicStreamToken::Thinking(_)) => vec![],
                Err(e) => vec![Err(e)],
            })
        });

        Ok(Box::pin(stream))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_type_default() {
        assert_eq!(ThinkingType::default(), ThinkingType::Disabled);
    }

    #[test]
    fn test_thinking_config_enabled() {
        let config = ThinkingConfig::enabled(10000);
        assert!(config.is_enabled());
        assert_eq!(config.budget_tokens, 10000);
    }

    #[test]
    fn test_thinking_config_disabled() {
        let config = ThinkingConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_anthropic_config_with_thinking() {
        let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(5000));
        assert!(config.thinking.is_enabled());
        assert_eq!(config.thinking.budget_tokens, 5000);
    }

    #[test]
    fn test_anthropic_config_default_no_thinking() {
        let config = AnthropicConfig::default();
        assert!(!config.thinking.is_enabled());
    }

    #[test]
    fn test_anthropic_chat_with_thinking() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config).with_thinking(8000);
        assert!(chat.thinking_config().is_enabled());
        assert_eq!(chat.thinking_config().budget_tokens, 8000);
    }

    #[test]
    fn test_build_request_body_without_thinking() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config);
        let body = chat.build_request_body(vec![], false);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_build_request_body_with_thinking() {
        let config = AnthropicConfig::new("test-key").with_thinking(ThinkingConfig::enabled(10000));
        let chat = AnthropicChat::new(config);
        let body = chat.build_request_body(vec![], false);

        let thinking = body.get("thinking").expect("thinking should be present");
        assert_eq!(thinking["type"], "enabled");
        assert_eq!(thinking["budget_tokens"], 10000);
    }

    #[test]
    fn test_anthropic_content_deserialize_thinking() {
        let json = r#"{"type": "thinking", "thinking": "Let me analyze..."}"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.content_type, "thinking");
        assert_eq!(content.thinking, "Let me analyze...");
    }

    #[test]
    fn test_anthropic_content_deserialize_text() {
        let json = r#"{"type": "text", "text": "The answer is 42."}"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.content_type, "text");
        assert_eq!(content.text, "The answer is 42.");
    }

    #[test]
    fn test_anthropic_delta_deserialize_thinking_delta() {
        let json = r#"{"type": "thinking_delta", "thinking": "Hmm..."}"#;
        let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.type_field, "thinking_delta");
        assert_eq!(delta.thinking, "Hmm...");
    }

    #[test]
    fn test_anthropic_delta_deserialize_text_delta() {
        let json = r#"{"type": "text_delta", "text": "Hello"}"#;
        let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.type_field, "text_delta");
        assert_eq!(delta.text, "Hello");
    }
}
