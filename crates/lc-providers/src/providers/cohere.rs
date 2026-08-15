// lc-providers/src/providers/cohere.rs
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

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::pin::Pin;

use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use lc_core::runnables::Runnable;
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Cohere API endpoint.
pub const COHERE_BASE_URL: &str = "https://api.cohere.com/v2";

/// Cohere model list.
pub const COHERE_MODELS: [&str; 4] = ["command-r-plus", "command-r", "command", "command-light"];

/// Cohere configuration.
#[derive(Debug, Clone)]
pub struct CohereConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub preamble: Option<String>,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: COHERE_BASE_URL.to_string(),
            model: "command-r-plus".to_string(),
            temperature: None,
            max_tokens: None,
            preamble: None,
        }
    }
}

impl CohereConfig {
    /// Creates a new CohereConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a CohereConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `COHERE_API_KEY`: API key (required)
    /// - `COHERE_BASE_URL`: API endpoint (optional)
    /// - `COHERE_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = env::var("COHERE_API_KEY")
            .map_err(|_| "COHERE_API_KEY environment variable not set".to_string())?;

        let base_url = env::var("COHERE_BASE_URL").unwrap_or_else(|_| COHERE_BASE_URL.to_string());

        let model = env::var("COHERE_MODEL").unwrap_or_else(|_| "command-r-plus".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Sets the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the max tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Sets the preamble (system prompt).
    pub fn with_preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }
}

/// Cohere error type.
#[derive(Debug)]
pub enum CohereError {
    /// HTTP request error.
    Http(String),
    /// API error (non-2xx response).
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for CohereError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CohereError::Http(msg) => write!(f, "Cohere HTTP error: {}", msg),
            CohereError::Api(msg) => write!(f, "Cohere API error: {}", msg),
            CohereError::Parse(msg) => write!(f, "Cohere parse error: {}", msg),
        }
    }
}

impl std::error::Error for CohereError {}

impl From<String> for CohereError {
    fn from(s: String) -> Self {
        CohereError::Api(s)
    }
}

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
    pub fn from_env_result() -> Result<Self, String> {
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, CohereError>> + Send>>, CohereError> {
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
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, CohereError>>(64);

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
                    if let Ok(Some(chunk)) = event.parse_openai_chunk() {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                if tx.send(Ok(content.clone())).await.is_err() {
                                    return;
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

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for CohereChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        lc_core::token_counter::count_tokens(text).unwrap_or(0)
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
            Ok(token) => Ok(LLMResult {
                content: token,
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
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

// ---- Cohere v2 Response Types ----

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CohereChatResponse {
    id: String,
    model: String,
    message: Option<CohereMessage>,
    usage: Option<CohereUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CohereMessage {
    role: String,
    content: Vec<CohereContentPart>,
    #[serde(default)]
    tool_calls: Vec<CohereToolCall>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CohereContentPart {
    r#type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CohereToolCall {
    id: String,
    r#type: String,
    function: CohereFunctionCall,
}

#[derive(Debug, Deserialize)]
struct CohereFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct CohereUsage {
    tokens: CohereTokenUsage,
}

#[derive(Debug, Deserialize)]
struct CohereTokenUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ENV_TEST_LOCK;

    fn save_and_set(key: &str, value: &str) -> Option<String> {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        old
    }

    fn restore(key: &str, old: Option<String>) {
        match old {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn test_config_new() {
        let config = CohereConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, COHERE_BASE_URL);
        assert_eq!(config.model, "command-r-plus");
    }

    #[test]
    fn test_config_builder() {
        let config = CohereConfig::new("key")
            .with_model("command-r")
            .with_base_url("https://custom.cohere.com/v2")
            .with_temperature(0.5)
            .with_max_tokens(1024)
            .with_preamble("You are a helpful assistant.");
        assert_eq!(config.model, "command-r");
        assert_eq!(config.base_url, "https://custom.cohere.com/v2");
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(1024));
        assert_eq!(
            config.preamble,
            Some("You are a helpful assistant.".to_string())
        );
    }

    #[test]
    fn test_config_from_env_result_ok() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = save_and_set("COHERE_API_KEY", "env-key");
        let result = CohereConfig::from_env_result();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().api_key, "env-key");
        restore("COHERE_API_KEY", old);
    }

    #[test]
    fn test_config_from_env_result_err_when_missing() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::env::var("COHERE_API_KEY").ok();
        std::env::remove_var("COHERE_API_KEY");
        let result = CohereConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("COHERE_API_KEY"));
        restore("COHERE_API_KEY", old);
    }

    #[test]
    fn test_chat_new() {
        let config = CohereConfig::new("test-key");
        let _chat = CohereChat::new(config);
    }

    #[test]
    fn test_model_name() {
        let config = CohereConfig::new("key").with_model("command-r");
        let chat = CohereChat::new(config);
        assert_eq!(chat.model_name(), "command-r");
    }

    #[test]
    fn test_build_request_body() {
        let config = CohereConfig::new("key").with_preamble("System prompt");
        let chat = CohereChat::new(config);
        let body = chat.build_request_body(vec![Message::human("hello")], false);
        assert_eq!(body["model"], "command-r-plus");
        assert!(body.get("messages").is_some());
        assert_eq!(body["preamble"], "System prompt");
    }

    #[test]
    fn test_error_display() {
        let err = CohereError::Http("timeout".to_string());
        assert!(err.to_string().contains("HTTP error"));
        let err = CohereError::Api("rate limit".to_string());
        assert!(err.to_string().contains("API error"));
        let err = CohereError::Parse("bad json".to_string());
        assert!(err.to_string().contains("parse error"));
    }

    #[test]
    fn test_message_to_cohere_format_human() {
        let msg = Message::human("Hello");
        let formatted = CohereChat::message_to_cohere_format(&msg);
        assert_eq!(formatted["role"], "user");
        assert_eq!(formatted["content"], "Hello");
    }

    #[test]
    fn test_message_to_cohere_format_system() {
        let msg = Message::system("You are helpful");
        let formatted = CohereChat::message_to_cohere_format(&msg);
        assert_eq!(formatted["role"], "system");
    }
}
