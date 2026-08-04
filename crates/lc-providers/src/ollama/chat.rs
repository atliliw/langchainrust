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
use crate::openai::sse::SSEParser;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use lc_core::runnables::Runnable;
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
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new OllamaChat with a custom configuration.
    ///
    /// # Arguments
    /// * `config` - A pre-configured OllamaConfig instance.
    pub fn with_config(config: OllamaConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates an OllamaChat from environment variables.
    #[deprecated(
        since = "0.7.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Self {
        Self::from_env_result().unwrap_or_else(|_| Self::with_config(OllamaConfig::default()))
    }

    /// Creates an OllamaChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, String> {
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
        let body = self.build_request_body(messages, false);

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
            content: message.content.clone(),
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, OllamaError>> + Send>>, OllamaError> {
        let url = format!("{}/chat/completions", self.config.base_url);
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
        let parser = Arc::new(Mutex::new(SSEParser::new()));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, OllamaError>>(64);

        let parser_clone = parser.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
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
                    let mut parser_guard = parser_clone.lock().unwrap_or_else(|e| e.into_inner());
                    let chunk_str = String::from_utf8_lossy(&chunk_bytes);
                    parser_guard.parse(&chunk_str)
                };
                // parser_guard is dropped here, before any await

                for event in events {
                    if event.is_done() {
                        return;
                    }
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
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        use futures_util::StreamExt;

        let model = self.config.model.clone();
        let token_stream = self.stream_chat_internal(input).await?;

        // H4: True streaming — emit one LLMResult per token
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for OllamaChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        lc_core::token_counter::count_tokens(text)
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

#[derive(Debug)]
pub enum OllamaError {
    Http(String),
    Api(String),
    Parse(String),
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaError::Http(msg) => write!(f, "HTTP error: {}", msg),
            OllamaError::Api(msg) => write!(f, "API error: {}", msg),
            OllamaError::Parse(msg) => write!(f, "Parse error: {}", msg),
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
    content: String,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub struct OllamaStructuredOutput<T: DeserializeOwned + JsonSchema> {
    config: OllamaConfig,
    client: reqwest::Client,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> OllamaStructuredOutput<T> {
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
    use crate::ENV_TEST_LOCK;
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
