// lc-providers/src/providers/azure.rs
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

use async_trait::async_trait;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::pin::Pin;

use crate::openai::sse::SSEParser;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use lc_core::runnables::Runnable;
use lc_core::tools::ToolCall;
use lc_core::RunnableConfig;
use lc_schema::Message;

/// Azure OpenAI API version.
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-02-15-preview";

/// Azure OpenAI configuration.
#[derive(Debug, Clone)]
pub struct AzureOpenAIConfig {
    /// Azure OpenAI resource endpoint (e.g., https://myresource.openai.azure.com).
    pub endpoint: String,
    /// Deployment name (e.g., "my-gpt4-deployment").
    pub deployment_name: String,
    /// API key for authentication.
    pub api_key: String,
    /// API version string (default: 2024-02-15-preview).
    pub api_version: String,
    /// Model name for LLMResult metadata (optional, defaults to deployment_name).
    pub model: Option<String>,
    /// Temperature for generation.
    pub temperature: Option<f32>,
    /// Maximum tokens for generation.
    pub max_tokens: Option<usize>,
    /// Top-p for nucleus sampling.
    pub top_p: Option<f32>,
}

impl AzureOpenAIConfig {
    /// Creates a new AzureOpenAIConfig.
    pub fn new(
        endpoint: impl Into<String>,
        deployment_name: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            deployment_name: deployment_name.into(),
            api_key: api_key.into(),
            api_version: AZURE_DEFAULT_API_VERSION.to_string(),
            model: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
        }
    }

    /// Creates an AzureOpenAIConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `AZURE_OPENAI_ENDPOINT`: Resource endpoint (required)
    /// - `AZURE_OPENAI_DEPLOYMENT_NAME`: Deployment name (required)
    /// - `AZURE_OPENAI_API_KEY`: API key (required)
    /// - `AZURE_OPENAI_API_VERSION`: API version (optional)
    /// - `AZURE_OPENAI_MODEL`: Model name override (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let endpoint = env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| "AZURE_OPENAI_ENDPOINT environment variable not set".to_string())?;

        let deployment_name = env::var("AZURE_OPENAI_DEPLOYMENT_NAME")
            .map_err(|_| "AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set".to_string())?;

        let api_key = env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| "AZURE_OPENAI_API_KEY environment variable not set".to_string())?;

        let api_version = env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| AZURE_DEFAULT_API_VERSION.to_string());

        let model = env::var("AZURE_OPENAI_MODEL").ok();

        Ok(Self {
            endpoint,
            deployment_name,
            api_key,
            api_version,
            model,
            temperature: None,
            max_tokens: None,
            top_p: None,
        })
    }

    /// Sets the API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Sets the model name for metadata.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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

    /// Sets the top-p.
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Builds the chat completions URL.
    fn chat_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment_name,
            self.api_version,
        )
    }

    /// Returns the effective model name.
    fn effective_model(&self) -> &str {
        self.model.as_deref().unwrap_or(&self.deployment_name)
    }
}

/// Azure OpenAI error type.
#[derive(Debug)]
pub enum AzureOpenAIError {
    /// HTTP request error.
    Http(String),
    /// API error (non-2xx response).
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for AzureOpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AzureOpenAIError::Http(msg) => write!(f, "Azure OpenAI HTTP error: {}", msg),
            AzureOpenAIError::Api(msg) => write!(f, "Azure OpenAI API error: {}", msg),
            AzureOpenAIError::Parse(msg) => write!(f, "Azure OpenAI parse error: {}", msg),
        }
    }
}

impl std::error::Error for AzureOpenAIError {}

impl From<String> for AzureOpenAIError {
    fn from(s: String) -> Self {
        AzureOpenAIError::Api(s)
    }
}

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
    pub fn from_env_result() -> Result<Self, String> {
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
        Pin<Box<dyn Stream<Item = Result<String, AzureOpenAIError>> + Send>>,
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
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, AzureOpenAIError>>(64);

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
impl BaseLanguageModel<Vec<Message>, LLMResult> for AzureOpenAIChat {
    fn model_name(&self) -> &str {
        self.config.effective_model()
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
        let _ = config; // Reserved for future use (e.g., stop conditions, metadata)

        let model = self.config.effective_model().to_string();
        let token_stream = self.stream_chat_internal(input).await?;

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

// ---- Response types (same structure as OpenAI) ----

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AzureChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<AzureChoice>,
    usage: Option<AzureUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AzureChoice {
    index: i32,
    message: AzureMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AzureMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct AzureUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
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
        let config = AzureOpenAIConfig::new(
            "https://myresource.openai.azure.com",
            "gpt-4-deployment",
            "test-key",
        );
        assert_eq!(config.endpoint, "https://myresource.openai.azure.com");
        assert_eq!(config.deployment_name, "gpt-4-deployment");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.api_version, AZURE_DEFAULT_API_VERSION);
    }

    #[test]
    fn test_config_builder() {
        let config = AzureOpenAIConfig::new("https://res.openai.azure.com", "deploy", "key")
            .with_api_version("2024-06-01")
            .with_model("gpt-4o")
            .with_temperature(0.5)
            .with_max_tokens(2048)
            .with_top_p(0.9);
        assert_eq!(config.api_version, "2024-06-01");
        assert_eq!(config.model, Some("gpt-4o".to_string()));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn test_config_from_env_result_ok() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_ep = save_and_set("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com");
        let old_dn = save_and_set("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy");
        let old_key = save_and_set("AZURE_OPENAI_API_KEY", "azure-key-123");
        let result = AzureOpenAIConfig::from_env_result();
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.endpoint, "https://test.openai.azure.com");
        assert_eq!(config.deployment_name, "my-deploy");
        assert_eq!(config.api_key, "azure-key-123");
        restore("AZURE_OPENAI_ENDPOINT", old_ep);
        restore("AZURE_OPENAI_DEPLOYMENT_NAME", old_dn);
        restore("AZURE_OPENAI_API_KEY", old_key);
    }

    #[test]
    fn test_config_from_env_result_err_when_missing() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_ep = std::env::var("AZURE_OPENAI_ENDPOINT").ok();
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
        let result = AzureOpenAIConfig::from_env_result();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("AZURE_OPENAI_ENDPOINT"));
        restore("AZURE_OPENAI_ENDPOINT", old_ep);
    }

    #[test]
    fn test_chat_url() {
        let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com", "gpt4", "key");
        let url = config.chat_url();
        assert!(url.contains("myresource.openai.azure.com"));
        assert!(url.contains("/openai/deployments/gpt4/chat/completions"));
        assert!(url.contains("api-version="));
    }

    #[test]
    fn test_chat_url_trailing_slash() {
        let config = AzureOpenAIConfig::new("https://myresource.openai.azure.com/", "gpt4", "key");
        let url = config.chat_url();
        // Should not have double slashes
        assert!(!url.contains("//openai"));
    }

    #[test]
    fn test_effective_model() {
        let config_no_model = AzureOpenAIConfig::new("https://ep", "deploy", "key");
        assert_eq!(config_no_model.effective_model(), "deploy");

        let config_with_model =
            AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
        assert_eq!(config_with_model.effective_model(), "gpt-4o");
    }

    #[test]
    fn test_chat_new() {
        let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
        let _chat = AzureOpenAIChat::new(config);
    }

    #[test]
    fn test_model_name() {
        let config = AzureOpenAIConfig::new("https://ep", "deploy", "key").with_model("gpt-4o");
        let chat = AzureOpenAIChat::new(config);
        assert_eq!(chat.model_name(), "gpt-4o");
    }

    #[test]
    fn test_build_request_body_no_model() {
        let config = AzureOpenAIConfig::new("https://ep", "deploy", "key");
        let chat = AzureOpenAIChat::new(config);
        let body = chat.build_request_body(vec![Message::human("hello")], false);
        // Azure request body should NOT contain "model" field
        assert!(body.get("model").is_none());
        assert!(body.get("messages").is_some());
    }

    #[test]
    fn test_error_display() {
        let err = AzureOpenAIError::Http("timeout".to_string());
        assert!(err.to_string().contains("HTTP error"));
        let err = AzureOpenAIError::Api("rate limit".to_string());
        assert!(err.to_string().contains("API error"));
        let err = AzureOpenAIError::Parse("bad json".to_string());
        assert!(err.to_string().contains("parse error"));
    }
}
