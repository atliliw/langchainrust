// src/language_models/openai/chat.rs
//! OpenAI chat model implementation.

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use serde::Deserialize;
use serde_json::json;

use crate::schema::Message;
use crate::RunnableConfig;
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::core::tools::{ToolDefinition, StructuredOutput};
use crate::callbacks::{RunTree, RunType};
use super::OpenAIConfig;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

/// OpenAI chat client for GPT models.
pub struct OpenAIChat {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIChat {
    /// Creates a new OpenAIChat with the given configuration.
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
    
    /// Creates an OpenAIChat from environment variables.
    pub fn from_env() -> Self {
        Self::new(OpenAIConfig::from_env())
    }
    
    /// Converts a Message to OpenAI API format.
    fn message_to_openai_format(message: &Message) -> serde_json::Value {
        match &message.message_type {
            crate::schema::MessageType::System => json!({
                "role": "system",
                "content": message.content,
            }),
            crate::schema::MessageType::Human => json!({
                "role": "user",
                "content": message.content,
            }),
            crate::schema::MessageType::AI => {
                let mut msg = json!({
                    "role": "assistant",
                    "content": message.content,
                });
                if let Some(tool_calls) = &message.tool_calls {
                    msg["tool_calls"] = serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
                }
                msg
            },
            crate::schema::MessageType::Tool { tool_call_id } => json!({
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
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(&self) -> StructuredOutputMethod<T> {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T))
            .unwrap_or(serde_json::Value::Null);
        
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

/// Method for structured output calls
pub struct StructuredOutputMethod<T: DeserializeOwned + JsonSchema> {
    config: OpenAIConfig,
    client: reqwest::Client,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> StructuredOutputMethod<T> {
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<T, OpenAIError> {
        let chat = OpenAIChat {
            config: self.config.clone(),
            client: self.client.clone(),
        };
        
        let result = chat.chat_internal(messages).await?;
        let structured = StructuredOutput::<T>::new(result);
        structured.parse().map_err(|e| OpenAIError::Parse(e.to_string()))
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
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error> {
        use futures_util::StreamExt;
        
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for OpenAIChat {
    fn model_name(&self) -> &str {
        &self.config.model
    }
    
    fn get_num_tokens(&self, text: &str) -> usize {
        // 简单估算：平均每个 token 约 4 个字符
        // 实际应用中应该使用 tiktoken 或类似的库
        text.len() / 4
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

impl OpenAIChat {
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, OpenAIError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let body = self.build_request_body(messages, false);
        
        let response = self.client
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
        
        let message = &chat_response.choices[0].message;
        
        Ok(LLMResult {
            content: message.content.clone().unwrap_or_default(),
            model: chat_response.model,
            token_usage: chat_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            tool_calls: message.tool_calls.clone(),
        })
    }
    
    async fn stream_chat_internal(&self, messages: Vec<Message>) -> Result<Pin<Box<dyn Stream<Item = Result<String, OpenAIError>> + Send>>, OpenAIError> {
        use super::sse::SSEParser;
        use futures_util::StreamExt;
        use std::sync::{Arc, Mutex};
        
        let url = format!("{}/chat/completions", self.config.base_url);
        let body = self.build_request_body(messages, true);
        
        let response = self.client
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
        
        let stream = byte_stream
            .then(move |chunk_result| {
                let parser = parser.clone();
                async move {
                    let mut parser_guard = parser.lock().unwrap();
                    match chunk_result {
                        Ok(bytes) => {
                            let chunk_str = String::from_utf8_lossy(&bytes);
                            let events = parser_guard.parse(&chunk_str);
                            
                            let mut tokens = Vec::new();
                            for event in events {
                                if event.is_done() {
                                    return None;
                                }
                                
                                if let Ok(Some(chunk)) = event.parse_openai_chunk() {
                                    if let Some(choice) = chunk.choices.first() {
                                        if let Some(content) = &choice.delta.content {
                                            tokens.push(Ok(content.clone()));
                                        }
                                    }
                                }
                            }
                            
                            if tokens.is_empty() {
                                None
                            } else {
                                let combined: String = tokens.into_iter().filter_map(|t: Result<String, OpenAIError>| t.ok()).collect();
                                if combined.is_empty() {
                                    None
                                } else {
                                    Some(Ok(combined))
                                }
                            }
                        },
                        Err(e) => Some(Err(OpenAIError::Http(e.to_string()))),
                    }
                }
            })
            .filter_map(|x| async move { x });
        
        Ok(Box::pin(stream))
    }
}

/// OpenAI 错误类型
#[derive(Debug)]
pub enum OpenAIError {
    Http(String),
    Api(String),
    Parse(String),
}

impl std::fmt::Display for OpenAIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIError::Http(msg) => write!(f, "HTTP 错误: {}", msg),
            OpenAIError::Api(msg) => write!(f, "API 错误: {}", msg),
            OpenAIError::Parse(msg) => write!(f, "解析错误: {}", msg),
        }
    }
}

impl std::error::Error for OpenAIError {}

/// OpenAI 响应结构
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
    tool_calls: Option<Vec<crate::core::tools::ToolCall>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}