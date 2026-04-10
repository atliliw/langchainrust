// src/language_models/openai/chat.rs

use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use serde::Deserialize;
use serde_json::json;

use crate::schema::Message;
use crate::RunnableConfig;
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::callbacks::{RunTree, RunType};
use super::OpenAIConfig;

/// OpenAI 聊天客户端
pub struct OpenAIChat {
    config: OpenAIConfig,
    client: reqwest::Client,
}

impl OpenAIChat {
    /// 创建新的 OpenAI 聊天客户端
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
    
    /// 从环境变量创建
    pub fn from_env() -> Self {
        Self::new(OpenAIConfig::from_env())
    }
    
    /// 将消息转换为 OpenAI 格式
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
            crate::schema::MessageType::AI => json!({
                "role": "assistant",
                "content": message.content,
            }),
            crate::schema::MessageType::Tool { tool_call_id } => json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": message.content,
            }),
        }
    }
    
    /// 构建请求体
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
        
        // 添加可选参数
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
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error> {
        // 流式实现稍后添加
        unimplemented!("流式聊天尚未实现")
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
        
        Ok(LLMResult {
            content: chat_response.choices[0].message.content.clone(),
            model: chat_response.model,
            token_usage: chat_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }
    
    async fn stream_chat_internal(&self, messages: Vec<Message>) -> Result<Pin<Box<dyn Stream<Item = Result<String, OpenAIError>> + Send>>, OpenAIError> {
        use super::sse::SSEParser;
        use futures_util::StreamExt;
        
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
        
        let stream = byte_stream
            .then(|chunk_result| async move {
                let mut parser = SSEParser::new();
                match chunk_result {
                    Ok(bytes) => {
                        let chunk_str = String::from_utf8_lossy(&bytes);
                        let events = parser.parse(&chunk_str);
                        
                        for event in events {
                            if event.is_done() {
                                return None;
                            }
                            
                            if let Ok(Some(chunk)) = event.parse_openai_chunk() {
                                if let Some(choice) = chunk.choices.first() {
                                    if let Some(content) = &choice.delta.content {
                                        return Some(Ok(content.clone()));
                                    }
                                }
                            }
                        }
                        
                        None
                    },
                    Err(e) => Some(Err(OpenAIError::Http(e.to_string()))),
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
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}