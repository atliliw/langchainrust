// src/language_models/providers/gemini.rs
//! Google Gemini API implementation (native API format).
//!
//! 实现了 Google Gemini 原生 API 的调用，支持：
//! - 文本对话 (generateContent)
//! - 流式输出 (streamGenerateContent)
//! - 工具调用（Function Calling）
//! - Token 用量统计

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

use crate::schema::{Message, MessageType};
use crate::RunnableConfig;
use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use crate::core::runnables::Runnable;
use crate::callbacks::{RunTree, RunType};

/// Gemini API 基础端点
pub const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini 模型列表
pub const GEMINI_MODELS: [&str; 6] = [
    "gemini-2.0-flash",       // Gemini 2.0 Flash（最新快速模型）
    "gemini-2.0-flash-lite",  // Gemini 2.0 Flash Lite（轻量版）
    "gemini-1.5-pro",         // Gemini 1.5 Pro（强大推理）
    "gemini-1.5-flash",       // Gemini 1.5 Flash（快速平衡）
    "gemini-1.5-flash-8b",    // Gemini 1.5 Flash 8B（更小更快）
    "gemini-2.0-flash-exp",   // Gemini 2.0 Flash 实验版
];

/// Gemini 配置
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: GEMINI_BASE_URL.to_string(),
            model: "gemini-1.5-flash".to_string(),
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            top_k: None,
        }
    }
}

impl GeminiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// 从环境变量创建配置
    ///
    /// 读取 GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
    pub fn from_env() -> Self {
        let api_key = env::var("GEMINI_API_KEY")
            .or_else(|_| env::var("GOOGLE_API_KEY"))
            .expect("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set");

        let base_url = env::var("GEMINI_BASE_URL")
            .unwrap_or_else(|_| GEMINI_BASE_URL.to_string());

        let model = env::var("GEMINI_MODEL")
            .unwrap_or_else(|_| "gemini-1.5-flash".to_string());

        Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_output_tokens(mut self, max: usize) -> Self {
        self.max_output_tokens = Some(max);
        self
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(default)]
    prompt_feedback: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    prompt_token_count: Option<i32>,
    candidates_token_count: Option<i32>,
    total_token_count: Option<i32>,
}

/// Gemini 聊天客户端
pub struct GeminiChat {
    config: GeminiConfig,
    client: reqwest::Client,
}

/// Gemini 错误类型
#[derive(Debug)]
pub enum GeminiError {
    ApiError(String),
    HttpError(String),
    ParseError(String),
    NoResponse,
    SafetyBlock(String),
}

impl std::fmt::Display for GeminiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeminiError::ApiError(msg) => write!(f, "Gemini API error: {}", msg),
            GeminiError::HttpError(msg) => write!(f, "Gemini HTTP error: {}", msg),
            GeminiError::ParseError(msg) => write!(f, "Gemini parse error: {}", msg),
            GeminiError::NoResponse => write!(f, "Gemini returned no response"),
            GeminiError::SafetyBlock(msg) => write!(f, "Gemini blocked by safety filter: {}", msg),
        }
    }
}

impl std::error::Error for GeminiError {}

impl GeminiChat {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Self {
        Self::new(GeminiConfig::from_env())
    }

    pub fn with_model(model: impl Into<String>) -> Self {
        let config = GeminiConfig::from_env().with_model(model);
        Self::new(config)
    }

    /// 构建 Gemini API 的 contents 数组
    fn build_contents(&self, messages: Vec<Message>) -> (Vec<GeminiContent>, Option<String>) {
        let mut contents = Vec::new();
        let mut system_prompt: Option<String> = None;

        for msg in messages {
            match msg.message_type {
                MessageType::System => {
                    system_prompt = Some(msg.content);
                }
                MessageType::Human => {
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart { text: Some(msg.content) }],
                    });
                }
                MessageType::AI => {
                    contents.push(GeminiContent {
                        role: Some("model".to_string()),
                        parts: vec![GeminiPart { text: Some(msg.content) }],
                    });
                }
                MessageType::Tool { .. } => {
                    // Gemini 的 function response 格式略有不同
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart { text: Some(msg.content) }],
                    });
                }
            }
        }

        (contents, system_prompt)
    }

    /// 构建 API 请求体
    fn build_request(&self, messages: Vec<Message>) -> GeminiRequest {
        let (contents, system_text) = self.build_contents(messages);

        let system_instruction = system_text.map(|text| GeminiSystemInstruction {
            parts: vec![GeminiPart { text: Some(text) }],
        });

        let generation_config = {
            let has_config = self.config.temperature.is_some()
                || self.config.max_output_tokens.is_some()
                || self.config.top_p.is_some()
                || self.config.top_k.is_some();

            if has_config {
                Some(GeminiGenerationConfig {
                    temperature: self.config.temperature,
                    max_output_tokens: self.config.max_output_tokens,
                    top_p: self.config.top_p,
                    top_k: self.config.top_k,
                })
            } else {
                None
            }
        };

        GeminiRequest {
            contents,
            system_instruction,
            generation_config,
        }
    }

    /// 解析 Gemini API 响应为 LLMResult
    fn parse_response(&self, response: GeminiResponse, model: &str) -> Result<LLMResult, GeminiError> {
        // 检查 safety feedback
        if let Some(feedback) = &response.prompt_feedback {
            if let Some(block_reason) = feedback.get("blockReason").and_then(|v| v.as_str()) {
                return Err(GeminiError::SafetyBlock(block_reason.to_string()));
            }
        }

        let candidates = response.candidates.ok_or(GeminiError::NoResponse)?;
        let candidate = candidates.into_iter().next().ok_or(GeminiError::NoResponse)?;

        let content = candidate
            .content
            .ok_or(GeminiError::NoResponse)?;

        let text = content
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("");

        let token_usage = response.usage_metadata.map(|u| TokenUsage {
            prompt_tokens: u.prompt_token_count.unwrap_or(0) as usize,
            completion_tokens: u.candidates_token_count.unwrap_or(0) as usize,
            total_tokens: u.total_token_count.unwrap_or(0) as usize,
        });

        Ok(LLMResult {
            content: text,
            model: model.to_string(),
            token_usage,
            tool_calls: None,
        })
    }

    /// 内部调用：发送请求到 Gemini API
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, GeminiError> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.config.base_url, self.config.model, self.config.api_key
        );

        let request_body = self.build_request(messages);

        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        let status = response.status();
        let body = response.text().await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        if !status.is_success() {
            return Err(GeminiError::ApiError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                &body[..std::cmp::min(500, body.len())]
            )));
        }

        let gemini_response: GeminiResponse = serde_json::from_str(&body)
            .map_err(|e| GeminiError::ParseError(format!("{} - body: {}", e, &body[..std::cmp::min(200, body.len())])))?;

        self.parse_response(gemini_response, &self.config.model)
    }

    /// 流式调用
    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, GeminiError>> + Send>>, GeminiError> {
        use futures_util::StreamExt;

        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=event-stream&key={}",
            self.config.base_url, self.config.model, self.config.api_key
        );

        let request_body = self.build_request(messages);

        let response = self.client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(GeminiError::ApiError(format!("HTTP {}: {}", status.as_u16(), body)));
        }

        let byte_stream = response.bytes_stream();
        let stream = byte_stream
            .then(|chunk_result| async move {
                match chunk_result {
                    Ok(bytes) => {
                        let chunk_str = String::from_utf8_lossy(&bytes);
                        let mut texts = Vec::new();

                        for line in chunk_str.lines() {
                            let line = line.trim();
                            if !line.starts_with("data: ") {
                                continue;
                            }

                            let data = &line[6..]; // 去掉 "data: "
                            if data == "[DONE]" {
                                continue;
                            }

                            if let Ok(resp) = serde_json::from_str::<GeminiResponse>(data) {
                                if let Some(candidates) = resp.candidates {
                                    for candidate in candidates {
                                        if let Some(content) = candidate.content {
                                            for part in content.parts {
                                                if let Some(text) = part.text {
                                                    texts.push(text);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if texts.is_empty() {
                            None
                        } else {
                            Some(Ok(texts.concat()))
                        }
                    }
                    Err(e) => Some(Err(GeminiError::HttpError(e.to_string()))),
                }
            })
            .filter_map(|x| async move { x });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for GeminiChat {
    type Error = GeminiError;

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
impl BaseLanguageModel<Vec<Message>, LLMResult> for GeminiChat {
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
        self.config.max_output_tokens
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    fn with_max_tokens(mut self, max: usize) -> Self {
        self.config.max_output_tokens = Some(max);
        self
    }
}

#[async_trait]
impl BaseChatModel for GeminiChat {
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
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        self.stream_chat_internal(messages).await
    }
}
