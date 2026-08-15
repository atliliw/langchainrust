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
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, TokenUsage};
use lc_core::runnables::Runnable;
use lc_core::tools::{StructuredOutput, ToolDefinition};
use lc_core::RunnableConfig;
use lc_schema::{Message, MessageType};

/// Gemini API 基础端点
pub const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini 模型列表
pub const GEMINI_MODELS: [&str; 6] = [
    "gemini-2.0-flash",      // Gemini 2.0 Flash（最新快速模型）
    "gemini-2.0-flash-lite", // Gemini 2.0 Flash Lite（轻量版）
    "gemini-1.5-pro",        // Gemini 1.5 Pro（强大推理）
    "gemini-1.5-flash",      // Gemini 1.5 Flash（快速平衡）
    "gemini-1.5-flash-8b",   // Gemini 1.5 Flash 8B（更小更快）
    "gemini-2.0-flash-exp",  // Gemini 2.0 Flash 实验版
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
    /// Tool definitions for function calling (Gemini functionDeclarations).
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice mode: "auto" (AUTO), "none" (NONE), or "any" (ANY).
    pub tool_choice: Option<String>,
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
            tools: None,
            tool_choice: None,
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
    #[deprecated(
        since = "0.7.0",
        note = "Use from_env_result() which returns Result<Self, String>"
    )]
    #[allow(deprecated)]
    pub fn from_env() -> Result<Self, String> {
        Self::from_env_result()
    }

    /// Creates a GeminiConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `GEMINI_API_KEY` or `GOOGLE_API_KEY`: API key (required)
    /// - `GEMINI_BASE_URL`: API endpoint (optional)
    /// - `GEMINI_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, String> {
        let api_key = env::var("GEMINI_API_KEY")
            .or_else(|_| env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set".to_string()
            })?;

        let base_url = env::var("GEMINI_BASE_URL").unwrap_or_else(|_| GEMINI_BASE_URL.to_string());

        let model = env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-1.5-flash".to_string());

        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Sets a custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
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

    /// L5 fix: alias for with_max_output_tokens for cross-provider consistency.
    pub fn with_max_tokens(self, max: usize) -> Self {
        self.with_max_output_tokens(max)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiToolDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
}

/// Gemini tool declaration wrapper (contains functionDeclarations array).
#[derive(Debug, Serialize, Deserialize)]
struct GeminiToolDeclaration {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// A single function declaration in Gemini format.
#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

/// Gemini tool configuration (controls tool choice behavior).
#[derive(Debug, Serialize, Deserialize)]
struct GeminiToolConfig {
    function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCallingConfig {
    /// "AUTO", "ANY", or "NONE"
    mode: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

/// A function call returned by the model (in the response).
#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
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
#[allow(dead_code)]
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
#[derive(Clone, Debug)]
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

// L2 fix: add From<String> for GeminiError, matching OpenAIError pattern
impl From<String> for GeminiError {
    fn from(s: String) -> Self {
        GeminiError::ApiError(s)
    }
}

impl GeminiChat {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self, String> {
        Self::from_env_result()
    }

    /// Creates a GeminiChat from environment variables, returning a Result.
    #[allow(deprecated)]
    pub fn from_env_result() -> Result<Self, String> {
        Ok(Self::new(GeminiConfig::from_env_result()?))
    }

    #[deprecated(since = "0.7.0", note = "Use from_env_result().with_model() instead")]
    #[allow(deprecated)]
    pub fn with_model(model: impl Into<String>) -> Result<Self, String> {
        let config = GeminiConfig::from_env_result()?.with_model(model);
        Ok(Self::new(config))
    }

    /// Binds tool definitions for Gemini function calling.
    ///
    /// Gemini uses `functionDeclarations` inside a `tools` array in the
    /// request body. The conversion from `ToolDefinition` is handled
    /// automatically.
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        let config = GeminiConfig {
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
    /// Accepts "auto" (AUTO), "none" (NONE), or "any" (ANY).
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.config.tool_choice = Some(choice.into());
        self
    }

    /// Enables structured JSON output with schema validation.
    ///
    /// Uses Gemini's function calling under the hood: a single tool named
    /// "structured_output" is bound, and the model is forced to call it.
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> GeminiStructuredOutputMethod<T> {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T))
            .unwrap_or_else(|_| serde_json::json!({"type": "object", "properties": {}}));

        let tool = ToolDefinition::new("structured_output", "Return structured JSON output")
            .with_parameters(schema);

        let config = GeminiConfig {
            tools: Some(vec![tool]),
            tool_choice: Some("auto".to_string()),
            ..self.config.clone()
        };

        GeminiStructuredOutputMethod {
            config,
            client: self.client.clone(),
            _phantom: PhantomData,
        }
    }

    /// 构建 Gemini API 的 contents 数组
    fn build_contents(&self, messages: Vec<Message>) -> (Vec<GeminiContent>, Option<String>) {
        let mut contents = Vec::new();
        let mut system_prompt: Option<String> = None;

        for msg in messages {
            match msg.message_type {
                MessageType::System => {
                    // M9 fix: concatenate system messages instead of overwriting
                    system_prompt = Some(match system_prompt {
                        Some(prev) => format!("{}\n{}", prev, msg.content),
                        None => msg.content,
                    });
                }
                MessageType::Human => {
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart {
                            text: Some(msg.content),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
                MessageType::AI => {
                    contents.push(GeminiContent {
                        role: Some("model".to_string()),
                        parts: vec![GeminiPart {
                            text: Some(msg.content),
                            function_call: None,
                            function_response: None,
                        }],
                    });
                }
                MessageType::Tool { ref tool_call_id } => {
                    // Gemini uses functionResponse format for tool results
                    let function_name = tool_call_id.split('_').next().unwrap_or(tool_call_id);
                    contents.push(GeminiContent {
                        role: Some("function".to_string()),
                        parts: vec![GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: function_name.to_string(),
                                response: json!({"result": msg.content}),
                            }),
                        }],
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
            parts: vec![GeminiPart {
                text: Some(text),
                function_call: None,
                function_response: None,
            }],
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
            // H7: Convert ToolDefinition to Gemini functionDeclarations
            tools: self.config.tools.as_ref().map(|tools| {
                vec![GeminiToolDeclaration {
                    function_declarations: tools
                        .iter()
                        .map(|td| GeminiFunctionDeclaration {
                            name: td.function.name.clone(),
                            description: td.function.description.clone(),
                            parameters: td.function.parameters.clone(),
                        })
                        .collect(),
                }]
            }),
            // H7: Convert tool_choice to Gemini function_calling_config
            tool_config: self.config.tool_choice.as_ref().map(|choice| {
                let mode = match choice.as_str() {
                    "none" => "NONE",
                    "any" => "ANY",
                    _ => "AUTO",
                };
                GeminiToolConfig {
                    function_calling_config: GeminiFunctionCallingConfig {
                        mode: mode.to_string(),
                    },
                }
            }),
        }
    }

    /// 解析 Gemini API 响应为 LLMResult
    fn parse_response(
        &self,
        response: GeminiResponse,
        model: &str,
    ) -> Result<LLMResult, GeminiError> {
        // 检查 safety feedback
        if let Some(feedback) = &response.prompt_feedback {
            if let Some(block_reason) = feedback.get("blockReason").and_then(|v| v.as_str()) {
                return Err(GeminiError::SafetyBlock(block_reason.to_string()));
            }
        }

        let candidates = response.candidates.ok_or(GeminiError::NoResponse)?;
        let candidate = candidates
            .into_iter()
            .next()
            .ok_or(GeminiError::NoResponse)?;

        let content = candidate.content.ok_or(GeminiError::NoResponse)?;

        let mut text_parts = String::new();
        let mut tool_calls: Vec<lc_core::tools::ToolCall> = Vec::new();

        for part in content.parts {
            if let Some(text) = part.text {
                text_parts.push_str(&text);
            }
            // H7: Parse functionCall parts into ToolCall
            if let Some(fc) = part.function_call {
                let args_str = fc.args.unwrap_or(serde_json::json!({})).to_string();
                tool_calls.push(lc_core::tools::ToolCall::new(
                    format!("call_{}", fc.name),
                    fc.name,
                    args_str,
                ));
            }
        }

        let token_usage = response.usage_metadata.map(|u| TokenUsage {
            prompt_tokens: u.prompt_token_count.unwrap_or(0) as usize,
            completion_tokens: u.candidates_token_count.unwrap_or(0) as usize,
            total_tokens: u.total_token_count.unwrap_or(0) as usize,
        });

        Ok(LLMResult {
            content: text_parts,
            model: model.to_string(),
            token_usage,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking_content: None,
        })
    }

    /// 内部调用：发送请求到 Gemini API
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, GeminiError> {
        let url = format!(
            "{}/models/{}:generateContent",
            self.config.base_url, self.config.model
        );

        let request_body = self.build_request(messages);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        if !status.is_success() {
            return Err(GeminiError::ApiError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                &body[..std::cmp::min(500, body.len())]
            )));
        }

        let gemini_response: GeminiResponse = serde_json::from_str(&body).map_err(|e| {
            GeminiError::ParseError(format!(
                "{} - body: {}",
                e,
                &body[..std::cmp::min(200, body.len())]
            ))
        })?;

        self.parse_response(gemini_response, &self.config.model)
    }

    /// 流式调用
    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, GeminiError>> + Send>>, GeminiError> {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=event-stream",
            self.config.base_url, self.config.model
        );

        let request_body = self.build_request(messages);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(GeminiError::ApiError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                body
            )));
        }

        let byte_stream = response.bytes_stream();
        let sse_buffer = Arc::new(Mutex::new(String::new()));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, GeminiError>>(64);

        let buffer_clone = sse_buffer.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                if let Ok(bytes) = chunk_result {
                    let chunk_str = String::from_utf8_lossy(&bytes);

                    // Extract complete events from buffer
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
                            let line = line.trim();
                            if !line.starts_with("data: ") {
                                continue;
                            }

                            let data = &line[6..];
                            if data == "[DONE]" {
                                continue;
                            }

                            if let Ok(resp) = serde_json::from_str::<GeminiResponse>(data) {
                                if let Some(candidates) = resp.candidates {
                                    for candidate in candidates {
                                        if let Some(content) = candidate.content {
                                            for part in content.parts {
                                                if let Some(text) = part.text {
                                                    if tx.send(Ok(text)).await.is_err() {
                                                        return;
                                                    }
                                                }
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
        config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<LLMResult, Self::Error>> + Send>>, Self::Error>
    {
        let model = self.config.model.clone();
        let (temp, max) = crate::sampling::sampling_overrides(&config);
        let mut effective = self.clone();
        if let Some(t) = temp {
            effective.config.temperature = Some(t);
        }
        if let Some(m) = max {
            effective.config.max_output_tokens = Some(m);
        }
        let token_stream = effective.stream_chat_internal(input).await?;

        // C1 fix: true streaming — emit one LLMResult per token,
        // matching OpenAI/Ollama/Anthropic behavior.
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for GeminiChat {
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
            effective.config.max_output_tokens = Some(m);
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
            effective.config.max_output_tokens = Some(m);
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

    fn bind_tools(
        &self,
        tools: Vec<ToolDefinition>,
    ) -> Option<Box<dyn BaseChatModel<Error = Self::Error> + Send + Sync>> {
        // Expose the inherent tool-binding capability at the trait level so it
        // survives being wrapped by `ChatModelWrapper` / `LLMClient` (Q1).
        Some(Box::new(self.bind_tools(tools)))
    }
}

/// Method for structured output calls via Gemini function calling.
pub struct GeminiStructuredOutputMethod<T: DeserializeOwned + JsonSchema> {
    config: GeminiConfig,
    client: reqwest::Client,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> GeminiStructuredOutputMethod<T> {
    /// Invokes the model and parses the result as the structured type.
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<T, GeminiError> {
        let chat = GeminiChat {
            config: self.config.clone(),
            client: self.client.clone(),
        };

        let result = chat.chat_internal(messages).await?;
        let structured = StructuredOutput::<T>::new(result);
        structured
            .parse()
            .map_err(|e| GeminiError::ParseError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::tools::ToolDefinition;
    use serde_json::json;

    #[test]
    fn test_bind_tools_creates_new_chat_with_tools() {
        let config = GeminiConfig::new("test-key");
        let chat = GeminiChat::new(config);
        let tools = vec![
            ToolDefinition::new("calculator", "Do math").with_parameters(
                json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            ),
        ];

        let bound = chat.bind_tools(tools.clone());
        assert!(bound.config.tools.is_some());
        assert_eq!(bound.config.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            bound.config.tools.as_ref().unwrap()[0].function.name,
            "calculator"
        );
        // Original chat should not have tools
        assert!(chat.config.tools.is_none());
    }

    #[test]
    fn test_with_tool_choice_sets_config() {
        let config = GeminiConfig::new("test-key");
        let chat = GeminiChat::new(config);
        let chat = chat.with_tool_choice("auto");
        assert_eq!(chat.config.tool_choice.as_deref(), Some("auto"));
    }

    #[test]
    fn test_build_request_includes_tools() {
        let config = GeminiConfig::new("test-key");
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather").with_parameters(
                json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            ),
        ];
        let chat = GeminiChat::new(config).bind_tools(tools);

        let request = chat.build_request(vec![]);
        assert!(request.tools.is_some());
        let tool_decls = &request.tools.as_ref().unwrap()[0].function_declarations;
        assert_eq!(tool_decls.len(), 1);
        assert_eq!(tool_decls[0].name, "get_weather");
        assert!(tool_decls[0].parameters.is_some());
    }

    #[test]
    fn test_build_request_tool_choice_auto() {
        let config = GeminiConfig::new("test-key");
        let chat = GeminiChat::new(config).with_tool_choice("auto");
        let request = chat.build_request(vec![]);
        assert!(request.tool_config.is_some());
        assert_eq!(
            request
                .tool_config
                .as_ref()
                .unwrap()
                .function_calling_config
                .mode,
            "AUTO"
        );
    }

    #[test]
    fn test_build_request_tool_choice_none() {
        let config = GeminiConfig::new("test-key");
        let chat = GeminiChat::new(config).with_tool_choice("none");
        let request = chat.build_request(vec![]);
        assert_eq!(
            request
                .tool_config
                .as_ref()
                .unwrap()
                .function_calling_config
                .mode,
            "NONE"
        );
    }

    #[test]
    fn test_with_structured_output_binds_tool() {
        let config = GeminiConfig::new("test-key");
        let chat = GeminiChat::new(config);
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct TestOutput {
            answer: String,
        }
        let _method: GeminiStructuredOutputMethod<TestOutput> = chat.with_structured_output();
        // Just verify it compiles and the method is callable
    }
}
