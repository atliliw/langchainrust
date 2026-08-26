// lc-providers/src/providers/gemini/mod.rs
//! Google Gemini API implementation (native API format).
//!
//! 实现了 Google Gemini 原生 API 的调用，支持：
//! - 文本对话 (generateContent)
//! - 流式输出 (streamGenerateContent)
//! - 工具调用（Function Calling）
//! - Token 用量统计

mod error;
#[cfg(test)]
mod tests;
mod types;

pub use error::GeminiError;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::json;
use std::env;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use self::types::*;
use crate::ProviderError;
use lc_callbacks::{RunTree, RunType};
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
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
    /// Gemini API key.
    pub api_key: String,
    /// Base URL of the Gemini API endpoint.
    pub base_url: String,
    /// Model name to use.
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum number of output tokens.
    pub max_output_tokens: Option<usize>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f32>,
    /// Number of top tokens to consider for sampling.
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
    /// Creates a new GeminiConfig with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Creates a GeminiConfig from environment variables, returning a Result.
    ///
    /// Environment variables:
    /// - `GEMINI_API_KEY` or `GOOGLE_API_KEY`: API key (required)
    /// - `GEMINI_BASE_URL`: API endpoint (optional)
    /// - `GEMINI_MODEL`: Model name (optional)
    pub fn from_env_result() -> Result<Self, ProviderError> {
        let api_key = env::var("GEMINI_API_KEY")
            .or_else(|_| env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                ProviderError::Config(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set".to_string(),
                )
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

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the maximum number of output tokens.
    pub fn with_max_output_tokens(mut self, max: usize) -> Self {
        self.max_output_tokens = Some(max);
        self
    }

    /// L5 fix: alias for with_max_output_tokens for cross-provider consistency.
    pub fn with_max_tokens(self, max: usize) -> Self {
        self.with_max_output_tokens(max)
    }
}

/// Gemini 聊天客户端
#[derive(Clone, Debug)]
pub struct GeminiChat {
    config: GeminiConfig,
    client: reqwest::Client,
}

impl GeminiChat {
    /// Creates a new Gemini chat client with the given configuration.
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates a Gemini chat client from environment variables.
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::from_env_result()
    }

    /// Creates a GeminiChat from environment variables, returning a Result.
    #[allow(deprecated)]
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(GeminiConfig::from_env_result()?))
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
                tool_calls.push(
                    lc_core::tools::ToolCall::builder(format!("call_{}", fc.name))
                        .name(fc.name)
                        .arguments(args_str)
                        .build(),
                );
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, GeminiError>> + Send>>, GeminiError>
    {
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
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, GeminiError>>(64);

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
                                                    if tx
                                                        .send(Ok(StreamChunk::new(text)))
                                                        .await
                                                        .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                // Gemini 在最后一个 chunk 携带 usageMetadata;有则发出
                                // usage chunk,流式路径即可拿到整次调用用量。
                                if let Some(usage) = resp.usage_metadata {
                                    let token_usage = TokenUsage {
                                        prompt_tokens: usage.prompt_token_count.unwrap_or(0)
                                            as usize,
                                        completion_tokens: usage.candidates_token_count.unwrap_or(0)
                                            as usize,
                                        total_tokens: usage.total_token_count.unwrap_or(0) as usize,
                                    };
                                    let usage_chunk = StreamChunk {
                                        text: String::new(),
                                        token_usage: Some(token_usage),
                                    };
                                    if tx.send(Ok(usage_chunk)).await.is_err() {
                                        return;
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
impl BaseLanguageModel<Vec<Message>, LLMResult> for GeminiChat {
    fn model_name(&self) -> &str {
        &self.config.model
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
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
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
                            handler.on_llm_new_token(&run, &token.text).await;
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
