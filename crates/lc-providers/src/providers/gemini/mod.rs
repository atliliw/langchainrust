// lc-providers/src/providers/gemini/mod.rs
//! Google Gemini API implementation (native API format).
//!
//! Implements calling Google's native Gemini API, supporting:
//! - text chat (generateContent)
//! - streaming (streamGenerateContent)
//! - function calling
//! - token usage statistics

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
use crate::openai::sse::SseByteFramer;
use crate::ProviderError;
use lc_callbacks::RunType;
use lc_core::language_models::{
    BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk, TokenUsage,
};
use lc_core::runnables::{run_tree_from_config, Runnable};
use lc_core::text::truncate_at_char_boundary;
use lc_core::tools::{StructuredOutput, ToolDefinition};
use lc_core::RunnableConfig;
use lc_schema::{Message, MessageType};

/// Gemini API base endpoint
pub const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini model list (B5, v0.22.4 — Gemini 3 / 2.5 generation; non-exhaustive).
pub const GEMINI_MODELS: [&str; 6] = [
    "gemini-3-pro",          // Gemini 3 Pro (2026 flagship)
    "gemini-3-flash",        // Gemini 3 Flash
    "gemini-2.5-pro",        // Gemini 2.5 Pro (strong reasoning)
    "gemini-2.5-flash",      // Gemini 2.5 Flash (fast and balanced)
    "gemini-2.5-flash-lite", // Gemini 2.5 Flash Lite (lightweight)
    "gemini-2.0-flash",      // Gemini 2.0 Flash (stable previous gen)
];

/// Gemini config
#[derive(Clone)]
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

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A13: redact the API key so `{:?}` never leaks the secret.
        f.debug_struct("GeminiConfig")
            .field("api_key", &"***")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
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

/// Gemini chat client
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
            // 0.22.0 audit fix (H-P1): shared client with a connect timeout.
            client: crate::retry::default_client(),
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

    /// Builds the contents array for the Gemini API
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
                    // B7: text part first, then one media part per attached medium.
                    // The async entries resolve every reference (fetch-to-inline
                    // for http(s), pass-through for gs://) before this runs.
                    let mut parts: Vec<GeminiPart> = Vec::new();
                    if !msg.content.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(msg.content.clone()),
                            function_call: None,
                            function_response: None,
                            inline_data: None,
                            file_data: None,
                        });
                    }
                    for media in msg.media_parts() {
                        if let Some(media_part) = Self::media_to_part(&media) {
                            parts.push(media_part);
                        }
                    }
                    if parts.is_empty() {
                        // Preserve prior behaviour: a bare user turn carries its
                        // text (even if empty) as a single text part.
                        parts.push(GeminiPart {
                            text: Some(msg.content),
                            function_call: None,
                            function_response: None,
                            inline_data: None,
                            file_data: None,
                        });
                    }
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts,
                    });
                }
                MessageType::AI => {
                    // A11: replay an assistant turn's tool calls as `functionCall` parts,
                    // aligned with OpenAI/Anthropic/Ollama — otherwise multi-round tool
                    // dialogs break from the second round because the model never sees
                    // its own prior function calls. `call_{name}` ids match what
                    // `parse_response` produces (and what the Tool-arm strips below).
                    let mut parts: Vec<GeminiPart> = Vec::new();
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            parts.push(GeminiPart {
                                text: None,
                                function_call: Some(GeminiFunctionCall {
                                    name: tc.function.name.clone(),
                                    args: serde_json::from_str(&tc.function.arguments).ok(),
                                }),
                                function_response: None,
                                inline_data: None,
                                file_data: None,
                            });
                        }
                    }
                    if !msg.content.is_empty() || parts.is_empty() {
                        parts.push(GeminiPart {
                            text: Some(msg.content),
                            function_call: None,
                            function_response: None,
                            inline_data: None,
                            file_data: None,
                        });
                    }
                    contents.push(GeminiContent {
                        role: Some("model".to_string()),
                        parts,
                    });
                }
                MessageType::Tool { ref tool_call_id } => {
                    // Gemini uses functionResponse format for tool results. The
                    // outgoing ToolCall id is built as `call_{name}` (see
                    // parse_response); recover the bare function name here.
                    // Previously `split('_').next()` always returned the literal
                    // "call", so the functionResponse.name never matched a real
                    // declaration and any multi-round tool dialog broke from the
                    // second round (0.22.0 audit H-P7).
                    let function_name = tool_call_id.strip_prefix("call_").unwrap_or(tool_call_id);
                    contents.push(GeminiContent {
                        role: Some("function".to_string()),
                        parts: vec![GeminiPart {
                            text: None,
                            function_call: None,
                            function_response: Some(GeminiFunctionResponse {
                                name: function_name.to_string(),
                                response: json!({"result": msg.content}),
                            }),
                            inline_data: None,
                            file_data: None,
                        }],
                    });
                }
            }
        }

        (contents, system_prompt)
    }

    /// B7: maps one unified [`lc_schema::MediaPart`] to a Gemini media part.
    ///
    /// Data URIs become `inline_data` (raw base64 + MIME); `gs://` references
    /// become `file_data`. The async entries run the Gemini media policy via
    /// `media::resolve_message_media` first — http(s) media is fetched
    /// SSRF-safely into data URIs and MIME types are validated — so anything
    /// that is neither a data URI nor a `gs://` reference is skipped
    /// defensively rather than sent to the API.
    fn media_to_part(part: &lc_schema::MediaPart<'_>) -> Option<GeminiPart> {
        let url = part.url();

        let (inline_data, file_data) = if let Some(gs_path) = url.strip_prefix("gs://") {
            let mime = part
                .mime_type()
                .or_else(|| crate::media::mime_from_extension(url))?
                .to_string();
            (
                None,
                Some(GeminiFileData {
                    file_uri: format!("gs://{gs_path}"),
                    mime_type: mime,
                }),
            )
        } else if let Some((uri_mime, data)) = crate::media::data_uri_parts(url) {
            // An explicitly declared file MIME wins over the data-URI header.
            let mime = part.mime_type().unwrap_or(uri_mime).to_string();
            (
                Some(GeminiInlineData {
                    mime_type: mime,
                    data: data.to_string(),
                }),
                None,
            )
        } else {
            return None;
        };

        Some(GeminiPart {
            text: None,
            function_call: None,
            function_response: None,
            inline_data,
            file_data,
        })
    }

    /// Builds the API request body
    fn build_request(&self, messages: Vec<Message>) -> GeminiRequest {
        let (contents, system_text) = self.build_contents(messages);

        let system_instruction = system_text.map(|text| GeminiSystemInstruction {
            parts: vec![GeminiPart {
                text: Some(text),
                function_call: None,
                function_response: None,
                inline_data: None,
                file_data: None,
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

    /// Parses a Gemini API response into an LLMResult
    fn parse_response(
        &self,
        response: GeminiResponse,
        model: &str,
    ) -> Result<LLMResult, GeminiError> {
        // Check the safety feedback
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

    /// Internal call: sends the request to the Gemini API
    async fn chat_internal(&self, messages: Vec<Message>) -> Result<LLMResult, GeminiError> {
        let url = format!(
            "{}/models/{}:generateContent",
            self.config.base_url, self.config.model
        );

        // B7: inline http(s) media through the SSRF guard, allow gs:// through,
        // and reject unsupported schemes/types before building the request.
        let mut messages = messages;
        crate::media::resolve_message_media(&mut messages, crate::media::MediaPolicy::Gemini)
            .await
            .map_err(|e| GeminiError::ApiError(e.to_string()))?;
        let request_body = self.build_request(messages);

        // 0.22.0 audit fix (H-P2): retry transient failures (429/5xx/network).
        // A14: non-idempotent POST — see retry::TransportRetryMode; use
        // retry::SAFE_RETRY to forbid replaying a possibly-dispatched request.
        let response = crate::retry::send_with_retry(
            || {
                self.client
                    .post(&url)
                    .header("x-goog-api-key", &self.config.api_key)
                    .header("Content-Type", "application/json")
                    .json(&request_body)
            },
            &crate::retry::DEFAULT_RETRY,
        )
        .await
        .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| GeminiError::HttpError(e.to_string()))?;

        if !status.is_success() {
            // 0.22.0 C6 fix: char-boundary truncation (byte slicing panicked on
            // non-ASCII error bodies).
            let preview: String = body.chars().take(500).collect();
            return Err(GeminiError::ApiError(format!(
                "HTTP {}: {}",
                status.as_u16(),
                preview
            )));
        }

        let gemini_response: GeminiResponse = serde_json::from_str(&body).map_err(|e| {
            // 0.22.0 C6 fix: char-boundary truncation.
            let preview: String = body.chars().take(200).collect();
            GeminiError::ParseError(format!("{} - body: {}", e, preview))
        })?;

        self.parse_response(gemini_response, &self.config.model)
    }

    /// Streaming call
    async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, GeminiError>> + Send>>, GeminiError>
    {
        // 0.25.0: the documented SSE media selector is `alt=sse`. The older
        // `alt=event-stream` value is not the public contract and some Gemini
        // endpoints answer it with the non-SSE JSON array framing, which the
        // SSE byte framer cannot parse.
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.config.base_url, self.config.model
        );

        // B7: same media resolution as the non-streaming path.
        let mut messages = messages;
        crate::media::resolve_message_media(&mut messages, crate::media::MediaPolicy::Gemini)
            .await
            .map_err(|e| GeminiError::ApiError(e.to_string()))?;
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
        // 0.22.0 C1: byte-level framer — complete events are decoded to UTF-8,
        // so CJK characters split across TCP chunks are never lossy-torn.
        let sse_buffer = Arc::new(Mutex::new(SseByteFramer::new()));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, GeminiError>>(64);

        let buffer_clone = sse_buffer.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            // A10: mirror the OpenAI A12 terminal guard. Gemini terminates a
            // stream with a `usageMetadata` chunk (and a candidate `finishReason`).
            // If the connection closes first (proxy reset, server crash, timeout),
            // the streamed text is a truncated prefix — report it as an error
            // rather than silently returning a partial answer as complete.
            let mut saw_terminal = false;
            while let Some(chunk_result) = byte_stream.next().await {
                if let Ok(bytes) = chunk_result {
                    // Extract complete events from the byte-level framer
                    let events = {
                        let mut buffer_guard =
                            buffer_clone.lock().unwrap_or_else(|e| e.into_inner());
                        buffer_guard.push(&bytes)
                    };
                    // buffer_guard is dropped here, before any await

                    for event_text in events {
                        for line in event_text.lines() {
                            let line = line.trim();
                            if !line.starts_with("data:") {
                                continue;
                            }

                            // Tolerate both "data: {...}" and "data:{...}"
                            let data = line.trim_start_matches("data:").trim();
                            if data == "[DONE]" {
                                continue;
                            }

                            match serde_json::from_str::<GeminiResponse>(data) {
                                Ok(resp) => {
                                    if let Some(candidates) = resp.candidates {
                                        for candidate in candidates {
                                            // A10: a `finishReason` marks the terminal chunk —
                                            // the model signalled the end of generation.
                                            if candidate.finish_reason.is_some() {
                                                saw_terminal = true;
                                            }
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
                                    // Gemini carries usageMetadata on the last chunk; if present,
                                    // emit a usage chunk so the streaming path gets the whole call's usage
                                    // and treat it as a terminal marker (A10).
                                    if let Some(usage) = resp.usage_metadata {
                                        saw_terminal = true;
                                        let token_usage = TokenUsage {
                                            prompt_tokens: usage.prompt_token_count.unwrap_or(0)
                                                as usize,
                                            completion_tokens: usage
                                                .candidates_token_count
                                                .unwrap_or(0)
                                                as usize,
                                            total_tokens: usage.total_token_count.unwrap_or(0)
                                                as usize,
                                        };
                                        let usage_chunk = StreamChunk {
                                            thinking_content: None,
                                            text: String::new(),
                                            token_usage: Some(token_usage),
                                            tool_calls: None,
                                        };
                                        if tx.send(Ok(usage_chunk)).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(e) => {
                                    // 0.22.0 (audit Medium): a bad datum no longer ends the
                                    // stream silently — log and skip (transport errors are
                                    // still surfaced below).
                                    log::error!(
                                        "Failed to parse Gemini streaming SSE event (skipping this token): {e}; data: {}",
                                        truncate_at_char_boundary(data, 200)
                                    );
                                }
                            }
                        }
                    }
                } else if let Err(e) = chunk_result {
                    // 0.22.0 (audit Medium): transport errors mid-stream no longer
                    // end the stream silently.
                    let _ = tx.send(Err(GeminiError::HttpError(e.to_string()))).await;
                    return;
                }
            }
            // A10: the byte stream ended without any terminal marker (usageMetadata or
            // finishReason). What was sent so far is a truncated prefix, not a complete
            // answer — surface the interruption instead of completing normally.
            if !saw_terminal {
                let _ = tx
                    .send(Err(GeminiError::StreamInterrupted(
                        "connection closed before usageMetadata or finishReason".to_string(),
                    )))
                    .await;
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
            // If the encoder fails to load, overestimate by byte length (better slightly high than silently counting 0, which would mislead routing/truncation)
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

        let mut run = run_tree_from_config(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>(),
                "model": self.config.model,
            }),
            config.as_ref(),
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

        let run = run_tree_from_config(
            run_name,
            RunType::Llm,
            json!({
                "messages": messages.len(),
                "model": self.config.model,
            }),
            config.as_ref(),
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
