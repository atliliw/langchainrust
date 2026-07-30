// src/language_models/openai/responses/types.rs
//! Type definitions for the OpenAI Responses API.
//!
//! Contains configuration, error types, API response structures,
//! and streaming event types used by the Responses API model.

use serde::Deserialize;
use serde_json::json;

// ---------------------------------------------------------------------------
// Builtin tools
// ---------------------------------------------------------------------------

/// Built-in tools available through the Responses API.
#[derive(Debug, Clone)]
pub enum BuiltinTool {
    /// Web search via OpenAI.
    WebSearch,
    /// File search over vector stores.
    FileSearch { vector_store_ids: Vec<String> },
    /// Code interpreter sandbox.
    CodeInterpreter,
    /// Computer use (GUI automation).
    ComputerUse {
        display_width: Option<u32>,
        display_height: Option<u32>,
    },
}

impl BuiltinTool {
    /// Convert to the JSON value expected by the Responses API `tools` field.
    pub(crate) fn to_api_value(&self) -> serde_json::Value {
        match self {
            BuiltinTool::WebSearch => json!({"type": "web_search"}),
            BuiltinTool::FileSearch { vector_store_ids } => json!({
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }),
            BuiltinTool::CodeInterpreter => json!({"type": "code_interpreter"}),
            BuiltinTool::ComputerUse {
                display_width,
                display_height,
            } => {
                let mut val = json!({"type": "computer_use"});
                if let Some(w) = display_width {
                    val["display_width"] = json!(w);
                }
                if let Some(h) = display_height {
                    val["display_height"] = json!(h);
                }
                val
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Responses API model.
#[derive(Debug, Clone)]
pub struct ResponsesConfig {
    /// API key.
    pub api_key: String,
    /// Model name (e.g. "gpt-4o").
    pub model: String,
    /// Base URL (defaults to `https://api.openai.com/v1`).
    pub base_url: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens.
    pub max_tokens: Option<usize>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f32>,
    /// Built-in tools to include in every request.
    pub builtin_tools: Vec<BuiltinTool>,
}

impl Default for ResponsesConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            builtin_tools: Vec::new(),
        }
    }
}

impl ResponsesConfig {
    /// Create a new config with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Create config from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` and
    /// `OPENAI_MODEL` (optional).
    pub fn from_env() -> Result<Self, ResponsesError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            ResponsesError::Api("OPENAI_API_KEY environment variable must be set".to_string())
        })?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        Ok(Self {
            api_key,
            base_url,
            model,
            ..Default::default()
        })
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set the max output tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Add a built-in tool.
    pub fn with_builtin_tool(mut self, tool: BuiltinTool) -> Self {
        self.builtin_tools.push(tool);
        self
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the Responses API model.
#[derive(Debug)]
pub enum ResponsesError {
    /// HTTP transport error.
    Http(String),
    /// API returned a non-success status.
    Api(String),
    /// Response parsing error.
    Parse(String),
}

impl std::fmt::Display for ResponsesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponsesError::Http(msg) => write!(f, "HTTP error: {}", msg),
            ResponsesError::Api(msg) => write!(f, "API error: {}", msg),
            ResponsesError::Parse(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ResponsesError {}

// ---------------------------------------------------------------------------
// Response structures
// ---------------------------------------------------------------------------

/// Top-level response from the `/v1/responses` endpoint.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ResponsesApiResponse {
    pub id: String,
    pub object: Option<String>,
    pub model: Option<String>,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: Option<ResponsesUsage>,
}

/// A single item in the `output` array.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ResponsesOutputItem {
    /// A text message from the model.
    #[serde(rename = "message")]
    Message(ResponsesMessage),
    /// A web search call.
    #[serde(rename = "web_search_call")]
    WebSearchCall(ResponsesWebSearchCall),
    /// A file search call.
    #[serde(rename = "file_search_call")]
    FileSearchCall(ResponsesFileSearchCall),
    /// A code interpreter call.
    #[serde(rename = "code_interpreter_call")]
    CodeInterpreterCall(ResponsesCodeInterpreterCall),
    /// A computer use call.
    #[serde(rename = "computer_call")]
    ComputerCall(ResponsesComputerCall),
}

/// Message output item.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ResponsesMessage {
    pub id: Option<String>,
    pub role: Option<String>,
    pub content: Vec<ResponsesContentPart>,
    pub status: Option<String>,
}

/// Content part inside a message.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ResponsesContentPart {
    #[serde(rename = "output_text")]
    OutputText(ResponsesOutputText),
    #[serde(rename = "refusal")]
    Refusal(ResponsesRefusal),
}

/// Text content part.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ResponsesOutputText {
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}

/// Refusal content part.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesRefusal {
    pub refusal: String,
}

/// Web search call output.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesWebSearchCall {
    pub id: String,
    pub status: String,
    pub query: Option<String>,
}

/// File search call output.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesFileSearchCall {
    pub id: String,
    pub status: String,
    pub query: Option<String>,
}

/// Code interpreter call output.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesCodeInterpreterCall {
    pub id: String,
    pub code: Option<String>,
    pub results: Option<Vec<serde_json::Value>>,
    pub status: Option<String>,
}

/// Computer use call output.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesComputerCall {
    pub id: String,
    pub action: Option<serde_json::Value>,
    pub status: Option<String>,
}

/// Token usage from the Responses API.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Stream structures
// ---------------------------------------------------------------------------

/// SSE event types for the Responses API streaming.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(tag = "type")]
pub(crate) enum ResponsesStreamEvent {
    /// Emitted when the response object is created.
    #[serde(rename = "response.created")]
    Created(serde_json::Value),
    /// In-progress output item added.
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(serde_json::Value),
    /// Content part added to an output item.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(serde_json::Value),
    /// Text delta for output_text.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(ResponsesTextDelta),
    /// Text content part completed.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(serde_json::Value),
    /// Content part completed.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(serde_json::Value),
    /// Output item completed.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(serde_json::Value),
    /// Web search call in progress.
    #[serde(rename = "response.web_search_call.in_progress")]
    WebSearchInProgress(serde_json::Value),
    /// Web search call searching.
    #[serde(rename = "response.web_search_call.searching")]
    WebSearchSearching(serde_json::Value),
    /// Web search call completed.
    #[serde(rename = "response.web_search_call.completed")]
    WebSearchCompleted(serde_json::Value),
    /// Code interpreter call in progress.
    #[serde(rename = "response.code_interpreter_call.in_progress")]
    CodeInterpreterInProgress(serde_json::Value),
    /// Code interpreter call code delta.
    #[serde(rename = "response.code_interpreter_call.code_delta")]
    CodeInterpreterCodeDelta(serde_json::Value),
    /// Code interpreter call completed.
    #[serde(rename = "response.code_interpreter_call.completed")]
    CodeInterpreterCompleted(serde_json::Value),
    /// File search call in progress.
    #[serde(rename = "response.file_search_call.in_progress")]
    FileSearchInProgress(serde_json::Value),
    /// File search call completed.
    #[serde(rename = "response.file_search_call.completed")]
    FileSearchCompleted(serde_json::Value),
    /// Response completed.
    #[serde(rename = "response.completed")]
    Completed(ResponsesCompletedEvent),
    /// Response failed.
    #[serde(rename = "response.failed")]
    Failed(serde_json::Value),
}

/// Text delta payload.
#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesTextDelta {
    pub delta: String,
}

/// Completed event payload (contains full response).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ResponsesCompletedEvent {
    pub response: ResponsesApiResponse,
}
