// src/core/batch/types.rs
//! Public and internal types for the batch API client.

use crate::language_models::LLMResult;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// User-provided identifier for correlating results.
    pub custom_id: String,
    /// Messages to send to the model.
    pub messages: Vec<lc_schema::Message>,
    /// Model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
    pub model: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
}

/// Batch identifier returned after submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchId(pub String);

/// Status of a batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// The batch is still being processed.
    InProgress,
    /// The batch completed successfully.
    Completed,
    /// The batch failed.
    Failed,
    /// The batch expired before completion.
    Expired,
    /// The batch was cancelled.
    Cancelled,
}

/// Result for a single request in a completed batch.
///
/// Not `Serialize`/`Deserialize`: the per-request error is the typed
/// [`BatchError`], which does not carry a JSON round-trip contract. Parsing
/// of the provider JSONL response happens on the internal `*ResultLine` types
/// in this module; `BatchResult` is built programmatically.
#[derive(Debug)]
pub struct BatchResult {
    /// Matches the `custom_id` from the original [`BatchRequest`].
    pub custom_id: String,
    /// The LLM response on success, or a typed error on failure.
    pub result: Result<LLMResult, BatchError>,
}

/// Batch provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchProvider {
    /// OpenAI batch API.
    OpenAI,
    /// Anthropic batch API.
    Anthropic,
}

/// Error type for batch operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BatchError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// API returned an error message.
    #[error("API error: {0}")]
    Api(String),
    /// The requested batch does not exist.
    #[error("batch not found: {0}")]
    NotFound(String),
    /// The batch has expired.
    #[error("batch expired")]
    Expired,
    /// JSON serialization / deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// The batch failed and results cannot be retrieved.
    #[error("batch failed: {0}")]
    Failed(String),
    /// Timed out waiting for the batch to complete.
    #[error("batch timed out after {0}ms")]
    Timeout(u64),
}

// ---------------------------------------------------------------------------
// Internal API response types
// ---------------------------------------------------------------------------

/// OpenAI batch creation response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAIBatchResponse {
    pub(crate) id: String,
    pub(crate) status: String,
    pub(crate) input_file_id: Option<String>,
    pub(crate) output_file_id: Option<String>,
    pub(crate) error_file_id: Option<String>,
}

/// OpenAI file upload response.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIFileResponse {
    pub(crate) id: String,
}

/// Anthropic batch creation response.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicBatchResponse {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) processing_status: Option<String>,
    #[serde(default)]
    pub(crate) request_counts: Option<AnthropicRequestCounts>,
}

/// Anthropic request counts in a batch.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AnthropicRequestCounts {
    #[serde(default)]
    pub(crate) processing: u32,
    #[serde(default)]
    pub(crate) succeeded: u32,
    #[serde(default)]
    pub(crate) errored: u32,
    #[serde(default)]
    pub(crate) canceled: u32,
    #[serde(default)]
    pub(crate) expired: u32,
}

/// A single result line from the Anthropic batch results JSONL stream.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResultLine {
    pub(crate) custom_id: String,
    pub(crate) result: AnthropicResultBody,
}

/// The body of a single Anthropic batch result.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResultBody {
    #[serde(rename = "type")]
    pub(crate) result_type: String,
    #[serde(default)]
    pub(crate) message: Option<AnthropicMessageBody>,
    #[serde(default)]
    pub(crate) error: Option<AnthropicErrorBody>,
}

/// Anthropic message body in a batch result.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicMessageBody {
    #[serde(default)]
    pub(crate) content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) usage: Option<AnthropicUsage>,
}

/// Anthropic content block.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub(crate) block_type: String,
    #[serde(default)]
    pub(crate) text: Option<String>,
}

/// Anthropic usage info.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicUsage {
    #[serde(default)]
    pub(crate) input_tokens: usize,
    #[serde(default)]
    pub(crate) output_tokens: usize,
}

/// Anthropic error body in a batch result.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub(crate) error_type: String,
    #[serde(default)]
    pub(crate) message: Option<String>,
}

/// A single result line from the OpenAI batch output JSONL.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIResultLine {
    pub(crate) custom_id: String,
    pub(crate) response: Option<OpenAIResponseBody>,
    pub(crate) error: Option<OpenAIErrorBody>,
}

/// OpenAI response body in a batch result line.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAIResponseBody {
    #[serde(default)]
    pub(crate) body: Option<OpenAIResponseBodyInner>,
    #[serde(default)]
    pub(crate) status_code: Option<u16>,
}

/// Inner body of an OpenAI response.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIResponseBodyInner {
    #[serde(default)]
    pub(crate) choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) usage: Option<OpenAIUsage>,
}

/// OpenAI choice in a batch result.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIChoice {
    #[serde(default)]
    pub(crate) message: Option<OpenAIMessageBody>,
}

/// OpenAI message body in a choice.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIMessageBody {
    #[serde(default)]
    pub(crate) content: Option<String>,
}

/// OpenAI usage info.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAIUsage {
    #[serde(default)]
    pub(crate) prompt_tokens: usize,
    #[serde(default)]
    pub(crate) completion_tokens: usize,
    #[serde(default)]
    pub(crate) total_tokens: usize,
}

/// OpenAI error body in a batch result line.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct OpenAIErrorBody {
    #[serde(default)]
    pub(crate) message: Option<String>,
    #[serde(default)]
    pub(crate) code: Option<String>,
}
