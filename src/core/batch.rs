// src/core/batch.rs
//! Batch API client for OpenAI and Anthropic batch processing.
//!
//! Provides a unified interface for submitting, polling, and retrieving
//! results from batch API endpoints:
//!
//! - **OpenAI Batch API**: POST /v1/batches (JSONL file upload + batch creation)
//! - **Anthropic Batch API**: POST /v1/messages/batches (inline requests array)
//!
//! # Example
//! ```
//! use langchainrust::core::batch::{BatchClient, BatchProvider, BatchRequest};
//! use langchainrust::Message;
//!
//! let client = BatchClient::new(BatchProvider::OpenAI, "sk-...");
//! let requests = vec![
//!     BatchRequest {
//!         custom_id: "req-1".into(),
//!         messages: vec![Message::human("Hello")],
//!         model: "gpt-4o".into(),
//!         temperature: None,
//!         max_tokens: None,
//!     },
//! ];
//! // let batch_id = client.submit(requests).await?;
//! // let results = client.submit_and_wait(requests, 5000, 300_000).await?;
//! ```

use crate::core::language_models::LLMResult;
use crate::schema::Message;
use serde::{Deserialize, Serialize};
use serde_json::json;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// User-provided identifier for correlating results.
    pub custom_id: String,
    /// Messages to send to the model.
    pub messages: Vec<Message>,
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
    InProgress,
    Completed,
    Failed,
    Expired,
    Cancelled,
}

/// Result for a single request in a completed batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Matches the `custom_id` from the original [`BatchRequest`].
    pub custom_id: String,
    /// The LLM response on success, or an error message on failure.
    pub result: Result<LLMResult, String>,
}

/// Batch provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchProvider {
    OpenAI,
    Anthropic,
}

/// Error type for batch operations.
#[derive(Debug, thiserror::Error)]
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
struct OpenAIBatchResponse {
    id: String,
    status: String,
    input_file_id: Option<String>,
    output_file_id: Option<String>,
    error_file_id: Option<String>,
}

/// OpenAI file upload response.
#[derive(Debug, Deserialize)]
struct OpenAIFileResponse {
    id: String,
}

/// Anthropic batch creation response.
#[derive(Debug, Deserialize)]
struct AnthropicBatchResponse {
    id: String,
    #[serde(default)]
    processing_status: Option<String>,
    #[serde(default)]
    request_counts: Option<AnthropicRequestCounts>,
}

/// Anthropic request counts in a batch.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicRequestCounts {
    #[serde(default)]
    processing: u32,
    #[serde(default)]
    succeeded: u32,
    #[serde(default)]
    errored: u32,
    #[serde(default)]
    canceled: u32,
    #[serde(default)]
    expired: u32,
}

/// A single result line from the Anthropic batch results JSONL stream.
#[derive(Debug, Deserialize)]
struct AnthropicResultLine {
    custom_id: String,
    result: AnthropicResultBody,
}

/// The body of a single Anthropic batch result.
#[derive(Debug, Deserialize)]
struct AnthropicResultBody {
    #[serde(rename = "type")]
    result_type: String,
    #[serde(default)]
    message: Option<AnthropicMessageBody>,
    #[serde(default)]
    error: Option<AnthropicErrorBody>,
}

/// Anthropic message body in a batch result.
#[derive(Debug, Deserialize)]
struct AnthropicMessageBody {
    #[serde(default)]
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

/// Anthropic content block.
#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

/// Anthropic usage info.
#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: usize,
    #[serde(default)]
    output_tokens: usize,
}

/// Anthropic error body in a batch result.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicErrorBody {
    #[serde(rename = "type")]
    error_type: String,
    #[serde(default)]
    message: Option<String>,
}

/// A single result line from the OpenAI batch output JSONL.
#[derive(Debug, Deserialize)]
struct OpenAIResultLine {
    custom_id: String,
    response: Option<OpenAIResponseBody>,
    error: Option<OpenAIErrorBody>,
}

/// OpenAI response body in a batch result line.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIResponseBody {
    #[serde(default)]
    body: Option<OpenAIResponseBodyInner>,
    #[serde(default)]
    status_code: Option<u16>,
}

/// Inner body of an OpenAI response.
#[derive(Debug, Deserialize)]
struct OpenAIResponseBodyInner {
    #[serde(default)]
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

/// OpenAI choice in a batch result.
#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    #[serde(default)]
    message: Option<OpenAIMessageBody>,
}

/// OpenAI message body in a choice.
#[derive(Debug, Deserialize)]
struct OpenAIMessageBody {
    #[serde(default)]
    content: Option<String>,
}

/// OpenAI usage info.
#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    #[serde(default)]
    prompt_tokens: usize,
    #[serde(default)]
    completion_tokens: usize,
    #[serde(default)]
    total_tokens: usize,
}

/// OpenAI error body in a batch result line.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIErrorBody {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<String>,
}

// ---------------------------------------------------------------------------
// BatchClient
// ---------------------------------------------------------------------------

/// Client for submitting, polling, and retrieving batch results.
pub struct BatchClient {
    http: reqwest::Client,
    // M35: API key stored as String — could leak via debug/display.
    // Consider using a Zeroize wrapper or secret type in a future refactor.
    api_key: String,
    provider: BatchProvider,
    base_url: String,
}

impl BatchClient {
    /// Creates a new `BatchClient` for the given provider.
    pub fn new(provider: BatchProvider, api_key: impl Into<String>) -> Self {
        let base_url = match provider {
            BatchProvider::OpenAI => "https://api.openai.com/v1".to_string(),
            BatchProvider::Anthropic => "https://api.anthropic.com/v1".to_string(),
        };
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            provider,
            base_url,
        }
    }

    /// Overrides the base URL (useful for proxies or compatible endpoints).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    // -- Auth header helpers ------------------------------------------------

    fn auth_headers(&self) -> Result<reqwest::header::HeaderMap, BatchError> {
        let mut headers = reqwest::header::HeaderMap::new();
        match self.provider {
            BatchProvider::OpenAI => {
                let value = format!("Bearer {}", self.api_key);
                let v = reqwest::header::HeaderValue::from_str(&value).map_err(|e| {
                    BatchError::Api(format!("invalid API key for Authorization header: {}", e))
                })?;
                headers.insert("Authorization", v);
            }
            BatchProvider::Anthropic => {
                let v = reqwest::header::HeaderValue::from_str(&self.api_key).map_err(|e| {
                    BatchError::Api(format!("invalid API key for x-api-key header: {}", e))
                })?;
                headers.insert("x-api-key", v);
                headers.insert(
                    "anthropic-version",
                    reqwest::header::HeaderValue::from_static("2023-06-01"),
                );
            }
        }
        Ok(headers)
    }

    // -- Message conversion -------------------------------------------------

    /// Convert a [`Message`] to the OpenAI chat format JSON value.
    fn message_to_openai(msg: &Message) -> serde_json::Value {
        match &msg.message_type {
            crate::schema::MessageType::System => json!({
                "role": "system",
                "content": msg.content,
            }),
            crate::schema::MessageType::Human => json!({
                "role": "user",
                "content": msg.content,
            }),
            crate::schema::MessageType::AI => {
                let mut m = json!({
                    "role": "assistant",
                    "content": msg.content,
                });
                if let Some(tc) = &msg.tool_calls {
                    m["tool_calls"] = serde_json::to_value(tc).unwrap_or(serde_json::Value::Null);
                }
                m
            }
            crate::schema::MessageType::Tool { tool_call_id } => json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": msg.content,
            }),
        }
    }

    /// Convert a [`Message`] to the Anthropic chat format JSON value.
    ///
    /// Anthropic uses a `system` top-level parameter for system messages,
    /// but in the messages array it only accepts `user` and `assistant` roles.
    /// System messages are mapped to `user` with a `[System]` prefix to
    /// preserve the instruction while conforming to the API constraints.
    /// Tool results are mapped to `user` role per Anthropic's convention.
    fn message_to_anthropic(msg: &Message) -> serde_json::Value {
        match &msg.message_type {
            // H40: SystemBSystem messages should be extracted to top-level `system` parameter
            // by the caller (submit_anthropic), not mapped as user messages.
            // Here we map them as system role for identification.
            crate::schema::MessageType::System => json!({
                "role": "system",
                "content": msg.content,
            }),
            crate::schema::MessageType::Human => json!({
                "role": "user",
                "content": msg.content,
            }),
            crate::schema::MessageType::AI => json!({
                "role": "assistant",
                "content": msg.content,
            }),
            // H40: Tool messages should use Anthropic tool_result content block format
            crate::schema::MessageType::Tool { tool_call_id } => json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": msg.content,
                }],
            }),
        }
    }

    // -- Submit -------------------------------------------------------------

    /// Submit a batch of requests. Returns the batch ID.
    ///
    /// For **OpenAI**: uploads a JSONL file via the Files API, then creates a
    /// batch via `POST /v1/batches` referencing the uploaded file.
    ///
    /// For **Anthropic**: creates a batch via `POST /v1/messages/batches` with
    /// the requests array inline.
    pub async fn submit(&self, requests: Vec<BatchRequest>) -> Result<BatchId, BatchError> {
        match self.provider {
            BatchProvider::OpenAI => self.submit_openai(requests).await,
            BatchProvider::Anthropic => self.submit_anthropic(requests).await,
        }
    }

    async fn submit_openai(&self, requests: Vec<BatchRequest>) -> Result<BatchId, BatchError> {
        // Build JSONL content for the input file.
        let jsonl_lines: Vec<String> = requests
            .iter()
            .map(|req| {
                let openai_msgs: Vec<serde_json::Value> =
                    req.messages.iter().map(Self::message_to_openai).collect();
                let mut body = json!({
                    "model": req.model,
                    "messages": openai_msgs,
                });
                if let Some(t) = req.temperature {
                    body["temperature"] = json!(t);
                }
                if let Some(m) = req.max_tokens {
                    body["max_tokens"] = json!(m);
                }
                let line = json!({
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                });
                // H39: Return error instead of empty string on serialization failure
                serde_json::to_string(&line).map_err(BatchError::Serialization)
            })
            .collect::<Result<Vec<String>, BatchError>>()?;
        let jsonl_content = jsonl_lines.join("\n");

        // Step 1: Upload the JSONL file.
        let file_id = self.upload_openai_file(&jsonl_content).await?;

        // Step 2: Create the batch.
        let headers = self.auth_headers()?;
        let body = json!({
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        });

        let resp = self
            .http
            .post(format!("{}/batches", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(
                "batches endpoint not found".to_string(),
            ));
        }

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "batch creation failed ({}): {}",
                status, text
            )));
        }

        let batch: OpenAIBatchResponse = serde_json::from_str(&text)?;
        Ok(BatchId(batch.id))
    }

    /// Upload a JSONL file to the OpenAI Files API and return the file ID.
    async fn upload_openai_file(&self, jsonl_content: &str) -> Result<String, BatchError> {
        let headers = self.auth_headers()?;
        let file_part = reqwest::multipart::Part::text(jsonl_content.to_string())
            .file_name("batch_input.jsonl")
            .mime_str("application/jsonl")
            .map_err(|e| BatchError::Api(format!("mime error: {e}")))?;

        let form = reqwest::multipart::Form::new()
            .text("purpose", "batch")
            .part("file", file_part);

        let resp = self
            .http
            .post(format!("{}/files", self.base_url))
            .headers(headers)
            .multipart(form)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "file upload failed ({}): {}",
                status, text
            )));
        }

        let file_resp: OpenAIFileResponse = serde_json::from_str(&text)?;
        Ok(file_resp.id)
    }

    async fn submit_anthropic(&self, requests: Vec<BatchRequest>) -> Result<BatchId, BatchError> {
        let req_items: Vec<serde_json::Value> = requests
            .iter()
            .map(|req| {
                // H40: Extract system messages to top-level `system` parameter,
                // and filter them from the messages array (Anthropic API requirement).
                let mut system_text = String::new();
                let anthropic_msgs: Vec<serde_json::Value> = req
                    .messages
                    .iter()
                    .filter_map(|msg| {
                        if matches!(msg.message_type, crate::schema::MessageType::System) {
                            if !system_text.is_empty() {
                                system_text.push('\n');
                            }
                            system_text.push_str(&msg.content);
                            None
                        } else {
                            Some(Self::message_to_anthropic(msg))
                        }
                    })
                    .collect();
                let mut body = json!({
                    "model": req.model,
                    "max_tokens": req.max_tokens.unwrap_or(4096),
                    "messages": anthropic_msgs,
                });
                if !system_text.is_empty() {
                    body["system"] = json!(system_text);
                }
                if let Some(t) = req.temperature {
                    body["temperature"] = json!(t);
                }
                json!({
                    "custom_id": req.custom_id,
                    "params": body,
                })
            })
            .collect();

        let headers = self.auth_headers()?;
        let body = json!({
            "requests": req_items,
        });

        let resp = self
            .http
            .post(format!("{}/messages/batches", self.base_url))
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(
                "messages/batches endpoint not found".to_string(),
            ));
        }

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "batch creation failed ({}): {}",
                status, text
            )));
        }

        let batch: AnthropicBatchResponse = serde_json::from_str(&text)?;
        Ok(BatchId(batch.id))
    }

    // -- Poll ---------------------------------------------------------------

    /// Poll the status of a batch job.
    pub async fn poll(&self, id: &BatchId) -> Result<BatchStatus, BatchError> {
        match self.provider {
            BatchProvider::OpenAI => self.poll_openai(id).await,
            BatchProvider::Anthropic => self.poll_anthropic(id).await,
        }
    }

    async fn poll_openai(&self, id: &BatchId) -> Result<BatchStatus, BatchError> {
        let headers = self.auth_headers()?;
        let resp = self
            .http
            .get(format!("{}/batches/{}", self.base_url, id.0))
            .headers(headers)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(id.0.clone()));
        }

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "poll failed ({}): {}",
                status, text
            )));
        }

        let batch: OpenAIBatchResponse = serde_json::from_str(&text)?;
        Ok(match batch.status.as_str() {
            "in_progress" | "validating" | "finalizing" => BatchStatus::InProgress,
            "completed" => BatchStatus::Completed,
            "failed" => BatchStatus::Failed,
            "expired" => BatchStatus::Expired,
            "cancelling" | "cancelled" => BatchStatus::Cancelled,
            other => return Err(BatchError::Api(format!("unknown batch status: {}", other))),
        })
    }

    async fn poll_anthropic(&self, id: &BatchId) -> Result<BatchStatus, BatchError> {
        let headers = self.auth_headers()?;
        let resp = self
            .http
            .get(format!("{}/messages/batches/{}", self.base_url, id.0))
            .headers(headers)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(id.0.clone()));
        }

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "poll failed ({}): {}",
                status, text
            )));
        }

        let batch: AnthropicBatchResponse = serde_json::from_str(&text)?;
        let proc = batch.processing_status.as_deref().unwrap_or("in_progress");
        Ok(match proc {
            "in_progress" => BatchStatus::InProgress,
            "ended" => {
                // Determine the terminal state from request_counts.
                if let Some(counts) = batch.request_counts {
                    if counts.errored > 0 && counts.succeeded == 0 {
                        BatchStatus::Failed
                    } else if counts.expired > 0 && counts.succeeded == 0 {
                        BatchStatus::Expired
                    } else {
                        BatchStatus::Completed
                    }
                } else {
                    BatchStatus::Completed
                }
            }
            other => {
                return Err(BatchError::Api(format!(
                    "unknown processing_status: {}",
                    other
                )))
            }
        })
    }

    // -- Results ------------------------------------------------------------

    /// Retrieve results of a completed batch.
    pub async fn results(&self, id: &BatchId) -> Result<Vec<BatchResult>, BatchError> {
        match self.provider {
            BatchProvider::OpenAI => self.results_openai(id).await,
            BatchProvider::Anthropic => self.results_anthropic(id).await,
        }
    }

    async fn results_openai(&self, id: &BatchId) -> Result<Vec<BatchResult>, BatchError> {
        // First, get the batch metadata to find the output file ID.
        let headers = self.auth_headers()?;
        let resp = self
            .http
            .get(format!("{}/batches/{}", self.base_url, id.0))
            .headers(headers.clone())
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(id.0.clone()));
        }

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "fetch batch metadata failed ({}): {}",
                status, text
            )));
        }

        let batch: OpenAIBatchResponse = serde_json::from_str(&text)?;

        // If the batch has an error file but no output file, report failure.
        let output_file_id = match batch.output_file_id {
            Some(fid) => fid,
            None => {
                if let Some(err_fid) = batch.error_file_id {
                    // Download the error file for details.
                    return self.download_openai_error_file(&err_fid, &headers).await;
                }
                return Err(BatchError::Failed(
                    "batch has no output file and no error file".to_string(),
                ));
            }
        };

        // Download the output JSONL file.
        let file_resp = self
            .http
            .get(format!(
                "{}/files/{}/content",
                self.base_url, output_file_id
            ))
            .headers(headers)
            .send()
            .await?;

        let file_status = file_resp.status();
        let file_text = file_resp.text().await?;

        if !file_status.is_success() {
            return Err(BatchError::Api(format!(
                "download output file failed ({}): {}",
                file_status, file_text
            )));
        }

        self.parse_openai_results_jsonl(&file_text)
    }

    fn parse_openai_results_jsonl(&self, text: &str) -> Result<Vec<BatchResult>, BatchError> {
        let mut results = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed: OpenAIResultLine = serde_json::from_str(line)?;
            let result = if let Some(err) = parsed.error {
                Err(err.message.unwrap_or_else(|| "unknown error".to_string()))
            } else if let Some(resp_body) = parsed.response {
                if let Some(inner) = resp_body.body {
                    let content = inner
                        .choices
                        .first()
                        .and_then(|c| c.message.as_ref())
                        .and_then(|m| m.content.clone())
                        .unwrap_or_default();

                    let token_usage =
                        inner
                            .usage
                            .map(|u| crate::core::language_models::TokenUsage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            });

                    Ok(LLMResult {
                        content,
                        model: inner.model.unwrap_or_default(),
                        token_usage,
                        tool_calls: None,
                        thinking_content: None,
                    })
                } else {
                    Err("empty response body".to_string())
                }
            } else {
                Err("no response and no error".to_string())
            };
            results.push(BatchResult {
                custom_id: parsed.custom_id,
                result,
            });
        }
        Ok(results)
    }

    async fn download_openai_error_file(
        &self,
        error_file_id: &str,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<Vec<BatchResult>, BatchError> {
        let resp = self
            .http
            .get(format!("{}/files/{}/content", self.base_url, error_file_id))
            .headers(headers.clone())
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "download error file failed ({}): {}",
                status, text
            )));
        }

        // Parse error JSONL the same way as output.
        self.parse_openai_results_jsonl(&text)
    }

    async fn results_anthropic(&self, id: &BatchId) -> Result<Vec<BatchResult>, BatchError> {
        let headers = self.auth_headers()?;
        let resp = self
            .http
            .get(format!(
                "{}/messages/batches/{}/results",
                self.base_url, id.0
            ))
            .headers(headers)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(id.0.clone()));
        }

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            return Err(BatchError::Api(format!(
                "fetch results failed ({}): {}",
                status, text
            )));
        }

        self.parse_anthropic_results_jsonl(&text)
    }

    fn parse_anthropic_results_jsonl(&self, text: &str) -> Result<Vec<BatchResult>, BatchError> {
        let mut results = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed: AnthropicResultLine = serde_json::from_str(line)?;
            let result = match parsed.result.result_type.as_str() {
                "succeeded" => {
                    if let Some(msg) = parsed.result.message {
                        let content = msg
                            .content
                            .iter()
                            .filter_map(|b| {
                                if b.block_type == "text" {
                                    b.text.clone()
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("");

                        let token_usage =
                            msg.usage.map(|u| crate::core::language_models::TokenUsage {
                                prompt_tokens: u.input_tokens,
                                completion_tokens: u.output_tokens,
                                total_tokens: u.input_tokens + u.output_tokens,
                            });

                        Ok(LLMResult {
                            content,
                            model: msg.model,
                            token_usage,
                            tool_calls: None,
                            thinking_content: None,
                        })
                    } else {
                        Err("succeeded result missing message body".to_string())
                    }
                }
                "errored" => {
                    let err_msg = parsed
                        .result
                        .error
                        .and_then(|e| e.message)
                        .unwrap_or_else(|| "unknown error".to_string());
                    Err(err_msg)
                }
                "expired" => Err("request expired".to_string()),
                "canceled" => Err("request canceled".to_string()),
                other => Err(format!("unknown result type: {}", other)),
            };
            results.push(BatchResult {
                custom_id: parsed.custom_id,
                result,
            });
        }
        Ok(results)
    }

    // -- Cancel -------------------------------------------------------------

    /// Cancel a batch job.
    pub async fn cancel(&self, id: &BatchId) -> Result<(), BatchError> {
        match self.provider {
            BatchProvider::OpenAI => self.cancel_openai(id).await,
            BatchProvider::Anthropic => Err(BatchError::Api(
                "Anthropic batch API does not support cancellation".to_string(),
            )),
        }
    }

    async fn cancel_openai(&self, id: &BatchId) -> Result<(), BatchError> {
        let headers = self.auth_headers()?;
        let resp = self
            .http
            .post(format!("{}/batches/{}/cancel", self.base_url, id.0))
            .headers(headers)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(BatchError::NotFound(id.0.clone()));
        }

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(BatchError::Api(format!(
                "cancel failed ({}): {}",
                status, text
            )));
        }

        Ok(())
    }

    // -- Convenience --------------------------------------------------------

    /// Submit and wait until complete, then return results.
    ///
    /// Polls every `poll_interval_ms` milliseconds. Returns
    /// [`BatchError::Timeout`] if `max_wait_ms` is exceeded.
    pub async fn submit_and_wait(
        &self,
        requests: Vec<BatchRequest>,
        poll_interval_ms: u64,
        max_wait_ms: u64,
    ) -> Result<Vec<BatchResult>, BatchError> {
        let batch_id = self.submit(requests).await?;
        let start = std::time::Instant::now();
        let poll_duration = std::time::Duration::from_millis(poll_interval_ms);
        let max_duration = std::time::Duration::from_millis(max_wait_ms);

        loop {
            let status = self.poll(&batch_id).await?;

            match status {
                BatchStatus::Completed => {
                    return self.results(&batch_id).await;
                }
                BatchStatus::Failed => {
                    return Err(BatchError::Failed(format!("batch {} failed", batch_id.0)));
                }
                BatchStatus::Expired => {
                    return Err(BatchError::Expired);
                }
                BatchStatus::Cancelled => {
                    return Err(BatchError::Api(format!(
                        "batch {} was cancelled",
                        batch_id.0
                    )));
                }
                BatchStatus::InProgress => {
                    // Continue polling.
                }
            }

            if start.elapsed() >= max_duration {
                return Err(BatchError::Timeout(max_wait_ms));
            }

            tokio::time::sleep(poll_duration).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_request_serialization() {
        let req = BatchRequest {
            custom_id: "req-1".into(),
            messages: vec![Message::human("Hello")],
            model: "gpt-4o".into(),
            temperature: Some(0.7),
            max_tokens: Some(256),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("req-1"));
        assert!(json.contains("gpt-4o"));
        let back: BatchRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.custom_id, "req-1");
        assert_eq!(back.model, "gpt-4o");
    }

    #[test]
    fn batch_id_clone_eq() {
        let id = BatchId("batch_abc".into());
        let id2 = id.clone();
        assert_eq!(id.0, id2.0);
    }

    #[test]
    fn batch_status_eq() {
        assert_eq!(BatchStatus::InProgress, BatchStatus::InProgress);
        assert_ne!(BatchStatus::Completed, BatchStatus::Failed);
    }

    #[test]
    fn batch_error_display() {
        let e = BatchError::Api("something went wrong".to_string());
        assert!(e.to_string().contains("something went wrong"));

        let e = BatchError::NotFound("batch_123".to_string());
        assert!(e.to_string().contains("batch_123"));

        let e = BatchError::Expired;
        assert!(e.to_string().contains("expired"));

        let e = BatchError::Timeout(60_000);
        assert!(e.to_string().contains("60000"));
    }

    #[test]
    fn openai_result_line_parsing() {
        let json = r#"{
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": "Hello!"}}],
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
            }
        }"#;
        let line: OpenAIResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-1");
        let body = line.response.unwrap().body.unwrap();
        assert_eq!(body.model.unwrap(), "gpt-4o");
        assert_eq!(
            body.choices[0].message.as_ref().unwrap().content.as_deref(),
            Some("Hello!")
        );
    }

    #[test]
    fn openai_result_line_with_error() {
        let json = r#"{
            "custom_id": "req-2",
            "error": {
                "message": "rate limit exceeded",
                "code": "rate_limit"
            }
        }"#;
        let line: OpenAIResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-2");
        let err = line.error.unwrap();
        assert_eq!(err.message.unwrap(), "rate limit exceeded");
    }

    #[test]
    fn anthropic_result_line_parsing() {
        let json = r#"{
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 8, "output_tokens": 4}
                }
            }
        }"#;
        let line: AnthropicResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.custom_id, "req-1");
        assert_eq!(line.result.result_type, "succeeded");
        let msg = line.result.message.unwrap();
        assert_eq!(msg.model, "claude-3-5-sonnet-20241022");
        assert_eq!(msg.content[0].text.as_deref(), Some("Hi there!"));
    }

    #[test]
    fn anthropic_result_line_errored() {
        let json = r#"{
            "custom_id": "req-2",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request",
                    "message": "max_tokens too large"
                }
            }
        }"#;
        let line: AnthropicResultLine = serde_json::from_str(json).unwrap();
        assert_eq!(line.result.result_type, "errored");
        let err = line.result.error.unwrap();
        assert_eq!(err.message.unwrap(), "max_tokens too large");
    }

    #[test]
    fn parse_openai_jsonl() {
        let client = BatchClient::new(BatchProvider::OpenAI, "test-key");
        let jsonl = r#"{"custom_id":"r1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"A1"}}],"model":"gpt-4o","usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}}}
{"custom_id":"r2","error":{"message":"fail","code":"err"}}"#;
        let results = client.parse_openai_results_jsonl(jsonl).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "r1");
        assert!(results[0].result.is_ok());
        let llm = results[0].result.as_ref().unwrap();
        assert_eq!(llm.content, "A1");
        assert_eq!(llm.model, "gpt-4o");
        assert!(llm.token_usage.is_some());
        assert_eq!(results[1].custom_id, "r2");
        assert!(results[1].result.is_err());
    }

    #[test]
    fn parse_anthropic_jsonl() {
        let client = BatchClient::new(BatchProvider::Anthropic, "test-key");
        let jsonl = r#"{"custom_id":"r1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"B1"}],"model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":6,"output_tokens":2}}}}
{"custom_id":"r2","result":{"type":"errored","error":{"type":"invalid","message":"bad"}}}"#;
        let results = client.parse_anthropic_results_jsonl(jsonl).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].custom_id, "r1");
        assert!(results[0].result.is_ok());
        let llm = results[0].result.as_ref().unwrap();
        assert_eq!(llm.content, "B1");
        assert!(llm.token_usage.is_some());
        assert_eq!(results[1].custom_id, "r2");
        assert!(results[1].result.is_err());
    }

    #[test]
    fn message_to_openai_format() {
        let msg = Message::human("Hello");
        let val = BatchClient::message_to_openai(&msg);
        assert_eq!(val["role"], "user");
        assert_eq!(val["content"], "Hello");

        let sys = Message::system("You are helpful");
        let val = BatchClient::message_to_openai(&sys);
        assert_eq!(val["role"], "system");
    }

    #[test]
    fn message_to_anthropic_format() {
        let msg = Message::human("Hello");
        let val = BatchClient::message_to_anthropic(&msg);
        assert_eq!(val["role"], "user");
        assert_eq!(val["content"], "Hello");

        let ai = Message::ai("Response");
        let val = BatchClient::message_to_anthropic(&ai);
        assert_eq!(val["role"], "assistant");

        // H40: System messages now mapped to "system" role (extracted to top-level by caller)
        let sys = Message::system("You are helpful");
        let val = BatchClient::message_to_anthropic(&sys);
        assert_eq!(val["role"], "system");
        assert_eq!(val["content"], "You are helpful");
    }

    #[test]
    fn batch_client_new_default_urls() {
        let openai = BatchClient::new(BatchProvider::OpenAI, "key");
        assert!(openai.base_url.contains("openai.com"));

        let anthropic = BatchClient::new(BatchProvider::Anthropic, "key");
        assert!(anthropic.base_url.contains("anthropic.com"));
    }

    #[test]
    fn batch_client_with_base_url() {
        let client = BatchClient::new(BatchProvider::OpenAI, "key")
            .with_base_url("https://custom.api.com/v1");
        assert_eq!(client.base_url, "https://custom.api.com/v1");
    }

    #[test]
    fn openai_batch_response_deserialize() {
        let json = r#"{
            "id": "batch_abc123",
            "status": "in_progress",
            "input_file_id": "file-xyz",
            "output_file_id": null,
            "error_file_id": null
        }"#;
        let resp: OpenAIBatchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "batch_abc123");
        assert_eq!(resp.status, "in_progress");
        assert_eq!(resp.input_file_id.unwrap(), "file-xyz");
        assert!(resp.output_file_id.is_none());
    }

    #[test]
    fn anthropic_batch_response_deserialize() {
        let json = r#"{
            "id": "msgbatch_abc",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 5,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        }"#;
        let resp: AnthropicBatchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msgbatch_abc");
        assert_eq!(resp.processing_status.as_deref(), Some("in_progress"));
        let counts = resp.request_counts.unwrap();
        assert_eq!(counts.processing, 5);
    }

    #[test]
    fn batch_result_serialization() {
        let ok_result = BatchResult {
            custom_id: "r1".into(),
            result: Ok(LLMResult {
                content: "Hello".into(),
                model: "gpt-4o".into(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            }),
        };
        let json = serde_json::to_string(&ok_result).unwrap();
        assert!(json.contains("r1"));
        assert!(json.contains("Hello"));

        let err_result = BatchResult {
            custom_id: "r2".into(),
            result: Err("failed".into()),
        };
        let json = serde_json::to_string(&err_result).unwrap();
        assert!(json.contains("failed"));
    }
}
