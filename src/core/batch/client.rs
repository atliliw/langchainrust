// src/core/batch/client.rs
//! BatchClient struct and all its method implementations.

use super::types::*;
use crate::core::language_models::LLMResult;
use crate::schema::Message;
use serde_json::json;

/// Client for submitting, polling, and retrieving batch results.
pub struct BatchClient {
    pub(crate) http: reqwest::Client,
    // M35: API key stored as String — could leak via debug/display.
    // Consider using a Zeroize wrapper or secret type in a future refactor.
    pub(crate) api_key: String,
    pub(crate) provider: BatchProvider,
    pub(crate) base_url: String,
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
    pub(crate) fn message_to_openai(msg: &Message) -> serde_json::Value {
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
    pub(crate) fn message_to_anthropic(msg: &Message) -> serde_json::Value {
        match &msg.message_type {
            // H40: System messages should be extracted to top-level `system` parameter
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

    pub(crate) fn parse_openai_results_jsonl(
        &self,
        text: &str,
    ) -> Result<Vec<BatchResult>, BatchError> {
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

    pub(crate) fn parse_anthropic_results_jsonl(
        &self,
        text: &str,
    ) -> Result<Vec<BatchResult>, BatchError> {
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
