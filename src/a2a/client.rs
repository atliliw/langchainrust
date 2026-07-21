//! A2A Client - connects to remote A2A agents over HTTP.
//!
//! The client uses `reqwest` (already in the project dependencies) to
//! communicate with A2A servers. It supports:
//!
//! - Fetching an agent card (`GET /.well-known/agent.json`)
//! - Sending a task (`tasks/send`)
//! - Getting a task (`tasks/get`)
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::a2a::{A2AClient, A2AMessage};
//!
//! let client = A2AClient::new("http://localhost:8080".to_string());
//! let card = client.get_agent_card().await?;
//! let task = client.send_task(A2AMessage::user("hello")).await?;
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use super::protocol::{A2ARequest, A2AResponse, A2ATask, A2AMessage, AgentCard, A2AErrorData};

/// Errors that can occur during A2A client operations.
#[derive(Debug, thiserror::Error)]
pub enum A2AError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(String),

    /// JSON parse error.
    #[error("Parse error: {0}")]
    Parse(String),

    /// API-level error (returned by the remote agent).
    #[error("API error [{code}]: {message}")]
    Api {
        code: i32,
        message: String,
    },

    /// Request timed out.
    #[error("Timeout: {0}")]
    Timeout(String),
}

impl From<reqwest::Error> for A2AError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            A2AError::Timeout(err.to_string())
        } else {
            A2AError::Http(err.to_string())
        }
    }
}

impl From<A2AErrorData> for A2AError {
    fn from(err: A2AErrorData) -> Self {
        A2AError::Api {
            code: err.code,
            message: err.message,
        }
    }
}

/// A2A Client - communicates with remote A2A agents.
pub struct A2AClient {
    /// Base URL of the remote agent (e.g. "http://localhost:8080").
    base_url: String,
    /// HTTP client.
    http: reqwest::Client,
    /// Monotonic request ID counter.
    next_id: AtomicU64,
}

impl A2AClient {
    /// Create a new client targeting the given base URL.
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a client with a custom `reqwest::Client` (for timeouts, etc.).
    pub fn with_http_client(base_url: String, http: reqwest::Client) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
            next_id: AtomicU64::new(1),
        }
    }

    /// Allocate the next request ID.
    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Fetch the agent card from `GET /.well-known/agent.json`.
    pub async fn get_agent_card(&self) -> Result<AgentCard, A2AError> {
        let url = format!("{}/.well-known/agent.json", self.base_url);
        let resp = self.http.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2AError::Http(format!(
                "Agent card request failed with status {}",
                status
            )));
        }
        let card: AgentCard = resp
            .json()
            .await
            .map_err(|e| A2AError::Parse(format!("Failed to parse agent card: {}", e)))?;
        Ok(card)
    }

    /// Send a task to the remote agent (`tasks/send`).
    pub async fn send_task(&self, message: A2AMessage) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = A2ARequest::send_task(id, &message);
        let resp = self.post_request(req).await?;

        // Extract the task from the response result.
        let result = resp.into_result().map_err(A2AError::from)?;
        let task: A2ATask = result
            .get("task")
            .ok_or_else(|| A2AError::Parse("Missing 'task' in response".to_string()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task: {}", e)))
            })?;
        Ok(task)
    }

    /// Get a task by ID (`tasks/get`).
    pub async fn get_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = A2ARequest::get_task(id, task_id);
        let resp = self.post_request(req).await?;

        let result = resp.into_result().map_err(A2AError::from)?;
        let task: A2ATask = result
            .get("task")
            .ok_or_else(|| A2AError::Parse("Missing 'task' in response".to_string()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task: {}", e)))
            })?;
        Ok(task)
    }

    /// Cancel a task by ID (`tasks/cancel`).
    pub async fn cancel_task(&self, task_id: &str) -> Result<A2ATask, A2AError> {
        let id = self.alloc_id();
        let req = A2ARequest::cancel_task(id, task_id);
        let resp = self.post_request(req).await?;

        let result = resp.into_result().map_err(A2AError::from)?;
        let task: A2ATask = result
            .get("task")
            .ok_or_else(|| A2AError::Parse("Missing 'task' in response".to_string()))
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| A2AError::Parse(format!("Failed to parse task: {}", e)))
            })?;
        Ok(task)
    }

    /// Send a raw A2A request via POST to the agent endpoint.
    pub async fn post_request(&self, req: A2ARequest) -> Result<A2AResponse, A2AError> {
        let url = format!("{}/", self.base_url);
        let resp = self.http.post(&url).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            return Err(A2AError::Http(format!(
                "A2A request failed with status {}",
                status
            )));
        }
        let a2a_resp: A2AResponse = resp
            .json()
            .await
            .map_err(|e| A2AError::Parse(format!("Failed to parse A2A response: {}", e)))?;
        Ok(a2a_resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a2a_error_from_reqwest_timeout() {
        // We can't easily create a reqwest::Error, so test the variant exists.
        let err = A2AError::Timeout("connection timed out".to_string());
        assert!(err.to_string().contains("Timeout"));
    }

    #[test]
    fn a2a_error_from_reqwest_http() {
        let err = A2AError::Http("404 not found".to_string());
        assert!(err.to_string().contains("HTTP error"));
    }

    #[test]
    fn a2a_error_from_error_data() {
        let data = A2AErrorData::method_not_found();
        let err: A2AError = data.into();
        match err {
            A2AError::Api { code, message } => {
                assert_eq!(code, -32601);
                assert!(message.contains("Method not found"));
            }
            _ => panic!("Expected Api variant"),
        }
    }

    #[test]
    fn a2a_error_parse() {
        let err = A2AError::Parse("bad json".to_string());
        assert!(err.to_string().contains("Parse error"));
    }

    #[test]
    fn client_new_trims_trailing_slash() {
        let client = A2AClient::new("http://localhost:8080/".to_string());
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn client_alloc_id_increments() {
        let client = A2AClient::new("http://localhost:8080".to_string());
        assert_eq!(client.alloc_id(), 1);
        assert_eq!(client.alloc_id(), 2);
        assert_eq!(client.alloc_id(), 3);
    }

    #[test]
    fn client_with_custom_http() {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        let client = A2AClient::with_http_client("http://localhost:8080".to_string(), http);
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[tokio::test]
    async fn get_agent_card_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string());
        let result = client.get_agent_card().await;
        assert!(result.is_err());
        match result.unwrap_err() {
            A2AError::Http(_) | A2AError::Timeout(_) => {} // expected
            other => panic!("Expected Http or Timeout error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn send_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string());
        let result = client.send_task(A2AMessage::user("hello")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string());
        let result = client.get_task("task-123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cancel_task_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string());
        let result = client.cancel_task("task-123").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn post_request_invalid_url() {
        let client = A2AClient::new("http://localhost:19999".to_string());
        let req = A2ARequest::new(1, "test", None);
        let result = client.post_request(req).await;
        assert!(result.is_err());
    }
}
