//! A2A Server - handler functions for the Agent-to-Agent protocol.
//!
//! Provides `A2AServer` which holds an underlying agent (a `BaseChain` or
//! `BaseTool`) and exposes handler functions that can be plugged into any
//! HTTP framework (axum, actix, warp, etc.) rather than running its own
//! server.
//!
//! # Endpoints
//!
//! - `GET /.well-known/agent.json` -> returns `AgentCard` (via `get_agent_card`)
//! - `POST /` -> accepts `A2ARequest`, dispatches, returns `A2AResponse`
//!   (via `handle_a2a_request`)
//!
//! # Task Persistence
//!
//! Tasks are stored in-memory using a `RwLock<HashMap>`. This allows
//! `tasks/get` to retrieve previously created tasks and `tasks/cancel`
//! to transition existing tasks to `Cancelled` status.
//!
//! For production use with persistence across restarts, wrap `A2AServer`
//! with your own task store backed by a database.
//!
//! # Example
//!
//! ```ignore
//! use lc_a2a::{A2AServer, AgentCard};
//! use lc_chains::LLMChain;
//! use std::sync::Arc;
//!
//! let chain = Arc::new(LLMChain::new(llm, "You are a helpful assistant"));
//! let server = A2AServer::new(chain)
//!     .with_card(AgentCard::new("my-agent", "A helpful agent", "http://localhost:8080"));
//!
//! // In your HTTP handler:
//! let response = server.handle_a2a_request(request).await;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::RwLock;

use lc_chains::base::BaseChain;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskResult, AgentCard,
    TaskStatus,
};

/// Stored task data including the result if completed.
#[derive(Debug, Clone)]
struct StoredTask {
    task: A2ATask,
    result: Option<A2ATaskResult>,
}

/// A2A Server - wraps an agent and provides handler functions.
///
/// The server does NOT start its own HTTP listener. Instead, it provides
/// Default maximum number of tasks stored before LRU eviction.
const DEFAULT_MAX_TASKS: usize = 10_000;

/// `handle_a2a_request()` and `get_agent_card()` that you can call from
/// any HTTP framework's route handler.
///
/// Tasks are stored in-memory so that `tasks/get` can retrieve them and
/// `tasks/cancel` can transition their status. When the task store exceeds
/// `max_tasks`, the oldest completed/failed/cancelled tasks are evicted first.
pub struct A2AServer {
    /// The underlying chain/agent.
    chain: Arc<dyn BaseChain>,
    /// The agent card metadata.
    card: AgentCard,
    /// In-memory task store.
    tasks: RwLock<HashMap<String, StoredTask>>,
    /// Maximum number of tasks before eviction.
    max_tasks: usize,
}

impl A2AServer {
    /// Create a new A2A server backed by a `BaseChain`.
    pub fn new(chain: Arc<dyn BaseChain>) -> Self {
        let card = AgentCard::new(
            chain.name(),
            format!("Agent backed by {}", chain.name()),
            "http://localhost:8080",
        );
        Self {
            chain,
            card,
            tasks: RwLock::new(HashMap::new()),
            max_tasks: DEFAULT_MAX_TASKS,
        }
    }

    /// Set the maximum number of tasks before LRU eviction.
    pub fn with_max_tasks(mut self, max: usize) -> Self {
        self.max_tasks = max.max(1);
        self
    }

    /// Evict oldest completed/failed/cancelled tasks if over capacity.
    async fn evict_if_needed(&self) {
        let mut tasks = self.tasks.write().await;
        if tasks.len() <= self.max_tasks {
            return;
        }

        // Evict terminal tasks (completed/failed/cancelled) to make room
        let terminal_ids: Vec<String> = tasks
            .iter()
            .filter(|(_, t)| {
                matches!(
                    t.task.status,
                    TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled
                )
            })
            .map(|(id, _)| id.clone())
            .collect();

        let excess = tasks.len().saturating_sub(self.max_tasks);
        for id in terminal_ids.into_iter().take(excess) {
            tasks.remove(&id);
        }
    }

    /// Set a custom agent card.
    pub fn with_card(mut self, card: AgentCard) -> Self {
        self.card = card;
        self
    }

    /// Get the agent card (for `GET /.well-known/agent.json`).
    pub fn get_agent_card(&self) -> &AgentCard {
        &self.card
    }

    /// Handle an incoming A2A request (for `POST /`).
    ///
    /// Dispatches based on the request method:
    /// - `tasks/send` -> invoke the chain and return a task result
    /// - `tasks/get` -> return a stored task
    /// - `tasks/cancel` -> cancel a stored task
    /// - unknown method -> method_not_found error
    pub async fn handle_a2a_request(&self, req: A2ARequest) -> A2AResponse {
        match req.method.as_str() {
            "tasks/send" => self.handle_tasks_send(req).await,
            "tasks/get" => self.handle_tasks_get(req).await,
            "tasks/cancel" => self.handle_tasks_cancel(req).await,
            _ => A2AResponse::from_error_data(req.id, A2AErrorData::method_not_found()),
        }
    }

    /// Handle `tasks/send`: invoke the chain with the message content.
    async fn handle_tasks_send(&self, req: A2ARequest) -> A2AResponse {
        let params = match req.params {
            Some(p) => p,
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing params for tasks/send"),
                )
            }
        };

        // Extract the message from params.
        let message: A2AMessage = match params.get("message") {
            Some(msg_val) => serde_json::from_value(msg_val.clone()).unwrap_or_else(|_| {
                A2AMessage::new(
                    "user",
                    msg_val
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )
            }),
            None => {
                // Fallback: treat the entire params as the input content.
                A2AMessage::user(params.to_string())
            }
        };

        // Build chain input from the message content.
        let inputs: HashMap<String, Value> = {
            let mut map = HashMap::new();
            // Use the first input key expected by the chain.
            let input_keys = self.chain.input_keys();
            if let Some(first_key) = input_keys.first() {
                map.insert(
                    first_key.to_string(),
                    Value::String(message.content.clone()),
                );
            } else {
                map.insert("input".to_string(), Value::String(message.content.clone()));
            }
            map
        };

        // Invoke the chain.
        match self.chain.invoke(inputs).await {
            Ok(result) => {
                // Extract the output text from the chain result.
                let output = result
                    .values()
                    .next()
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let task_id = uuid::Uuid::new_v4().to_string();
                let task = A2ATask {
                    id: task_id.clone(),
                    message,
                    status: TaskStatus::Completed,
                };
                let task_result = A2ATaskResult::new(output);

                // Store the task in-memory.
                {
                    self.tasks.write().await.insert(
                        task_id,
                        StoredTask {
                            task: task.clone(),
                            result: Some(task_result.clone()),
                        },
                    );
                }
                self.evict_if_needed().await;

                A2AResponse::ok(
                    req.id,
                    json!({
                        "task": task,
                        "result": task_result,
                    }),
                )
            }
            Err(e) => {
                let task_id = uuid::Uuid::new_v4().to_string();
                let task = A2ATask {
                    id: task_id.clone(),
                    message,
                    status: TaskStatus::Failed,
                };

                // Store the failed task in-memory.
                {
                    self.tasks.write().await.insert(
                        task_id,
                        StoredTask {
                            task: task.clone(),
                            result: None,
                        },
                    );
                }
                self.evict_if_needed().await;

                A2AResponse::error(req.id, -32000, format!("Chain execution failed: {}", e))
            }
        }
    }

    /// Handle `tasks/get`: return a task by ID.
    async fn handle_tasks_get(&self, req: A2ARequest) -> A2AResponse {
        let task_id = req
            .params
            .as_ref()
            .and_then(|p| p.get("taskId"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if task_id.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Missing taskId parameter"),
            );
        }

        let tasks = self.tasks.read().await;
        match tasks.get(task_id) {
            Some(stored) => {
                let mut result = json!({ "task": stored.task });
                if let Some(ref task_result) = stored.result {
                    result["result"] = json!(task_result);
                }
                A2AResponse::ok(req.id, result)
            }
            None => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(-32001, format!("Task not found: {}", task_id)),
            ),
        }
    }

    /// Handle `tasks/cancel`: cancel a task by ID.
    async fn handle_tasks_cancel(&self, req: A2ARequest) -> A2AResponse {
        let task_id = req
            .params
            .as_ref()
            .and_then(|p| p.get("taskId"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if task_id.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Missing taskId parameter"),
            );
        }

        let mut tasks = self.tasks.write().await;
        match tasks.get_mut(task_id) {
            Some(stored) => {
                stored.task.status = TaskStatus::Cancelled;
                A2AResponse::ok(req.id, json!({ "task": stored.task }))
            }
            None => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(-32001, format!("Task not found: {}", task_id)),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_chains::base::{BaseChain, ChainError, ChainResult};

    /// A simple mock chain that echoes the input.
    struct EchoChain;

    #[async_trait::async_trait]
    impl BaseChain for EchoChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }

        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }

        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            let input = inputs.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let mut result = HashMap::new();
            result.insert("output".to_string(), Value::String(input.to_string()));
            Ok(result)
        }

        fn name(&self) -> &str {
            "echo-chain"
        }
    }

    /// A chain that always fails.
    struct FailChain;

    #[async_trait::async_trait]
    impl BaseChain for FailChain {
        fn input_keys(&self) -> Vec<&str> {
            vec!["input"]
        }

        fn output_keys(&self) -> Vec<&str> {
            vec!["output"]
        }

        async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
            Err(ChainError::ExecutionError(
                "intentional failure".to_string(),
            ))
        }

        fn name(&self) -> &str {
            "fail-chain"
        }
    }

    fn echo_server() -> A2AServer {
        A2AServer::new(Arc::new(EchoChain))
    }

    fn fail_server() -> A2AServer {
        A2AServer::new(Arc::new(FailChain))
    }

    #[test]
    fn get_agent_card_default() {
        let server = echo_server();
        let card = server.get_agent_card();
        assert_eq!(card.name, "echo-chain");
        assert!(card.description.contains("echo-chain"));
    }

    #[test]
    fn get_agent_card_custom() {
        let card = AgentCard::new("custom", "Custom agent", "http://example.com")
            .with_capability("text-generation");
        let server = echo_server().with_card(card);
        let card = server.get_agent_card();
        assert_eq!(card.name, "custom");
        assert_eq!(card.url, "http://example.com");
        assert_eq!(card.capabilities.len(), 1);
    }

    #[tokio::test]
    async fn handle_tasks_send_success() {
        let server = echo_server();
        let msg = A2AMessage::user("hello world");
        let req = A2ARequest::send_task(1, &msg);
        let resp = server.handle_a2a_request(req).await;
        assert!(!resp.is_error());

        let result = resp.result.unwrap();
        let task = result.get("task").unwrap();
        assert_eq!(task["status"], "completed");

        let task_result = result.get("result").unwrap();
        assert_eq!(task_result["output"], "hello world");
    }

    #[tokio::test]
    async fn handle_tasks_send_failure() {
        let server = fail_server();
        let msg = A2AMessage::user("hello");
        let req = A2ARequest::send_task(2, &msg);
        let resp = server.handle_a2a_request(req).await;
        // Chain failure now returns an error response
        assert!(resp.is_error());

        let err = resp.error.unwrap();
        assert!(err.message.contains("Chain execution failed"));
    }

    #[tokio::test]
    async fn handle_tasks_send_missing_params() {
        let server = echo_server();
        let req = A2ARequest::new(3, "tasks/send", None);
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32602);
    }

    #[tokio::test]
    async fn handle_tasks_get_missing_task_id() {
        let server = echo_server();
        let req = A2ARequest::new(4, "tasks/get", Some(json!({})));
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
    }

    #[tokio::test]
    async fn handle_tasks_get_not_found() {
        let server = echo_server();
        let req = A2ARequest::get_task(5, "nonexistent-task");
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert!(err.message.contains("Task not found"));
    }

    #[tokio::test]
    async fn handle_tasks_get_after_send() {
        let server = echo_server();
        let msg = A2AMessage::user("hello");
        let send_req = A2ARequest::send_task(10, &msg);
        let send_resp = server.handle_a2a_request(send_req).await;
        let result = send_resp.result.unwrap();
        let task_id = result["task"]["id"].as_str().unwrap().to_string();

        // Now retrieve the task via tasks/get.
        let get_req = A2ARequest::get_task(11, &task_id);
        let get_resp = server.handle_a2a_request(get_req).await;
        assert!(!get_resp.is_error());

        let get_result = get_resp.result.unwrap();
        let task = get_result.get("task").unwrap();
        assert_eq!(task["id"], task_id);
        assert_eq!(task["status"], "completed");
        assert!(get_result.get("result").is_some());
    }

    #[tokio::test]
    async fn handle_tasks_cancel_nonexistent() {
        let server = echo_server();
        let req = A2ARequest::cancel_task(6, "task-123");
        // Cancelling a non-existent task now returns an error
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert!(err.message.contains("Task not found"));
    }

    #[tokio::test]
    async fn handle_tasks_cancel_existing_task() {
        let server = echo_server();
        let msg = A2AMessage::user("hello");
        let send_req = A2ARequest::send_task(20, &msg);
        let send_resp = server.handle_a2a_request(send_req).await;
        let result = send_resp.result.unwrap();
        let task_id = result["task"]["id"].as_str().unwrap().to_string();

        // Cancel the task.
        let cancel_req = A2ARequest::cancel_task(21, &task_id);
        let cancel_resp = server.handle_a2a_request(cancel_req).await;
        assert!(!cancel_resp.is_error());

        let cancel_result = cancel_resp.result.unwrap();
        assert_eq!(cancel_result["task"]["status"], "cancelled");

        // Verify the task is cancelled when retrieved.
        let get_req = A2ARequest::get_task(22, &task_id);
        let get_resp = server.handle_a2a_request(get_req).await;
        assert!(!get_resp.is_error());
        let get_result = get_resp.result.unwrap();
        assert_eq!(get_result["task"]["status"], "cancelled");
    }

    #[tokio::test]
    async fn handle_tasks_cancel_missing_task_id() {
        let server = echo_server();
        let req = A2ARequest::new(7, "tasks/cancel", Some(json!({})));
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
    }

    #[tokio::test]
    async fn handle_unknown_method() {
        let server = echo_server();
        let req = A2ARequest::new(8, "foo/bar", None);
        let resp = server.handle_a2a_request(req).await;
        assert!(resp.is_error());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
    }

    #[tokio::test]
    async fn handle_tasks_send_with_raw_params() {
        // When params has no "message" key, the entire params become the content.
        let server = echo_server();
        let req = A2ARequest::new(9, "tasks/send", Some(json!({"query": "test query"})));
        let resp = server.handle_a2a_request(req).await;
        assert!(!resp.is_error());
    }

    #[tokio::test]
    async fn handle_tasks_send_chain_with_no_input_keys() {
        /// A chain with no input keys.
        struct NoKeyChain;

        #[async_trait::async_trait]
        impl BaseChain for NoKeyChain {
            fn input_keys(&self) -> Vec<&str> {
                vec![]
            }

            fn output_keys(&self) -> Vec<&str> {
                vec!["output"]
            }

            async fn invoke(
                &self,
                inputs: HashMap<String, Value>,
            ) -> Result<ChainResult, ChainError> {
                let input = inputs
                    .get("input")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default");
                let mut result = HashMap::new();
                result.insert("output".to_string(), Value::String(input.to_string()));
                Ok(result)
            }

            fn name(&self) -> &str {
                "no-key-chain"
            }
        }

        let server = A2AServer::new(Arc::new(NoKeyChain));
        let msg = A2AMessage::user("hello");
        let req = A2ARequest::send_task(10, &msg);
        let resp = server.handle_a2a_request(req).await;
        assert!(!resp.is_error());
    }
}
