use super::*;
use lc_agents::{AgentError, AgentFinish, AgentOutput, AgentStep, BaseAgent};
use lc_chains::base::{BaseChain, ChainError, ChainResult};
use tokio::sync::Notify;

use crate::protocol::WorkflowStep;

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

/// A chain that signals when it starts and blocks until released.
/// Lets tests observe the `working` state and cancel mid-flight.
struct BlockingChain {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait::async_trait]
impl BaseChain for BlockingChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }

    async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        self.started.notify_one();
        self.release.notified().await;
        let mut result = HashMap::new();
        result.insert("output".to_string(), Value::String("done".to_string()));
        Ok(result)
    }

    fn name(&self) -> &str {
        "blocking-chain"
    }
}

/// A chain that asks for more input until it sees "alice" (P2-3).
struct InputRequiredChain;

#[async_trait::async_trait]
impl BaseChain for InputRequiredChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }

    async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let input = inputs.get("input").and_then(|v| v.as_str()).unwrap_or("");
        if !input.contains("alice") {
            return Err(ChainError::MissingInput(
                "please provide your name".to_string(),
            ));
        }
        let mut result = HashMap::new();
        result.insert("output".to_string(), Value::String(input.to_string()));
        Ok(result)
    }

    fn name(&self) -> &str {
        "input-required-chain"
    }
}

/// A chain whose output is a fixed label (for skill-routing observability).
struct NamedChain(String);

#[async_trait::async_trait]
impl BaseChain for NamedChain {
    fn input_keys(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn output_keys(&self) -> Vec<&str> {
        vec!["output"]
    }

    async fn invoke(&self, _inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
        let mut out = HashMap::new();
        out.insert("output".to_string(), Value::String(self.0.clone()));
        Ok(out)
    }

    fn name(&self) -> &str {
        &self.0
    }
}

fn echo_server() -> A2AServer {
    A2AServer::new(Arc::new(EchoChain))
}

fn fail_server() -> A2AServer {
    A2AServer::new(Arc::new(FailChain))
}

/// A planner that echoes the input back — drives an `AgentExecutor` for the
/// `from_agent` end-to-end path.
struct EchoAgent;

#[async_trait::async_trait]
impl BaseAgent for EchoAgent {
    async fn plan(
        &self,
        _intermediate_steps: &[AgentStep],
        inputs: &HashMap<String, String>,
    ) -> Result<AgentOutput, AgentError> {
        let input = inputs.get("input").cloned().unwrap_or_default();
        Ok(AgentOutput::Finish(AgentFinish::new(
            format!("agent-said: {}", input),
            String::new(),
        )))
    }
}

/// Poll `tasks/get` until the task reaches `want`, then return the response.
async fn wait_for_status(server: &A2AServer, task_id: &str, want: &str) -> A2AResponse {
    for _ in 0..200 {
        let resp = server
            .handle_a2a_request(A2ARequest::get_task(99, task_id))
            .await;
        if let Some(r) = &resp.result {
            if r.get("task")
                .and_then(|t| t.get("status"))
                .and_then(|s| s.as_str())
                == Some(want)
            {
                return resp;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("task {task_id} did not reach status {want} in time");
}

/// Send a task and return the created task id.
async fn send_task_id(server: &A2AServer, content: &str) -> String {
    let resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user(content)))
        .await;
    assert!(!resp.is_error());
    resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn get_agent_card_default() {
    let server = echo_server();
    let card = server.get_agent_card();
    assert_eq!(card.name, "echo-chain");
    assert!(card.description.contains("echo-chain"));
    assert_eq!(card.protocol_version, "0.3.0");
    assert_eq!(card.skills.len(), 1);
    assert_eq!(card.skills[0].id, "default");
    assert_eq!(card.skills[0].name, "echo-chain");
}

#[test]
fn get_agent_card_custom() {
    let card = AgentCard::new("custom", "Custom agent", "http://example.com")
        .with_skill(AgentSkill::new("s1", "text-generation", "Generates text"));
    let server = echo_server().with_card(card);
    let card = server.get_agent_card();
    assert_eq!(card.name, "custom");
    assert_eq!(card.url, "http://example.com");
    assert_eq!(card.skills.len(), 1);
    assert_eq!(card.skills[0].id, "s1");
}

#[tokio::test]
async fn handle_tasks_send_returns_submitted_immediately() {
    let server = echo_server();
    let msg = A2AMessage::user("hello world");
    let req = A2ARequest::send_task(1, &msg);
    let resp = server.handle_a2a_request(req).await;
    assert!(!resp.is_error());

    // The request acknowledges immediately with a `submitted` task.
    let result = resp.result.unwrap();
    let task = result.get("task").unwrap();
    assert_eq!(task["status"], "submitted");
    // No result yet.
    assert!(result.get("result").is_none());
}

#[tokio::test]
async fn from_agent_serves_tasks_end_to_end() {
    // P1-8: an A2AServer backed by an AgentExecutor (adapted to BaseChain).
    let executor = Arc::new(AgentExecutor::new(Arc::new(EchoAgent), Vec::new()));
    let server = A2AServer::from_agent(executor);

    let task_id = send_task_id(&server, "hi").await;
    let done = wait_for_status(&server, &task_id, "completed").await;
    let output = done.result.unwrap()["result"]["output"]
        .as_str()
        .unwrap()
        .to_string();
    // The agent (not a plain chain) actually ran and produced its answer.
    assert_eq!(output, "agent-said: hi");
}

#[tokio::test]
async fn handle_tasks_get_shows_working_then_completed() {
    let server = echo_server();
    let msg = A2AMessage::user("hello");
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &msg))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let done = wait_for_status(&server, &task_id, "completed").await;
    let result = done.result.unwrap();
    let task = result.get("task").unwrap();
    assert_eq!(task["id"], task_id);
    assert_eq!(task["status"], "completed");
    let task_result = result.get("result").unwrap();
    assert_eq!(task_result["output"], "hello");
}

#[tokio::test]
async fn handle_tasks_send_failure() {
    let server = fail_server();
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(2, &A2AMessage::user("hello")))
        .await;
    // Submission itself succeeds; the failure is recorded on the task.
    assert!(!send_resp.is_error());

    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let done = wait_for_status(&server, &task_id, "failed").await;
    let result = done.result.unwrap();
    assert_eq!(result["task"]["status"], "failed");
    let error = result.get("error").unwrap();
    assert!(error.as_str().unwrap().contains("intentional failure"));
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
async fn handle_tasks_get_not_found() {
    let server = echo_server();
    let req = A2ARequest::get_task(5, "nonexistent-task");
    let resp = server.handle_a2a_request(req).await;
    assert!(resp.is_error());
    let err = resp.error.unwrap();
    assert!(err.message.contains("Task not found"));
}

#[tokio::test]
async fn handle_tasks_cancel_nonexistent() {
    let server = echo_server();
    let req = A2ARequest::cancel_task(6, "task-123");
    let resp = server.handle_a2a_request(req).await;
    assert!(resp.is_error());
    let err = resp.error.unwrap();
    assert!(err.message.contains("Task not found"));
}

#[tokio::test]
async fn handle_tasks_cancel_working_task() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let chain = Arc::new(BlockingChain {
        started: started.clone(),
        release: release.clone(),
    });
    let server = A2AServer::new(chain);

    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Wait for the background chain to start -> task is `working`.
    started.notified().await;
    let get_resp = server
        .handle_a2a_request(A2ARequest::get_task(2, &task_id))
        .await;
    assert_eq!(get_resp.result.unwrap()["task"]["status"], "working");

    // Cancel it mid-flight.
    let cancel_resp = server
        .handle_a2a_request(A2ARequest::cancel_task(3, &task_id))
        .await;
    assert!(!cancel_resp.is_error());
    assert_eq!(cancel_resp.result.unwrap()["task"]["status"], "cancelled");

    // Release the chain; the background worker must NOT clobber the
    // cancelled status back to completed.
    release.notify_one();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let get_resp = server
        .handle_a2a_request(A2ARequest::get_task(4, &task_id))
        .await;
    assert_eq!(get_resp.result.unwrap()["task"]["status"], "cancelled");
}

#[tokio::test]
async fn handle_tasks_cancel_completed_is_idempotent() {
    let server = echo_server();
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    wait_for_status(&server, &task_id, "completed").await;

    let cancel_resp = server
        .handle_a2a_request(A2ARequest::cancel_task(2, &task_id))
        .await;
    assert!(!cancel_resp.is_error());
    // Already terminal: returned unchanged.
    assert_eq!(cancel_resp.result.unwrap()["task"]["status"], "completed");
}

#[tokio::test]
async fn handle_tasks_cancel_missing_task_id() {
    let server = echo_server();
    let req = A2ARequest::new(7, "tasks/cancel", Some(json!({})));
    let resp = server.handle_a2a_request(req).await;
    assert!(resp.is_error());
}

#[tokio::test]
async fn handle_tasks_list_returns_tasks() {
    let server = echo_server();
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = server
        .handle_a2a_request(A2ARequest::new(2, "tasks/list", None))
        .await;
    assert!(!resp.is_error());
    let result = resp.result.unwrap();
    let tasks = result["tasks"].as_array().unwrap();
    assert!(
        tasks
            .iter()
            .any(|t| t["id"].as_str() == Some(task_id.as_str())),
        "expected task {task_id} in list"
    );
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

        async fn invoke(&self, inputs: HashMap<String, Value>) -> Result<ChainResult, ChainError> {
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

#[tokio::test]
async fn handle_a2a_request_authenticated_requires_token() {
    let server = echo_server().with_auth_token("secret");
    assert_eq!(
        server.get_agent_card().authentication,
        Some(vec!["bearer".to_string()])
    );

    let msg = A2AMessage::user("hi");
    let resp = server
        .handle_a2a_request_authenticated(A2ARequest::send_task(1, &msg), None)
        .await;
    assert!(resp.is_error());
    assert_eq!(resp.error.unwrap().code, 401);
}

#[tokio::test]
async fn handle_a2a_request_authenticated_invalid_token() {
    let server = echo_server().with_auth_token("secret");
    let msg = A2AMessage::user("hi");
    let resp = server
        .handle_a2a_request_authenticated(A2ARequest::send_task(1, &msg), Some("wrong"))
        .await;
    assert!(resp.is_error());
    assert_eq!(resp.error.unwrap().code, 401);
}

#[tokio::test]
async fn handle_a2a_request_authenticated_valid_token() {
    let server = echo_server().with_auth_token("secret");
    let msg = A2AMessage::user("hi");
    let resp = server
        .handle_a2a_request_authenticated(A2ARequest::send_task(1, &msg), Some("secret"))
        .await;
    assert!(!resp.is_error());
}

#[tokio::test]
async fn handle_a2a_request_unauthenticated_passes() {
    // Without an auth token configured, requests pass straight through.
    let server = echo_server();
    let msg = A2AMessage::user("hi");
    let resp = server
        .handle_a2a_request_authenticated(A2ARequest::send_task(1, &msg), None)
        .await;
    assert!(!resp.is_error());
}

#[tokio::test]
async fn handle_a2a_request_rate_limited() {
    let server = echo_server().with_rate_limiter(Arc::new(RateLimiter::new(0, 1)));
    let msg = A2AMessage::user("hi");
    let r1 = server
        .handle_a2a_request(A2ARequest::send_task(1, &msg))
        .await;
    assert!(!r1.is_error());

    let r2 = server
        .handle_a2a_request(A2ARequest::send_task(2, &msg))
        .await;
    assert!(r2.is_error());
    assert_eq!(r2.error.unwrap().code, 429);
}

// ---- P1-6: idempotent tasks/send ----

#[tokio::test]
async fn handle_tasks_send_idempotent_message_id() {
    let server = echo_server();
    let msg = A2AMessage::user("hello");
    let req = A2ARequest::send_task_with_message_id(1, &msg, "idem-1");
    let r1 = server.handle_a2a_request(req.clone()).await;
    assert!(!r1.is_error());
    let task1 = r1.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let r2 = server.handle_a2a_request(req).await;
    assert!(!r2.is_error());
    let task2 = r2.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Same message_id -> same task returned, not a second run.
    assert_eq!(task1, task2);
}

// ---- P1-4: ownership ----

#[tokio::test]
async fn handle_tasks_get_owner_enforced() {
    let server = echo_server();
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Same owner -> allowed.
    let ok = server
        .handle_a2a_request(A2ARequest::get_task(2, &task_id).with_owner("alice"))
        .await;
    assert!(!ok.is_error());

    // Different owner -> 403.
    let denied = server
        .handle_a2a_request(A2ARequest::get_task(3, &task_id).with_owner("bob"))
        .await;
    assert!(denied.is_error());
    assert_eq!(denied.error.unwrap().code, -32003);

    // No owner identity -> 403 (task is protected).
    let anon = server
        .handle_a2a_request(A2ARequest::get_task(4, &task_id))
        .await;
    assert!(anon.is_error());
    assert_eq!(anon.error.unwrap().code, -32003);
}

#[tokio::test]
async fn handle_tasks_cancel_owner_enforced() {
    let server = echo_server();
    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let denied = server
        .handle_a2a_request(A2ARequest::cancel_task(2, &task_id).with_owner("bob"))
        .await;
    assert!(denied.is_error());
    assert_eq!(denied.error.unwrap().code, -32003);

    let ok = server
        .handle_a2a_request(A2ARequest::cancel_task(3, &task_id).with_owner("alice"))
        .await;
    assert!(!ok.is_error());
    assert_eq!(ok.result.unwrap()["task"]["status"], "cancelled");
}

#[tokio::test]
async fn handle_tasks_list_filters_by_owner() {
    let server = echo_server();
    let _a = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"))
        .await;
    let _b = server
        .handle_a2a_request(A2ARequest::send_task(2, &A2AMessage::user("hi")).with_owner("bob"))
        .await;

    // Alice's identity lists only her tasks by default.
    let resp = server
        .handle_a2a_request(A2ARequest::new(3, "tasks/list", None).with_owner("alice"))
        .await;
    let tasks = resp.result.unwrap()["tasks"].as_array().unwrap().clone();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0]["owner"], "alice");

    // Anonymous listing sees only unowned tasks (ownership enforced per task).
    let _c = server
        .handle_a2a_request(A2ARequest::send_task(5, &A2AMessage::user("hi")))
        .await;
    let resp = server
        .handle_a2a_request(A2ARequest::new(6, "tasks/list", None))
        .await;
    let tasks = resp.result.unwrap()["tasks"].as_array().unwrap().clone();
    assert_eq!(tasks.len(), 1);
    assert!(tasks[0].get("owner").is_none() || tasks[0]["owner"].is_null());
}

// ---- P2-2/P2-3: multi-turn continuation ----

#[tokio::test]
async fn handle_tasks_send_continue_terminal_rejected() {
    let server = echo_server();
    let task_id = send_task_id(&server, "hello").await;
    wait_for_status(&server, &task_id, "completed").await;

    // Second continue after completion is rejected.
    let continue_resp = server
        .handle_a2a_request(A2ARequest::continue_task(
            2,
            &task_id,
            &A2AMessage::user("x"),
        ))
        .await;
    assert!(continue_resp.is_error());
    assert_eq!(continue_resp.error.unwrap().code, -32004);
}

// ---- P2-3: input-required flow ----

#[tokio::test]
async fn handle_tasks_send_input_required_then_resume() {
    let server = A2AServer::new(Arc::new(InputRequiredChain));
    let task_id = send_task_id(&server, "hello").await;

    // The chain asks for more input -> task goes input-required.
    let pending = wait_for_status(&server, &task_id, "input-required").await;
    let error = pending.result.unwrap()["error"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(error.contains("please provide your name"));

    // Resume with the missing information via taskId.
    let resume_resp = server
        .handle_a2a_request(A2ARequest::continue_task(
            2,
            &task_id,
            &A2AMessage::user("my name is alice"),
        ))
        .await;
    assert!(!resume_resp.is_error());

    let done = wait_for_status(&server, &task_id, "completed").await;
    let output = done.result.unwrap()["result"]["output"]
        .as_str()
        .unwrap()
        .to_string();
    // The resumed run sees the full conversation: original turn + answer.
    assert!(
        output.contains("hello"),
        "resumed output missing first turn: {output}"
    );
    assert!(
        output.contains("alice"),
        "resumed output missing answer: {output}"
    );
}

#[tokio::test]
async fn handle_tasks_send_continue_working_rejected() {
    // Only input-required tasks can be resumed. Continuing a task that is
    // still `working` would spawn a second worker racing the first.
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let chain = Arc::new(BlockingChain {
        started: started.clone(),
        release: release.clone(),
    });
    let server = A2AServer::new(chain);

    let send_resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")))
        .await;
    let task_id = send_resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Wait until the chain is running -> the task is `working`.
    started.notified().await;

    let continue_resp = server
        .handle_a2a_request(A2ARequest::continue_task(
            2,
            &task_id,
            &A2AMessage::user("more"),
        ))
        .await;
    assert!(continue_resp.is_error());
    assert_eq!(continue_resp.error.unwrap().code, -32004);

    // Let the first worker finish so it does not outlive the test.
    release.notify_one();
}

// ---- P1-1: custom store ----

#[tokio::test]
async fn with_store_custom_backend() {
    let store = InMemoryTaskStore::with_max_tasks(1);
    let server = echo_server().with_store(Arc::new(store));

    let first = send_task_id(&server, "one").await;
    let second = send_task_id(&server, "two").await;

    // Capacity 1: the first task was evicted, the second survives.
    let gone = server
        .handle_a2a_request(A2ARequest::get_task(2, &first))
        .await;
    assert!(gone.is_error());
    assert!(gone.error.unwrap().message.contains("Task not found"));

    let present = server
        .handle_a2a_request(A2ARequest::get_task(3, &second))
        .await;
    assert!(!present.is_error());
}

// ---- P2-4: skill routing ----

#[tokio::test]
async fn handle_tasks_send_routes_by_skill() {
    let router =
        SkillMapRouter::new().with_skill("math", Arc::new(NamedChain("math-chain".to_string())));
    let server = echo_server().with_skill_map(router);

    // A request with skillId=math is handled by the routed chain.
    let params = json!({
        "message": { "role": "user", "content": "hi" },
        "skillId": "math"
    });
    let resp = server
        .handle_a2a_request(A2ARequest::new(1, "tasks/send", Some(params)))
        .await;
    let task_id = resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let done = wait_for_status(&server, &task_id, "completed").await;
    let output = done.result.unwrap()["result"]["output"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(output, "math-chain");

    // Without a skillId the default echo chain handles it.
    let task_id = send_task_id(&server, "hello").await;
    let done = wait_for_status(&server, &task_id, "completed").await;
    let output = done.result.unwrap()["result"]["output"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(output, "hello");
}

#[tokio::test]
async fn handle_tasks_send_unknown_skill_falls_back() {
    let router =
        SkillMapRouter::new().with_skill("math", Arc::new(NamedChain("math-chain".to_string())));
    let server = echo_server().with_skill_map(router);

    let params = json!({
        "message": { "role": "user", "content": "hi" },
        "skillId": "unknown"
    });
    let resp = server
        .handle_a2a_request(A2ARequest::new(1, "tasks/send", Some(params)))
        .await;
    let task_id = resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let done = wait_for_status(&server, &task_id, "completed").await;
    let output = done.result.unwrap()["result"]["output"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(output, "hi");
}

// ---- P2-1: streaming events ----

#[tokio::test]
async fn with_streaming_publishes_events_and_advertises_sse() {
    let server = echo_server().with_streaming(16);
    let card = server.get_agent_card();
    assert_eq!(
        card.interfaces,
        Some(json!({ "sse": true })),
        "streaming advertises sse interface"
    );

    let mut rx = server.subscribe().expect("subscribed");
    let task_id = send_task_id(&server, "hello").await;

    // Collect events until the terminal Completed status arrives.
    let mut saw_working = false;
    let mut saw_completed = false;
    let mut saw_artifact = false;
    for _ in 0..8 {
        match rx.recv().await {
            Ok(event) => {
                assert_eq!(event.id(), task_id);
                match event.status_value() {
                    Some(TaskStatus::Working) => saw_working = true,
                    Some(TaskStatus::Completed) => saw_completed = true,
                    // ArtifactUpdate carries no status.
                    None => saw_artifact = true,
                    _ => {}
                }
                if saw_completed && saw_artifact {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    assert!(saw_working, "expected a working event");
    assert!(saw_completed, "expected a completed event");
    assert!(saw_artifact, "expected an artifact event");
}

#[tokio::test]
async fn without_streaming_no_subscriber() {
    let server = echo_server();
    assert!(server.subscribe().is_none());
}

#[tokio::test]
async fn sweep_expired_tasks_cleans_terminal_and_expires_live() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));

    // Terminal task past TTL -> deleted.
    let mut terminal = StoredTask::new(
        A2ATask::new("t-term", A2AMessage::user("done")).with_status(TaskStatus::Completed),
    );
    terminal.updated_at = std::time::Instant::now() - Duration::from_secs(100);

    // Live task past TTL -> marked expired.
    let mut live = StoredTask::new(
        A2ATask::new("t-live", A2AMessage::user("run")).with_status(TaskStatus::Working),
    );
    live.updated_at = std::time::Instant::now() - Duration::from_secs(100);

    // Fresh task -> untouched.
    let fresh = StoredTask::new(A2ATask::new("t-fresh", A2AMessage::user("new")));

    store.upsert(terminal).await.unwrap();
    store.upsert(live).await.unwrap();
    store.upsert(fresh).await.unwrap();

    sweep_expired_tasks(&store, Duration::from_secs(10)).await;

    assert!(
        store.get("t-term").await.unwrap().is_none(),
        "terminal task past TTL is deleted"
    );
    let live = store.get("t-live").await.unwrap().expect("live task kept");
    assert_eq!(live.task.status, TaskStatus::Expired);
    let fresh = store
        .get("t-fresh")
        .await
        .unwrap()
        .expect("fresh task kept");
    assert_eq!(fresh.task.status, TaskStatus::Submitted);
}

#[tokio::test]
async fn background_cleanup_sweeps_periodically() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let mut expired = StoredTask::new(
        A2ATask::new("t-old", A2AMessage::user("hi")).with_status(TaskStatus::Working),
    );
    expired.updated_at = std::time::Instant::now() - Duration::from_secs(100);
    store.upsert(expired).await.unwrap();

    // The background sweeper ticks every 5ms and expires t-old without any
    // read-path trigger.
    let _server = A2AServer::new(Arc::new(EchoChain))
        .with_store(store.clone())
        .with_task_ttl(Some(Duration::from_secs(10)))
        .with_background_cleanup(Duration::from_millis(5));

    tokio::time::sleep(Duration::from_millis(50)).await;

    let t = store
        .get("t-old")
        .await
        .unwrap()
        .expect("live task still present");
    assert_eq!(t.task.status, TaskStatus::Expired);
}

#[tokio::test]
async fn task_stores_request_trace_id() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain)).with_store(store.clone());

    let req = A2ARequest::send_task(1, &A2AMessage::user("hello")).with_trace_id("trace-abc");
    let resp = server.handle_a2a_request(req).await;
    assert!(!resp.is_error());
    let task_id = resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let stored = store.get(&task_id).await.unwrap().expect("task stored");
    assert_eq!(stored.trace_id.as_deref(), Some("trace-abc"));
}

#[tokio::test]
async fn task_without_trace_id_has_none() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain)).with_store(store.clone());

    let resp = server
        .handle_a2a_request(A2ARequest::send_task(1, &A2AMessage::user("hi")))
        .await;
    assert!(!resp.is_error());
    let task_id = resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let stored = store.get(&task_id).await.unwrap().expect("task stored");
    assert!(stored.trace_id.is_none());
}

#[tokio::test]
async fn run_workflow_executes_steps_in_order_and_aggregates() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain)).with_store(store.clone());

    let workflow = A2AWorkflow::new(vec![
        WorkflowStep::new("s1", "first"),
        WorkflowStep::new("s2", "second"),
    ]);
    let resp = server
        .handle_a2a_request(A2ARequest::run_workflow(1, &workflow))
        .await;
    assert!(!resp.is_error());
    let result = resp.result.unwrap();
    assert_eq!(result["task"]["status"], "completed");
    assert_eq!(result["results"]["s1"], "first");
    assert_eq!(result["results"]["s2"], "second");

    // The backing task was persisted with the aggregated output.
    let task_id = result["task"]["id"].as_str().unwrap();
    let stored = store.get(task_id).await.unwrap().expect("task stored");
    assert_eq!(stored.task.status, TaskStatus::Completed);
    assert_eq!(stored.result.as_ref().unwrap().output, "first\nsecond");
}

#[tokio::test]
async fn run_workflow_respects_supplied_workflow_id_and_owner() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain)).with_store(store.clone());

    let workflow = A2AWorkflow::new(vec![WorkflowStep::new("s1", "hi")])
        .with_workflow_id("wf-42")
        .with_name("my workflow");
    let resp = server
        .handle_a2a_request(A2ARequest::run_workflow(1, &workflow).with_owner("alice"))
        .await;
    assert!(!resp.is_error());
    let result = resp.result.unwrap();
    assert_eq!(result["task"]["id"], "wf-42");

    let stored = store.get("wf-42").await.unwrap().expect("task stored");
    assert_eq!(stored.task.owner.as_deref(), Some("alice"));
}

#[tokio::test]
async fn run_workflow_routes_steps_by_skill() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain))
        .with_store(store.clone())
        .with_skill_map(
            SkillMapRouter::new()
                .with_skill("translate", Arc::new(NamedChain("translated".to_string()))),
        );

    let workflow = A2AWorkflow::new(vec![
        WorkflowStep::new("s1", "hello"),
        WorkflowStep::with_skill("s2", "bonjour", "translate"),
    ]);
    let resp = server
        .handle_a2a_request(A2ARequest::run_workflow(1, &workflow))
        .await;
    assert!(!resp.is_error());
    let result = resp.result.unwrap();
    assert_eq!(result["results"]["s1"], "hello");
    assert_eq!(result["results"]["s2"], "translated");
}

#[tokio::test]
async fn run_workflow_step_failure_marks_task_failed_and_stops() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let failing = A2AServer::new(Arc::new(EchoChain))
        .with_store(store.clone())
        .with_skill_map(SkillMapRouter::new().with_skill("failing", Arc::new(FailChain)));

    let workflow = A2AWorkflow::new(vec![
        WorkflowStep::new("s1", "ok"),
        WorkflowStep::with_skill("s2", "boom", "failing"),
    ]);
    let resp = failing
        .handle_a2a_request(A2ARequest::run_workflow(1, &workflow))
        .await;
    assert!(!resp.is_error()); // the task itself records the failure
    let result = resp.result.unwrap();
    assert_eq!(result["task"]["status"], "failed");
    assert!(result["results"].get("s1").is_some());
    assert!(result["results"].get("s2").is_none());
    assert!(result["error"]
        .as_str()
        .unwrap()
        .contains("step `s2` failed"));

    let stored = store
        .get(result["task"]["id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.task.status, TaskStatus::Failed);
    assert!(stored.error.as_deref().unwrap().contains("s2"));
}

#[tokio::test]
async fn run_workflow_missing_params_invalid() {
    let server = A2AServer::new(Arc::new(EchoChain));
    let resp = server
        .handle_a2a_request(A2ARequest::new(1, "tasks/runWorkflow", None))
        .await;
    assert!(resp.is_error());
}

#[tokio::test]
async fn run_workflow_empty_steps_invalid() {
    let server = A2AServer::new(Arc::new(EchoChain));
    let resp = server
        .handle_a2a_request(A2ARequest::run_workflow(1, &A2AWorkflow::new(vec![])))
        .await;
    assert!(resp.is_error());
}

#[tokio::test]
async fn run_workflow_carries_trace_id_onto_backing_task() {
    let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::with_max_tasks(10));
    let server = A2AServer::new(Arc::new(EchoChain)).with_store(store.clone());

    let workflow = A2AWorkflow::new(vec![WorkflowStep::new("s1", "hi")]);
    let resp = server
        .handle_a2a_request(A2ARequest::run_workflow(1, &workflow).with_trace_id("trace-wf"))
        .await;
    assert!(!resp.is_error());
    let task_id = resp.result.unwrap()["task"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    let stored = store.get(&task_id).await.unwrap().unwrap();
    assert_eq!(stored.trace_id.as_deref(), Some("trace-wf"));
}
