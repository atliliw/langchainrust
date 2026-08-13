//! A2A Server - handler functions for the Agent-to-Agent protocol.
//!
//! Provides `A2AServer` which holds an underlying agent (a `BaseChain`) and
//! exposes handler functions that can be plugged into any HTTP framework
//! (axum, actix, warp, etc.) rather than running its own server.
//!
//! # Endpoints
//!
//! - `GET /.well-known/agent-card.json` -> returns `AgentCard` (via `get_agent_card`)
//! - `POST /` -> accepts `A2ARequest`, dispatches, returns `A2AResponse`
//!   (via `handle_a2a_request` / `handle_a2a_request_authenticated`)
//!
//! # Task Model
//!
//! `tasks/send` follows the A2A asynchronous task lifecycle. The request is
//! acknowledged immediately with a `submitted` task and the chain runs in the
//! background, transitioning the task `submitted -> working -> completed`
//! (or `failed`). Poll `tasks/get` to observe progress. Every transition is
//! guarded by the [`TaskStatus`] state machine, so a task cancelled while the
//! chain is still running is never clobbered back to a live state.
//!
//! # Multi-turn & Input-Required (P2-2/P2-3)
//!
//! Re-sending `tasks/send` with a `taskId` appends a message to the existing
//! task's history and re-runs the chain over the whole conversation. A chain
//! that needs more information returns a `ChainError::MissingInput` /
//! `ChainError::InputError`, which the server maps to the `input-required`
//! state; the client then resumes with `tasks/send {taskId, message}`.
//!
//! # Ownership & Idempotency (P1-4/P1-6)
//!
//! Tasks carry an optional `owner` taken from request metadata. `tasks/get`
//! and `tasks/cancel` from a caller whose metadata `owner` does not match the
//! task's are rejected (`-32003`). A `message_id` in request metadata makes
//! `tasks/send` idempotent: re-sending the same id returns the already
//! created task instead of running the chain twice.
//!
//! # Task Persistence (P1-1)
//!
//! Tasks are stored through the [`TaskStore`] trait, defaulting to an
//! in-memory [`InMemoryTaskStore`] shared with background workers. Swap in
//! your own backend with [`A2AServer::with_store`]. Terminal tasks older than
//! the configured TTL are cleaned up lazily on read access.
//!
//! # Streaming (P2-1)
//!
//! Enable [`A2AServer::with_streaming`] to get a `broadcast` channel of
//! [`TaskPushNotification`]s (`subscribe()`), which an HTTP layer can expose
//! as an SSE endpoint. The agent card then advertises `{"sse": true}`.
//!
//! # Skill routing (P2-4)
//!
//! [`A2AServer::with_skill_router`] dispatches `tasks/send` requests that
//! carry a `skillId` param to a different chain based on the card's skills.
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
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{broadcast, RwLock};

use lc_agents::AgentExecutor;
use lc_chains::base::{BaseChain, ChainError, ChainResult};

use super::agent_adapter::AgentExecutorChain;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskResult, A2AWorkflow,
    AgentCard, AgentSkill, TaskFilter, TaskPushNotification, TaskStatus,
};
use super::rate_limiter::RateLimiter;
use super::router::{SkillMapRouter, SkillRouter};
use super::store::{InMemoryTaskStore, StoredTask, TaskStore, DEFAULT_MAX_TASKS};

/// Default task time-to-live before expiry cleanup (24 hours).
const DEFAULT_TASK_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// A2A Server - wraps an agent and provides handler functions.
///
/// The server does NOT start its own HTTP listener. Instead, it provides
/// `handle_a2a_request()` and `get_agent_card()` that you can call from
/// any HTTP framework's route handler.
///
/// Tasks are stored through the [`TaskStore`] trait so that `tasks/get` can
/// retrieve them and `tasks/cancel` can transition their status. When the
/// default in-memory store exceeds its capacity, the least recently updated
/// task is evicted (LRU).
pub struct A2AServer {
    /// The underlying chain/agent.
    chain: Arc<dyn BaseChain>,
    /// The agent card metadata.
    card: AgentCard,
    /// Task persistence backend (P1-1).
    store: Arc<dyn TaskStore>,
    /// `message_id -> task_id` map for idempotent `tasks/send` (P1-6).
    message_ids: Arc<RwLock<HashMap<String, String>>>,
    /// Optional skill -> chain router (P2-4).
    skill_router: Option<Arc<dyn SkillRouter>>,
    /// Optional SSE event bus (P2-1).
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    /// Expected bearer token for authenticated requests (None = auth disabled).
    expected_token: Option<String>,
    /// Optional rate limiter applied to every request.
    rate_limiter: Option<Arc<RateLimiter>>,
    /// Time-to-live for tasks before they expire.
    task_ttl: Option<Duration>,
}

impl A2AServer {
    /// Create a new A2A server backed by a `BaseChain`.
    pub fn new(chain: Arc<dyn BaseChain>) -> Self {
        let card = AgentCard::new(
            chain.name(),
            format!("Agent backed by {}", chain.name()),
            "http://localhost:8080",
        )
        .with_skill(AgentSkill::new(
            "default",
            chain.name(),
            format!("Agent backed by {}", chain.name()),
        ));
        Self {
            chain,
            card,
            store: Arc::new(InMemoryTaskStore::with_max_tasks(DEFAULT_MAX_TASKS)),
            message_ids: Arc::new(RwLock::new(HashMap::new())),
            skill_router: None,
            event_bus: None,
            expected_token: None,
            rate_limiter: None,
            task_ttl: Some(DEFAULT_TASK_TTL),
        }
    }

    /// Create a server backed directly by a stateful agent (P1-8).
    ///
    /// The [`AgentExecutor`] is adapted to the chain interface, so A2A tasks
    /// get genuine conversational continuity. Attach memory to the executor
    /// (`.with_memory(...)`) before wrapping for multi-turn state.
    pub fn from_agent(executor: Arc<AgentExecutor>) -> Self {
        Self::new(Arc::new(AgentExecutorChain::new(executor)))
    }

    /// Replace the default in-memory task store with a custom backend (P1-1).
    pub fn with_store(mut self, store: Arc<dyn TaskStore>) -> Self {
        self.store = store;
        self
    }

    /// Set the maximum number of tasks before LRU eviction.
    ///
    /// Replaces the store with a fresh in-memory store of the given capacity,
    /// discarding any tasks stored so far. Call this before sending tasks.
    pub fn with_max_tasks(mut self, max: usize) -> Self {
        self.store = Arc::new(InMemoryTaskStore::with_max_tasks(max.max(1)));
        self
    }

    /// Attach a skill router so `tasks/send` requests with a `skillId` are
    /// dispatched to a different chain (P2-4).
    pub fn with_skill_router(mut self, router: Arc<dyn SkillRouter>) -> Self {
        self.skill_router = Some(router);
        self
    }

    /// Attach a default skill router built from a static `skill_id -> chain`
    /// map (P2-4).
    pub fn with_skill_map(mut self, map: SkillMapRouter) -> Self {
        self.skill_router = Some(Arc::new(map));
        self
    }

    /// Enable streaming push notifications over an SSE-compatible channel
    /// (P2-1).
    ///
    /// Creates a `broadcast` channel with the given capacity and advertises
    /// `{"sse": true}` on the agent card. Subscribe with
    /// [`A2AServer::subscribe`].
    pub fn with_streaming(mut self, capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(1));
        self.event_bus = Some(Arc::new(tx));
        self.card = self.card.clone().with_interfaces(json!({ "sse": true }));
        self
    }

    /// Subscribe to task push notifications, if streaming is enabled (P2-1).
    ///
    /// Returns `None` when the server was not built with
    /// [`A2AServer::with_streaming`].
    pub fn subscribe(&self) -> Option<broadcast::Receiver<TaskPushNotification>> {
        self.event_bus.as_ref().map(|tx| tx.subscribe())
    }

    /// Require a bearer token on every request.
    ///
    /// Enables authentication on the server and advertises `bearer` as a
    /// supported scheme on the agent card. Requests without a matching
    /// `Authorization: Bearer <token>` header are rejected with a 401.
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.expected_token = Some(token.into());
        self.card = self
            .card
            .clone()
            .with_authentication(vec!["bearer".to_string()]);
        self
    }

    /// Attach a rate limiter applied to every incoming request.
    pub fn with_rate_limiter(mut self, limiter: Arc<RateLimiter>) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    /// Set the task time-to-live before expiry cleanup (`None` disables expiry).
    pub fn with_task_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.task_ttl = ttl;
        self
    }

    /// Spawn a background sweeper that periodically scans for expired tasks
    /// (P1-2), in addition to the lazy cleanup on the read paths.
    ///
    /// The loop calls [`sweep_expired_tasks`] every `interval` (clamped to at
    /// least 1s). It runs until the current Tokio runtime shuts down. If the
    /// server has no TTL configured (`with_task_ttl(None)`), no task is
    /// spawned — there is nothing to expire.
    pub fn with_background_cleanup(self, interval: Duration) -> Self {
        let Some(ttl) = self.task_ttl else {
            return self;
        };
        let store = self.store.clone();
        // Clamp away a zero interval (which `tokio::time::interval` rejects);
        // sub-second intervals are allowed so tests can drive the sweeper fast.
        let interval = interval.max(Duration::from_millis(1));
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // The first tick completes immediately; consume it so the first
            // sweep happens after one full interval.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                sweep_expired_tasks(&store, ttl).await;
            }
        });
        self
    }

    /// Set a custom agent card.
    pub fn with_card(mut self, card: AgentCard) -> Self {
        self.card = card;
        self
    }

    /// Get the agent card (for `GET /.well-known/agent-card.json`).
    pub fn get_agent_card(&self) -> &AgentCard {
        &self.card
    }

    /// Handle an incoming A2A request (for `POST /`).
    ///
    /// Applies the optional rate limiter, then dispatches based on the
    /// request method:
    /// - `tasks/send` -> acknowledge a new async task (or continue one)
    /// - `tasks/get` -> return a stored task
    /// - `tasks/cancel` -> cancel a stored task
    /// - `tasks/list` -> list stored tasks
    /// - unknown method -> method_not_found error
    pub async fn handle_a2a_request(&self, req: A2ARequest) -> A2AResponse {
        if let Some(limiter) = &self.rate_limiter {
            if let Err(e) = limiter.try_acquire().await {
                return A2AResponse::error(req.id, 429, e.to_string());
            }
        }
        self.dispatch(req).await
    }

    /// Handle an incoming request with an optional bearer token.
    ///
    /// If the server was configured with [`A2AServer::with_auth_token`],
    /// requests without a matching bearer token are rejected with a 401.
    pub async fn handle_a2a_request_authenticated(
        &self,
        req: A2ARequest,
        bearer: Option<&str>,
    ) -> A2AResponse {
        if let Some(expected) = &self.expected_token {
            match bearer {
                None => return A2AResponse::error(req.id, 401, "Authentication required"),
                Some(token) if token != expected => {
                    return A2AResponse::error(req.id, 401, "Invalid authentication token");
                }
                Some(_) => {}
            }
        }
        self.handle_a2a_request(req).await
    }

    /// Dispatch a request to the matching handler.
    ///
    /// Requests carrying a W3C-style `trace_id` in metadata are logged so a
    /// distributed trace can be followed across agents (P1-5).
    async fn dispatch(&self, req: A2ARequest) -> A2AResponse {
        if let Some(trace_id) = req.trace_id() {
            log::debug!(
                "a2a request method={} id={} trace_id={}",
                req.method,
                req.id,
                trace_id
            );
        }
        match req.method.as_str() {
            "tasks/send" => self.handle_tasks_send(req).await,
            "tasks/get" => self.handle_tasks_get(req).await,
            "tasks/cancel" => self.handle_tasks_cancel(req).await,
            "tasks/list" => self.handle_tasks_list(req).await,
            "tasks/runWorkflow" => self.handle_workflow_run(req).await,
            _ => A2AResponse::from_error_data(req.id, A2AErrorData::method_not_found()),
        }
    }

    /// Handle `tasks/send`: create a new async task (or continue an existing
    /// one) and run the chain in the background.
    ///
    /// New tasks are acknowledged immediately with a `submitted` task and the
    /// chain runs in a spawned task. When the request carries a `taskId`
    /// (continuation, P2-2/P2-3), the message is appended to that task's
    /// history and it is re-run. A `message_id` makes the call idempotent
    /// (P1-6). A `skillId` param routes to a different chain (P2-4).
    async fn handle_tasks_send(&self, req: A2ARequest) -> A2AResponse {
        let params = match req.params.clone() {
            Some(p) => p,
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing params for tasks/send"),
                )
            }
        };

        let message = extract_message(&params);

        // P1-6: idempotent re-send — a repeated message_id returns the
        // already-created task instead of running the chain a second time.
        let message_id = req.message_id().map(|s| s.to_string());
        if let Some(mid) = &message_id {
            if let Some(existing_id) = self.message_ids.read().await.get(mid).cloned() {
                if let Ok(Some(stored)) = self.store.get(&existing_id).await {
                    if !self.caller_owns(&req, &stored.task) {
                        return forbidden(req.id, "caller does not own the existing task");
                    }
                    return A2AResponse::ok(req.id, json!({ "task": stored.task }));
                }
                // The referenced task was evicted/expired: forget the mapping
                // and treat this as a fresh send.
                self.message_ids.write().await.remove(mid);
            }
        }

        // P2-2/P2-3: continuation — append to an existing task and re-run.
        if let Some(task_id) = req.task_id().map(|s| s.to_string()) {
            return self
                .handle_tasks_send_continue(req, task_id, message, message_id)
                .await;
        }

        // Fresh task.
        let task_id = uuid::Uuid::new_v4().to_string();
        let mut task = A2ATask::new(task_id.clone(), message);
        if let Some(owner) = req.owner() {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
        // P1-5: carry the request's trace id onto the task so the trace can be
        // correlated after creation.
        if let Some(trace_id) = req.trace_id() {
            stored = stored.with_trace_id(trace_id);
        }
        if self.store.upsert(stored).await.is_err() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }

        if let Some(mid) = &message_id {
            self.message_ids
                .write()
                .await
                .insert(mid.clone(), task_id.clone());
        }

        // P2-4: route by skill if a router is configured.
        let skill_id = params.get("skillId").and_then(Value::as_str);
        let chain = self.resolve_chain(skill_id);

        let store = self.store.clone();
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus).await;
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }

    /// Append a message to an existing task and re-run it (P2-2/P2-3).
    ///
    /// Only tasks in the `input-required` state can be resumed — this is the
    /// one A2A flow where the client sends another message to the same task
    /// (to supply the information the agent asked for). Continuing a
    /// `working` task would spawn a second background worker that races the
    /// first, and terminal tasks cannot change, so both are rejected with
    /// `-32004`.
    async fn handle_tasks_send_continue(
        &self,
        req: A2ARequest,
        task_id: String,
        message: A2AMessage,
        message_id: Option<String>,
    ) -> A2AResponse {
        let mut stored = match self.store.get(&task_id).await {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => return task_not_found(req.id, &task_id),
        };

        // P1-4: ownership.
        if !self.caller_owns(&req, &stored.task) {
            return forbidden(req.id, "caller does not own this task");
        }

        // P2-3: only an input-required task can be resumed with new input.
        if stored.task.status != TaskStatus::InputRequired {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(
                    -32004,
                    format!("Cannot continue task in state {}", stored.task.status),
                ),
            );
        }

        stored.task.push_message(message);
        if stored.task.status.can_transition_to(&TaskStatus::Working) {
            stored.task.status = TaskStatus::Working;
        }
        stored.touch();
        let task = stored.task.clone();
        if self.store.upsert(stored).await.is_err() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }

        // P1-6: idempotency — a retried resume with the same message_id must
        // not append the message a second time.
        if let Some(mid) = message_id {
            self.message_ids.write().await.insert(mid, task_id.clone());
        }

        let store = self.store.clone();
        let chain = self.resolve_chain(None);
        let bus = self.event_bus.clone();
        let history = task.message_history().into_owned();
        let spawned_id = task_id.clone();
        tokio::spawn(async move {
            run_task(&store, chain, &spawned_id, history, bus).await;
        });

        A2AResponse::ok(req.id, json!({ "task": task }))
    }

    /// Handle `tasks/get`: return a task by ID.
    ///
    /// Ownership (P1-4): tasks carrying an `owner` are only readable by the
    /// matching caller.
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

        self.cleanup_expired_tasks().await;

        match self.store.get(task_id).await {
            Ok(Some(stored)) => {
                if !self.caller_owns(&req, &stored.task) {
                    return forbidden(req.id, "caller does not own this task");
                }
                task_details_response(req.id, &stored)
            }
            Ok(None) => task_not_found(req.id, task_id),
            Err(_) => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store read failed"),
            ),
        }
    }

    /// Handle `tasks/cancel`: cancel a task by ID.
    ///
    /// Cancellation is only legal from a non-terminal state; cancelling an
    /// already-terminal task is an idempotent no-op that returns it unchanged.
    /// Ownership (P1-4) is enforced like `tasks/get`.
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

        let mut stored = match self.store.get(task_id).await {
            Ok(Some(s)) => s,
            Ok(None) | Err(_) => return task_not_found(req.id, task_id),
        };

        if !self.caller_owns(&req, &stored.task) {
            return forbidden(req.id, "caller does not own this task");
        }

        if stored.task.status.is_terminal() {
            // Idempotent: already finished, return it unchanged.
            return A2AResponse::ok(req.id, json!({ "task": stored.task }));
        }
        if stored.task.status.can_transition_to(&TaskStatus::Cancelled) {
            stored.task.status = TaskStatus::Cancelled;
            stored.touch();
            if self.store.upsert(stored.clone()).await.is_err() {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::internal_error("task store write failed"),
                );
            }
            publish_status(&self.event_bus, task_id, TaskStatus::Cancelled, None);
            A2AResponse::ok(req.id, json!({ "task": stored.task }))
        } else {
            A2AResponse::from_error_data(
                req.id,
                A2AErrorData::new(
                    -32002,
                    format!("Cannot cancel task in state {}", stored.task.status),
                ),
            )
        }
    }

    /// Handle `tasks/list`: list stored tasks, optionally filtered by
    /// `owner` / `status` params (P1-1).
    ///
    /// A caller that carries an `owner` identity and does not pass an explicit
    /// `owner` param only sees its own tasks.
    async fn handle_tasks_list(&self, req: A2ARequest) -> A2AResponse {
        self.cleanup_expired_tasks().await;

        let mut filter = TaskFilter::new();
        if let Some(params) = &req.params {
            if let Some(owner) = params.get("owner").and_then(Value::as_str) {
                filter = filter.with_owner(owner);
            }
            if let Some(status) = params.get("status").and_then(Value::as_str) {
                if let Ok(ts) =
                    serde_json::from_value::<TaskStatus>(Value::String(status.to_string()))
                {
                    filter = filter.with_statuses(vec![ts]);
                }
            }
        }
        if filter.owner.is_none() {
            if let Some(owner) = req.owner() {
                filter = filter.with_owner(owner);
            }
        }

        match self.store.list(&filter).await {
            Ok(stored) => {
                let tasks: Vec<&A2ATask> = stored.iter().map(|s| &s.task).collect();
                A2AResponse::ok(req.id, json!({ "tasks": tasks }))
            }
            Err(_) => A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store read failed"),
            ),
        }
    }

    /// Handle `tasks/runWorkflow`: execute an ordered multi-step workflow and
    /// aggregate per-step results (P2-8).
    ///
    /// A workflow is backed by a single task (id = `workflow.workflow_id` or a
    /// fresh UUID). Steps run in order, each routed to the chain selected by
    /// its `skill_id` (falling back to the default chain). A step failure
    /// marks the workflow task `failed` and stops execution — the results
    /// aggregated up to that point are still returned. Ownership (P1-4) and
    /// trace propagation (P1-5) apply to the backing task like `tasks/send`.
    async fn handle_workflow_run(&self, req: A2ARequest) -> A2AResponse {
        let params = match req.params.clone() {
            Some(p) => p,
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing params for tasks/runWorkflow"),
                );
            }
        };
        let workflow: A2AWorkflow = match params.get("workflow") {
            Some(w) => match serde_json::from_value(w.clone()) {
                Ok(wf) => wf,
                Err(_) => {
                    return A2AResponse::from_error_data(
                        req.id,
                        A2AErrorData::invalid_params("Malformed workflow"),
                    );
                }
            },
            None => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::invalid_params("Missing workflow for tasks/runWorkflow"),
                );
            }
        };
        if workflow.steps.is_empty() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::invalid_params("Workflow has no steps"),
            );
        }

        // Create the backing task, propagating owner and trace id.
        let task_id = workflow
            .workflow_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut task = A2ATask::new(
            task_id.clone(),
            A2AMessage::user(format!(
                "workflow: {}",
                workflow.name.as_deref().unwrap_or("unnamed")
            )),
        )
        .with_status(TaskStatus::Working);
        if let Some(owner) = req.owner() {
            task = task.with_owner(owner);
        }
        let mut stored = StoredTask::new(task.clone());
        if let Some(trace_id) = req.trace_id() {
            stored = stored.with_trace_id(trace_id);
        }
        if self.store.upsert(stored).await.is_err() {
            return A2AResponse::from_error_data(
                req.id,
                A2AErrorData::internal_error("task store write failed"),
            );
        }
        publish_status(&self.event_bus, &task_id, TaskStatus::Working, None);

        // Execute steps in order, aggregating per-step outputs.
        let mut results = serde_json::Map::new();
        let mut failure: Option<(String, String)> = None; // (step_id, message)
        for step in &workflow.steps {
            let chain = self.resolve_chain(step.skill_id.as_deref());
            let input = build_chain_input(&step.message.content, chain.as_ref());
            match chain.invoke(input).await {
                Ok(result) => {
                    let output = extract_output(&result);
                    results.insert(step.id.clone(), Value::String(output));
                }
                Err(e) => {
                    failure = Some((step.id.clone(), e.to_string()));
                    break;
                }
            }
        }

        // Finalize the backing task with the aggregated outcome.
        let mut finalize = match self.store.get(&task_id).await {
            Ok(Some(s)) => s,
            _ => {
                return A2AResponse::from_error_data(
                    req.id,
                    A2AErrorData::internal_error("workflow task vanished"),
                );
            }
        };
        let aggregated = results
            .values()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        match failure {
            Some((step_id, message)) => {
                finalize.task.status = TaskStatus::Failed;
                finalize.error = Some(format!("step `{step_id}` failed: {message}"));
                let error = finalize.error.clone();
                let _ = self.store.upsert(finalize.clone()).await;
                publish_status(
                    &self.event_bus,
                    &task_id,
                    TaskStatus::Failed,
                    error.as_deref(),
                );
                A2AResponse::ok(
                    req.id,
                    json!({
                        "task": finalize.task,
                        "error": error,
                        "results": Value::Object(results),
                    }),
                )
            }
            None => {
                finalize.task.status = TaskStatus::Completed;
                finalize.result = Some(A2ATaskResult::new(aggregated));
                finalize.error = None;
                let _ = self.store.upsert(finalize.clone()).await;
                publish_status(&self.event_bus, &task_id, TaskStatus::Completed, None);
                publish_artifact(&self.event_bus, &task_id, finalize.result.clone().unwrap());
                A2AResponse::ok(
                    req.id,
                    json!({
                        "task": finalize.task,
                        "result": finalize.result,
                        "results": Value::Object(results),
                    }),
                )
            }
        }
    }

    /// Whether `req` may access a task with `owner`-based protection (P1-4).
    ///
    /// Tasks without an `owner` are open to any caller; tasks with an `owner`
    /// are only accessible to a caller whose metadata `owner` matches exactly.
    fn caller_owns(&self, req: &A2ARequest, task: &A2ATask) -> bool {
        match &task.owner {
            Some(task_owner) => req.owner() == Some(task_owner.as_str()),
            None => true,
        }
    }

    /// Resolve the chain for a skill id, falling back to the default chain
    /// (P2-4).
    fn resolve_chain(&self, skill_id: Option<&str>) -> Arc<dyn BaseChain> {
        if let Some(sid) = skill_id {
            if let Some(router) = &self.skill_router {
                if let Some(chain) = router.chain_for(sid) {
                    return chain;
                }
            }
        }
        self.chain.clone()
    }

    /// Lazily expire tasks older than the configured TTL.
    ///
    /// Terminal tasks older than the TTL are removed to bound memory; live
    /// tasks older than the TTL are transitioned to `expired`.
    async fn cleanup_expired_tasks(&self) {
        let Some(ttl) = self.task_ttl else {
            return;
        };
        sweep_expired_tasks(&self.store, ttl).await;
    }
}

/// Sweep tasks older than `ttl` (P1-2).
///
/// Terminal tasks are deleted to bound memory; live tasks that may transition
/// are marked `expired`. Shared by the lazy read-path cleanup and the
/// background [`A2AServer::with_background_cleanup`] sweeper.
async fn sweep_expired_tasks(store: &Arc<dyn TaskStore>, ttl: Duration) {
    let Ok(list) = store.list(&TaskFilter::new()).await else {
        return;
    };
    for stored in list {
        if stored.age() < ttl {
            continue;
        }
        if stored.task.status.is_terminal() {
            // Free memory for tasks that have been terminal for > ttl.
            let _ = store.delete(&stored.task.id).await;
        } else if stored.task.status.can_transition_to(&TaskStatus::Expired) {
            let mut expired = stored;
            expired.task.status = TaskStatus::Expired;
            expired.touch();
            let _ = store.upsert(expired).await;
        }
    }
}

/// Execute a task in the background: `submitted -> working ->
/// completed/failed/input-required`, guarded by the task state machine so it
/// never clobbers a terminal status (e.g. a task cancelled while the chain
/// was still running).
async fn run_task(
    store: &Arc<dyn TaskStore>,
    chain: Arc<dyn BaseChain>,
    task_id: &str,
    history: Vec<A2AMessage>,
    event_bus: Option<Arc<broadcast::Sender<TaskPushNotification>>>,
) {
    // submitted / input-required -> working
    if let Ok(Some(mut stored)) = store.get(task_id).await {
        if stored.task.status.can_transition_to(&TaskStatus::Working) {
            stored.task.status = TaskStatus::Working;
            stored.touch();
            let _ = store.upsert(stored).await;
            publish_status(&event_bus, task_id, TaskStatus::Working, None);
        }
    }

    let input = build_chain_input_from_history(&history, chain.as_ref());
    match chain.invoke(input).await {
        Ok(result) => {
            let output = extract_output(&result);
            if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Completed) {
                    stored.task.status = TaskStatus::Completed;
                    stored.result = Some(A2ATaskResult::new(output.clone()));
                    stored.error = None;
                    stored.touch();
                    let _ = store.upsert(stored).await;
                    publish_status(&event_bus, task_id, TaskStatus::Completed, None);
                    publish_artifact(&event_bus, task_id, A2ATaskResult::new(output));
                }
            }
        }
        Err(e) => {
            if is_input_required(&e) {
                // P2-3: the chain is asking the client for more information.
                let prompt = e.to_string();
                if let Ok(Some(mut stored)) = store.get(task_id).await {
                    if stored
                        .task
                        .status
                        .can_transition_to(&TaskStatus::InputRequired)
                    {
                        stored.task.status = TaskStatus::InputRequired;
                        stored.error = Some(prompt.clone());
                        stored.touch();
                        let _ = store.upsert(stored).await;
                        publish_status(
                            &event_bus,
                            task_id,
                            TaskStatus::InputRequired,
                            Some(&prompt),
                        );
                    }
                }
            } else if let Ok(Some(mut stored)) = store.get(task_id).await {
                if stored.task.status.can_transition_to(&TaskStatus::Failed) {
                    stored.task.status = TaskStatus::Failed;
                    stored.error = Some(e.to_string());
                    stored.touch();
                    let _ = store.upsert(stored).await;
                    publish_status(
                        &event_bus,
                        task_id,
                        TaskStatus::Failed,
                        Some(&e.to_string()),
                    );
                }
            }
        }
    }
}

/// Whether a chain error signals that more input is needed (P2-3).
///
/// `MissingInput` (a required key is absent) and `InputError` (the input is
/// present but incomplete/malformed) are mapped to the `input-required` task
/// state so the client can resume the conversation.
fn is_input_required(e: &ChainError) -> bool {
    matches!(e, ChainError::MissingInput(_) | ChainError::InputError(_))
}

/// Extract the `A2AMessage` from `tasks/send` params.
///
/// Reads `params.message` (a message object); when absent, the whole params
/// become the input content.
fn extract_message(params: &Value) -> A2AMessage {
    match params.get("message") {
        Some(msg_val) => serde_json::from_value(msg_val.clone()).unwrap_or_else(|_| {
            A2AMessage::new(
                "user",
                msg_val
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
            )
        }),
        None => A2AMessage::user(params.to_string()),
    }
}

/// Build the chain input map from a full message history (P2-2).
///
/// A single-message history keeps its original content (backward compatible);
/// multi-turn histories are joined as `role: content` lines so the chain sees
/// the whole conversation.
fn build_chain_input_from_history(
    history: &[A2AMessage],
    chain: &dyn BaseChain,
) -> HashMap<String, Value> {
    let content = if history.len() == 1 {
        history[0].content.clone()
    } else {
        history
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    };
    build_chain_input(&content, chain)
}

/// Build the chain input map from message content, using the chain's first
/// declared input key (or a fallback `"input"` key).
fn build_chain_input(content: &str, chain: &dyn BaseChain) -> HashMap<String, Value> {
    let mut map = HashMap::new();
    let input_keys = chain.input_keys();
    if let Some(first_key) = input_keys.first() {
        map.insert(first_key.to_string(), Value::String(content.to_string()));
    } else {
        map.insert("input".to_string(), Value::String(content.to_string()));
    }
    map
}

/// Extract the output text from a chain result (first value).
fn extract_output(result: &ChainResult) -> String {
    result
        .values()
        .next()
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Build a `{task, result?, error?}` response for `tasks/get`.
fn task_details_response(id: u64, stored: &StoredTask) -> A2AResponse {
    let mut result = json!({ "task": stored.task });
    if let Some(ref task_result) = stored.result {
        result["result"] = json!(task_result);
    }
    if let Some(ref error) = stored.error {
        result["error"] = json!(error);
    }
    A2AResponse::ok(id, result)
}

/// `-32001` task-not-found error.
fn task_not_found(id: u64, task_id: &str) -> A2AResponse {
    A2AResponse::from_error_data(
        id,
        A2AErrorData::new(-32001, format!("Task not found: {}", task_id)),
    )
}

/// `-32003` ownership violation error.
fn forbidden(id: u64, message: impl Into<String>) -> A2AResponse {
    A2AResponse::from_error_data(id, A2AErrorData::new(-32003, message))
}

/// Publish a status-update event on the optional event bus (P2-1).
fn publish_status(
    bus: &Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    task_id: &str,
    status: TaskStatus,
    error: Option<&str>,
) {
    if let Some(sender) = bus {
        let event = match error {
            Some(e) => TaskPushNotification::status_with_error(task_id, status, e),
            None => TaskPushNotification::status(task_id, status),
        };
        let _ = sender.send(event);
    }
}

/// Publish an artifact-update event on the optional event bus (P2-1).
fn publish_artifact(
    bus: &Option<Arc<broadcast::Sender<TaskPushNotification>>>,
    task_id: &str,
    artifact: A2ATaskResult,
) {
    if let Some(sender) = bus {
        let _ = sender.send(TaskPushNotification::artifact(task_id, artifact));
    }
}

#[cfg(test)]
mod tests {
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
            .handle_a2a_request(
                A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"),
            )
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
            .handle_a2a_request(
                A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"),
            )
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
            .handle_a2a_request(
                A2ARequest::send_task(1, &A2AMessage::user("hi")).with_owner("alice"),
            )
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

        // Anonymous listing sees everything (no owner filter requested).
        let resp = server
            .handle_a2a_request(A2ARequest::new(4, "tasks/list", None))
            .await;
        let tasks = resp.result.unwrap()["tasks"].as_array().unwrap().clone();
        assert_eq!(tasks.len(), 2);
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
        let router = SkillMapRouter::new()
            .with_skill("math", Arc::new(NamedChain("math-chain".to_string())));
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
        let router = SkillMapRouter::new()
            .with_skill("math", Arc::new(NamedChain("math-chain".to_string())));
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
}
