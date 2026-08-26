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

mod execution;
mod handlers;
mod message;
mod routes;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{broadcast, RwLock};

use lc_agents::AgentExecutor;
use lc_chains::base::BaseChain;

use super::agent_adapter::AgentExecutorChain;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2ATaskResult, A2AWorkflow,
    AgentCard, AgentSkill, TaskFilter, TaskPushNotification, TaskStatus,
};
use super::rate_limiter::RateLimiter;
use super::router::{SkillMapRouter, SkillRouter};
use super::store::{InMemoryTaskStore, StoredTask, TaskStore, DEFAULT_MAX_TASKS};

use execution::{run_task, sweep_expired_tasks, InflightResume};
use handlers::{
    forbidden, publish_artifact, publish_status, task_details_response, task_not_found,
};
use message::{build_chain_input, extract_message, extract_output};

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
    ///
    /// A mapping whose value is the empty string marks a `message_id` claimed
    /// by an in-flight request whose task has not been created yet; concurrent
    /// retries with the same id see it and are rejected instead of double-executing.
    message_ids: Arc<RwLock<HashMap<String, String>>>,
    /// Task ids currently being resumed by `tasks/send_continue`.
    ///
    /// Guards the read-check-write of the resume path so two concurrent
    /// resumes of the same `input-required` task cannot both pass the state
    /// check and spawn racing workers (P2-3). A `std::sync::Mutex` suffices:
    /// the critical section is a short contains+insert with no awaits.
    inflight_resumes: Arc<std::sync::Mutex<HashSet<String>>>,
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
            inflight_resumes: Arc::new(std::sync::Mutex::new(HashSet::new())),
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
    /// The loop calls `sweep_expired_tasks` every `interval` (clamped to at
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

    /// Reserve a `message_id` for an idempotent `tasks/send` (P1-6).
    ///
    /// The reservation makes the check-then-act atomic: only the caller that
    /// wins the claim may create the task, so two concurrent retries with the
    /// same `message_id` cannot both run the chain.
    ///
    /// Returns:
    /// - `Ok(Some(task_id))`: a prior send with this `message_id` completed
    ///   and its task still exists.
    /// - `Ok(None)`: this caller won the reservation; it must create the task
    ///   and then call [`Self::finish_message_id`], or [`Self::abort_message_id`]
    ///   if creation fails.
    /// - `Err(())`: another request with the same `message_id` is being
    ///   processed right now; the caller should return a retryable error.
    async fn reserve_message_id(&self, mid: &str) -> Result<Option<String>, ()> {
        // Read the current mapping under a short lock; never await inside it.
        let mapped = { self.message_ids.read().await.get(mid).cloned() };
        if let Some(task_id) = mapped {
            if !task_id.is_empty() {
                return match self.store.get(&task_id).await {
                    Ok(Some(_)) => Ok(Some(task_id)),
                    // Referenced task evicted/expired: reclaim the id.
                    _ => self.claim_message_id(mid).await,
                };
            }
            // In-flight reservation by another request.
            return Err(());
        }
        self.claim_message_id(mid).await
    }

    /// Atomically claim `mid`, inserting an in-flight marker.
    async fn claim_message_id(&self, mid: &str) -> Result<Option<String>, ()> {
        let mut guard = self.message_ids.write().await;
        match guard.get(mid).cloned() {
            Some(task_id) if !task_id.is_empty() => Ok(Some(task_id)), // finished concurrently
            Some(_) => Err(()),                                        // claimed concurrently
            None => {
                guard.insert(mid.to_string(), String::new());
                Ok(None)
            }
        }
    }

    /// Record that a send carrying `mid` created task `task_id`.
    async fn finish_message_id(&self, mid: &str, task_id: &str) {
        self.message_ids
            .write()
            .await
            .insert(mid.to_string(), task_id.to_string());
    }

    /// Release an unused `message_id` reservation (task creation failed).
    async fn abort_message_id(&self, mid: &str) {
        self.message_ids.write().await.remove(mid);
    }

    /// Release a `message_id` reservation held by a continuation that failed
    /// before completing, so a retry can claim it again.
    async fn release_resume_id(&self, message_id: &Option<String>) {
        if let Some(mid) = message_id {
            self.abort_message_id(mid).await;
        }
    }

    /// Claim a task id for an in-flight resume (P2-3).
    ///
    /// Returns `None` if the task is already being resumed by another request.
    /// The returned guard releases the claim on drop, covering every exit path
    /// (early returns included).
    fn begin_resume(&self, task_id: &str) -> Option<InflightResume> {
        let mut guard = self
            .inflight_resumes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if guard.contains(task_id) {
            return None;
        }
        guard.insert(task_id.to_string());
        Some(InflightResume {
            inner: self.inflight_resumes.clone(),
            task_id: task_id.to_string(),
        })
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

#[cfg(test)]
mod tests;
