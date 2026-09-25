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

mod auth;
mod execution;
mod handlers;
mod message;
mod routes;

pub use auth::{AuthError, Authenticator, Principal, StaticBearer};

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::{broadcast, RwLock};

use lc_agents::AgentExecutor;
use lc_chains::base::BaseChain;

use super::agent_adapter::AgentExecutorChain;

use super::protocol::{
    A2AErrorData, A2AMessage, A2ARequest, A2AResponse, A2ATask, A2AWorkflow, AgentCard, AgentSkill,
    TaskFilter, TaskPushNotification, TaskStatus,
};
use super::rate_limiter::RateLimiter;
use super::router::{SkillMapRouter, SkillRouter};
use super::store::{InMemoryTaskStore, StoredTask, TaskStore, DEFAULT_MAX_TASKS};
use crate::security::SandboxConfig;

use execution::{run_task, run_workflow, sweep_expired_tasks, InflightResume, MAX_WORKFLOW_STEPS};
use handlers::{forbidden, publish_status, task_details_response, task_not_found};
use message::extract_message;

/// Default task time-to-live before expiry cleanup (24 hours).
const DEFAULT_TASK_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Maximum number of tracked idempotency keys before the oldest entry is
/// evicted (0.22.0 audit fix H-P5: the `message_id` table used to grow
/// unboundedly).
const MAX_MESSAGE_IDS: usize = 10_000;

/// Bounded insertion-ordered map backing the `message_id -> task_id`
/// idempotency table (0.22.0 audit fix H-P5).
///
/// The `map` holds the reservation state (empty `task_id` = in-flight claim);
/// `order` records insertion order so the oldest entry can be evicted when
/// the table reaches [`MAX_MESSAGE_IDS`]. Aborted keys leave a stale entry in
/// `order`, which eviction skips over.
#[derive(Default)]
struct MessageIdTable {
    map: HashMap<String, String>,
    order: VecDeque<String>,
}

impl MessageIdTable {
    fn get(&self, mid: &str) -> Option<&String> {
        self.map.get(mid)
    }

    fn insert(&mut self, mid: String, task_id: String) {
        if !self.map.contains_key(&mid) {
            // At capacity, pop-oldest and remove it from the map; skip
            // entries that were already aborted.
            while self.map.len() >= MAX_MESSAGE_IDS {
                match self.order.pop_front() {
                    Some(oldest) if self.map.remove(&oldest).is_some() => break,
                    Some(_) => continue,
                    None => break,
                }
            }
            self.order.push_back(mid.clone());
        }
        self.map.insert(mid, task_id);
    }

    fn remove(&mut self, mid: &str) {
        self.map.remove(mid);
    }
}

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
    /// retries with the same id see it and are rejected instead of
    /// double-executing. The table is bounded (see [`MAX_MESSAGE_IDS`]).
    message_ids: Arc<RwLock<MessageIdTable>>,
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
    /// Per-identity bearer tokens (`token -> owner principal`). When non-empty,
    /// the presented token must match one of these (or `expected_token`); the
    /// matched principal overrides any client-supplied owner and drives SSE
    /// notification filtering (0.25.0 authz fix: self-reported `owner` used to
    /// be a privilege-escalation hole).
    auth_principals: HashMap<String, String>,
    /// Optional rate limiter applied to every request.
    rate_limiter: Option<Arc<RateLimiter>>,
    /// Time-to-live for tasks before they expire.
    task_ttl: Option<Duration>,
    /// Optional least-privilege sandbox applied to incoming requests
    /// (0.25.0: `SandboxConfig` used to have no production call site). Only
    /// its payload-size limit is enforceable at this layer; path/network
    /// rules are for the agent's own tool execution.
    sandbox: Option<Arc<SandboxConfig>>,
    /// A2A `protocolVersion` values this server will accept on incoming
    /// requests (0.25.0 contract enforcement). Requests whose metadata
    /// carries an unsupported version are rejected with `-32600`.
    allowed_protocol_versions: HashSet<String>,
    /// Optional pluggable authenticator (B7). When set it is the sole authority
    /// for resolving a bearer token to a principal; otherwise a
    /// [`StaticBearer`] built from `expected_token`/`auth_principals` is used.
    authenticator: Option<Arc<dyn Authenticator>>,
    /// Whether the unauthenticated boundary may trust a client-supplied
    /// metadata `owner` (B7 escape hatch). The *authenticated* boundary never
    /// trusts metadata — the principal is authoritative. Hardened deployments
    /// set this to `false` so self-reported owners are ignored on every path.
    trust_metadata_owner: bool,
    /// task_id -> cancellation signal, so `tasks/cancel` can terminate an
    /// in-flight `run_task` future rather than only flip status (B7).
    cancellations: Arc<RwLock<HashMap<String, Arc<tokio::sync::Notify>>>>,
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
            message_ids: Arc::new(RwLock::new(MessageIdTable::default())),
            inflight_resumes: Arc::new(std::sync::Mutex::new(HashSet::new())),
            skill_router: None,
            event_bus: None,
            expected_token: None,
            auth_principals: HashMap::new(),
            rate_limiter: None,
            task_ttl: Some(DEFAULT_TASK_TTL),
            sandbox: None,
            allowed_protocol_versions: {
                let mut versions = HashSet::new();
                // 0.3.0 and the accepted A2A 1.0 lines (B7 A2A-1.0 compat fix):
                // real 1.0 clients advertise "1.0"/"1.0.1" and must not be
                // rejected as speaking an unknown version.
                for v in ["0.3.0", "1.0", "1.0.1"] {
                    versions.insert(v.to_string());
                }
                versions
            },
            authenticator: None,
            // 0.25.0 secure default (C-A2A-1): do NOT trust a client-supplied
            // `owner` in request metadata. A self-reported owner is only honored
            // when the operator explicitly opts back in via
            // `with_trust_metadata_owner(true)`; owner is otherwise issued by the
            // server (auth principal stamp) or absent.
            trust_metadata_owner: false,
            cancellations: Arc::new(RwLock::new(HashMap::new())),
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

    /// Register a per-identity bearer token bound to an owner principal.
    ///
    /// Unlike [`A2AServer::with_auth_token`] (one token shared by every
    /// caller, so no per-tenant identity exists), every token registered here
    /// names a distinct principal: requests authenticated with it run as that
    /// owner regardless of any `owner` field the client supplies, and the SSE
    /// stream only delivers notifications for that owner's tasks. Register one
    /// token per tenant/principal.
    pub fn with_auth_identity(
        mut self,
        token: impl Into<String>,
        owner: impl Into<String>,
    ) -> Self {
        if self.auth_principals.is_empty() && self.expected_token.is_none() {
            self.card = self
                .card
                .clone()
                .with_authentication(vec!["bearer".to_string()]);
        }
        self.auth_principals.insert(token.into(), owner.into());
        self
    }

    /// Install a custom [`Authenticator`] as the sole authority for resolving
    /// a bearer token to a [`Principal`].
    ///
    /// When set, `check_auth` delegates to it and ignores the legacy
    /// `expected_token`/`auth_principals` tables. The returned principal drives
    /// owner-based authorization and SSE notification filtering.
    pub fn with_authenticator(mut self, authenticator: Arc<dyn Authenticator>) -> Self {
        self.authenticator = Some(authenticator);
        self.card = self
            .card
            .clone()
            .with_authentication(vec!["bearer".to_string()]);
        self
    }

    /// Whether a client-supplied metadata `owner` is trusted on the
    /// *unauthenticated* boundary (defaults to `false` — the 0.25.0 secure
    /// default, see MIGRATION).
    ///
    /// The authenticated boundary (via [`Self::with_auth_token`] /
    /// [`Self::with_auth_identity`] / [`Self::with_authenticator`]) never trusts
    /// metadata: the server stamps the request with the authenticated principal,
    /// and any client `owner` is ignored regardless of this flag. With this flag
    /// off (default), an unauthenticated caller's self-reported `owner` is
    /// ignored too, so no request can create or claim an owned task without
    /// authenticating. Opt in to `true` only to restore the legacy pre-0.25
    /// behavior where a naked metadata `owner` is believed on servers that front
    /// the crate with their own authenticating proxy.
    pub fn with_trust_metadata_owner(mut self, trust: bool) -> Self {
        self.trust_metadata_owner = trust;
        self
    }

    /// Whether any authentication is configured on this server.
    pub fn has_auth(&self) -> bool {
        self.authenticator.is_some()
            || self.expected_token.is_some()
            || !self.auth_principals.is_empty()
    }

    /// Attach a rate limiter applied to every incoming request.
    pub fn with_rate_limiter(mut self, limiter: Arc<RateLimiter>) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    /// Attach a sandbox policy. Its payload-size limit is enforced on every
    /// incoming request (oversized envelopes get a `-32600`/HTTP 413 error);
    /// its path/network rules govern the delegated agent's own tool access
    /// and are checked via [`SandboxConfig::check`] there (0.25.0).
    pub fn with_sandbox(mut self, sandbox: Arc<SandboxConfig>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// The configured sandbox, if any. Used by HTTP layers that can measure
    /// the raw wire body before JSON-RPC parsing.
    pub fn sandbox(&self) -> Option<&SandboxConfig> {
        self.sandbox.as_deref()
    }

    /// Accept an additional A2A `protocolVersion` on incoming requests
    /// (0.25.0). `0.3.0` is accepted by default; only requests that
    /// explicitly carry a `protocolVersion` are checked.
    pub fn with_supported_protocol_version(mut self, version: impl Into<String>) -> Self {
        self.allowed_protocol_versions.insert(version.into());
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
        // C-A2A-1 fail-closed: this plain (in-process) entry has no bearer
        // channel of its own — the HTTP/SSE layers extract `Authorization` and
        // call [`Self::handle_a2a_request_authenticated`]. Delegate here with
        // `bearer = None` so an auth-configured server surfaces a 401 instead of
        // silently serving an unauthenticated caller (the legacy behavior
        // honored a self-reported metadata `owner` without any credential).
        self.handle_a2a_request_authenticated(req, None).await
    }

    /// Shared dispatch funnel: apply the JSON-RPC / protocol-version / sandbox /
    /// rate-limit checks, then dispatch with the boundary-resolved `owner`.
    pub(crate) async fn handle_a2a_request_with_owner(
        &self,
        req: A2ARequest,
        owner: Option<&str>,
    ) -> A2AResponse {
        // 0.25.0 contract enforcement: the JSON-RPC envelope must be 2.0 and
        // an explicitly advertised A2A protocolVersion must be one this
        // server speaks. Failures use the JSON-RPC invalid-request code.
        if req.jsonrpc != "2.0" {
            return A2AResponse::error(
                req.id,
                -32600,
                format!("unsupported JSON-RPC version: {}", req.jsonrpc),
            );
        }
        if let Some(version) = req.protocol_version() {
            if !self.allowed_protocol_versions.contains(version) {
                return A2AResponse::error(
                    req.id,
                    -32600,
                    format!("unsupported A2A protocolVersion: {version}"),
                );
            }
        }
        if let Some(sandbox) = &self.sandbox {
            // Framework-neutral fallback: measure the re-serialized request.
            // The axum layer additionally measures the raw wire body.
            let size = serde_json::to_vec(&req).map(|b| b.len()).unwrap_or(0);
            if !sandbox.accepts_payload(size) {
                return A2AResponse::error(
                    req.id,
                    413,
                    format!("request payload of {size} bytes exceeds sandbox limit"),
                );
            }
        }
        // 0.22.0 audit fix (H-P3): bind the permit in the enclosing scope so
        // it is held across the dispatch await. Holding it in a temporary
        // `if let` value dropped it at the end of the statement, before the
        // dispatch ran, so the concurrency cap never applied.
        let _permit = match &self.rate_limiter {
            Some(limiter) => match limiter.try_acquire().await {
                Ok(permit) => Some(permit),
                Err(e) => return A2AResponse::error(req.id, 429, e.to_string()),
            },
            None => None,
        };
        self.dispatch(req, owner).await
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
        let principal = match self.check_auth(bearer) {
            Ok(principal) => principal,
            Err(resp) => return *resp,
        };
        // C-A2A-1: the authenticated principal is authoritative and is stamped
        // as the task owner regardless of the legacy `trust_metadata_owner`
        // flag; a client self-reported owner can never impersonate another
        // tenant. When a shared/system token resolves no principal, ownership
        // falls back to the unauthenticated rules in [`Self::resolve_owner`].
        let resolved = self.resolve_owner(&req, principal.as_deref());
        self.handle_a2a_request_with_owner(req, resolved.as_deref())
            .await
    }

    /// Validate the bearer token if the server requires one.
    ///
    /// Returns `Ok(())` when no token is configured or the token matches;
    /// otherwise `Err` carries the 401 [`A2AResponse`] to return to the caller.
    ///
    /// Shared by the JSON-RPC handler and the SSE streaming endpoint so a
    /// `with_auth_token` server cannot be bypassed by connecting to `/events`
    /// directly (0.20.0 S4 G1).
    // Err 装箱:A2AResponse 最宽 136+ 字节,clippy::result_large_err 要求收窄;
    // 401 响应是仅有的 Err 形态,装箱后调用方按需解包。
    /// Authenticate a presented bearer token.
    ///
    /// Returns the owner principal when the token is identity-bound
    /// ([`A2AServer::with_auth_identity`]); `None` for the shared
    /// [`A2AServer::with_auth_token`] token or an auth-disabled server.
    pub(crate) fn check_auth(
        &self,
        bearer: Option<&str>,
    ) -> Result<Option<String>, Box<A2AResponse>> {
        // A custom authenticator is the sole authority when installed (B7).
        // Otherwise materialize the legacy fields into the default StaticBearer
        // so there is exactly one authentication path (constant-time compare).
        let authenticator: Arc<dyn Authenticator> = match &self.authenticator {
            Some(a) => a.clone(),
            None => Arc::new(self.static_bearer()),
        };
        match authenticator.authenticate(bearer) {
            Ok(Some(principal)) => Ok(Some(principal.0)),
            Ok(None) => Ok(None),
            Err(e) => Err(Box::new(A2AResponse::error(0, 401, e.to_string()))),
        }
    }

    /// The default static-bearer authenticator built from the legacy
    /// `expected_token` / `auth_principals` configuration. Constant-time
    /// comparison; identity tokens resolve to a principal, the shared token to
    /// none.
    fn static_bearer(&self) -> StaticBearer {
        let mut bearer = StaticBearer::new();
        if let Some(token) = &self.expected_token {
            bearer = bearer.with_shared_token(token.clone());
        }
        for (token, principal) in &self.auth_principals {
            bearer = bearer.with_identity(token.clone(), principal.clone());
        }
        bearer
    }

    /// Whether `principal` may receive a push notification for `task_id`.
    ///
    /// An identity-bound SSE connection only sees its own tasks; an unknown or
    /// already-evicted task is hidden rather than leaked. An open or
    /// shared-token server (`principal == None`) preserves the prior behavior
    /// of delivering every notification.
    //
    // The only production call site lives in the axum SSE handler; with the
    // default feature set (no axum) the method is exercised only by tests.
    #[cfg_attr(not(feature = "axum"), allow(dead_code))]
    pub(crate) async fn notification_visible(
        &self,
        principal: Option<&str>,
        task_id: &str,
    ) -> bool {
        match principal {
            Some(principal) => match self.store.get(task_id).await {
                Ok(Some(stored)) => stored.task.owner.as_deref() == Some(principal),
                _ => false,
            },
            // No identity principal. On an auth-configured server this is a
            // shared/top-level token (or a credential that failed to map to an
            // identity) — it must NOT receive other tenants' notifications;
            // only owner-less tasks are visible to it, matching the pull side
            // (`tasks/get`/`tasks/list`). A genuinely open server (no auth at
            // all) keeps the prior "deliver everything" behavior. B7
            // cross-tenant fix: previously a shared-token connection streamed
            // every tenant's ArtifactUpdate while the pull API refused them.
            None => {
                if self.has_auth() {
                    match self.store.get(task_id).await {
                        Ok(Some(stored)) => stored.task.owner.is_none(),
                        _ => false,
                    }
                } else {
                    true
                }
            }
        }
    }

    /// Dispatch a request to the matching handler.
    ///
    /// Requests carrying a W3C-style `trace_id` in metadata are logged so a
    /// distributed trace can be followed across agents (P1-5).
    async fn dispatch(&self, req: A2ARequest, owner: Option<&str>) -> A2AResponse {
        if let Some(trace_id) = req.trace_id() {
            log::debug!(
                "a2a request method={} id={} trace_id={}",
                req.method,
                req.id,
                trace_id
            );
        }
        match req.method.as_str() {
            "tasks/send" => self.handle_tasks_send(req, owner).await,
            "tasks/get" => self.handle_tasks_get(req, owner).await,
            "tasks/cancel" => self.handle_tasks_cancel(req, owner).await,
            "tasks/list" => self.handle_tasks_list(req, owner).await,
            "tasks/runWorkflow" => self.handle_workflow_run(req, owner).await,
            _ => A2AResponse::from_error_data(req.id, A2AErrorData::method_not_found()),
        }
    }

    /// Resolve the trusted owner for a request (C-A2A-1).
    ///
    /// The authenticated principal — resolved by an `Authenticator` at the
    /// HTTP boundary — is always authoritative and wins unconditionally. In the
    /// absence of a principal, a self-reported metadata `owner` is only honored
    /// when the operator explicitly opted into legacy trust via
    /// [`A2AServer::with_trust_metadata_owner`]; otherwise the caller is
    /// treated as un-owned. Authorization never consults the raw metadata
    /// `owner` directly.
    fn resolve_owner(&self, req: &A2ARequest, principal: Option<&str>) -> Option<String> {
        principal.map(String::from).or_else(|| {
            // C-A2A-1: the legacy self-reported-owner trust applies only on a
            // genuinely open server. On an auth-configured server a
            // `principal = None` means a shared/system token was presented —
            // that caller must NOT be able to self-declare an owner and thereby
            // access another tenant's tasks, so the metadata-owner fallback is
            // suppressed here (closes the shared-token + trust_metadata_owner
            // impersonation path).
            if self.has_auth() {
                return None;
            }
            self.trust_metadata_owner
                .then(|| req.owner())
                .flatten()
                .map(String::from)
        })
    }

    /// Whether a caller with the resolved `owner` may access a task with
    /// `owner`-based protection (P1-4).
    ///
    /// Tasks without an `owner` are open to any caller; tasks with an `owner`
    /// are only accessible to a caller whose resolved owner matches exactly.
    /// The raw metadata `owner` is never compared against here — an
    /// unauthenticated caller cannot self-declare ownership (C-A2A-1).
    fn caller_owns(&self, owner: Option<&str>, task: &A2ATask) -> bool {
        match &task.owner {
            Some(task_owner) => owner == Some(task_owner.as_str()),
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
