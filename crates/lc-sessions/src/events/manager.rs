//! Event-sourced session manager (0.22.0 S4.5) — the events-API counterpart
//! of the deprecated mutable-history [`crate::SessionManager`].
//!
//! Same conversation semantics (chat = append user message → LLM → append
//! reply → persist), but persistence is an append-only [`EventStore`]:
//! history is a projection, compaction appends a `Snapshot`, and forks copy
//! prefixes. Crash-safe by construction: every append is idempotent, so a
//! crash between "persist" and "ack" is recovered by replaying the batch.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lc_core::BaseChatModel;
use lc_memory::BaseMemory;
use lc_schema::Message;
use tokio::sync::Mutex;

use super::checkpoint::SessionCheckpoint;
use super::model::{EventPayload, SessionEvent};
use super::replay::{
    age_bounded_turns, merge_projections, project, replay_active_session, ProjectedSession,
};
use super::store::EventStore;
use crate::store::SessionError;

const MAIN_BRANCH: &str = "main";

/// Configuration for automatic compaction: when the projected turn count
/// exceeds `max_turns`, a deterministic `Snapshot` event is appended that
/// represents all but the most recent `max_turns` turns. Mirrors the
/// lc-agents `CompactionStrategy` discipline (turn granularity, no LLM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoCompaction {
    /// Keep at most this many recent turns verbatim (floor: ≥ 1).
    pub max_turns: usize,
    /// Optional age bound (T2): turns older than this are eligible for
    /// compaction even when `max_turns` is not exceeded.
    ///
    /// A turn-count threshold alone never reclaims an *idle* session — three
    /// turns from two years ago sit below every count-based limit, so their
    /// events stay in the log forever. `max_age` is what lets
    /// [`EventSessionManager::gc`] reclaim them. `None` = count-only (the
    /// historical behavior).
    pub max_age: Option<Duration>,
}

impl AutoCompaction {
    /// Validates the turn floor.
    pub fn new(max_turns: usize) -> Result<Self, SessionError> {
        if max_turns == 0 {
            return Err(SessionError::StoreError(
                "AutoCompaction::max_turns must be >= 1".into(),
            ));
        }
        Ok(Self {
            max_turns,
            max_age: None,
        })
    }

    /// Adds an age bound (see [`AutoCompaction::max_age`]).
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }
}

/// Factory that produces a fresh per-session event memory instance.
pub(crate) type MemoryFactory = Arc<dyn Fn() -> Arc<Mutex<dyn BaseMemory>> + Send + Sync + 'static>;
/// Lazily-created per-session event memory instances, keyed by session id.
pub(crate) type SessionMemories = Arc<Mutex<HashMap<String, Arc<Mutex<dyn BaseMemory>>>>>;

/// Event-sourced session manager.
pub struct EventSessionManager {
    store: Arc<dyn EventStore>,
    branch: String,

    /// Optional memory component (same wiring discipline as the deprecated
    /// manager): with memory, the LLM context = memory history + current
    /// user message; without it, the projected session history is used.
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    memory_input_key: String,
    memory_output_key: String,

    /// 0.22.0 H-M1: per-session memory **factory**. When set, each session
    /// gets a fresh memory instance on first use, so one session's history
    /// can never leak into another's context. Without it the legacy shared
    /// `memory` instance is used (manager-level singleton → cross-session
    /// contamination; prefer the factory once memory spans multiple sessions).
    memory_factory: Option<MemoryFactory>,
    /// Lazily-created per-session memory instances, keyed by session id.
    session_memories: SessionMemories,

    /// Turn-based context window: with `Some(n)`, a `chat()` without memory
    /// feeds only the most recent `n` turns' messages; `None` = full history.
    max_context_turns: Option<usize>,

    /// Automatic compaction policy (`None` = never compact).
    auto_compaction: Option<AutoCompaction>,

    /// Optional durable checkpoint (`None` = recovery always replays the whole
    /// log, which is always correct but never skips work).
    checkpoint: Option<Arc<dyn SessionCheckpoint>>,

    /// 0.22.0 C7 fix: per-session-id striped lock serializing the whole
    /// turn_count → append → LLM → append → compaction sequence of `chat()`
    /// (and `record_tool_use`). Without it two concurrent requests compute
    /// the same next event id and the EventStore's idempotent append silently
    /// drops the loser's user message. Same discipline as the deprecated
    /// manager (Q2); the outer map's Mutex guards only the map itself.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

/// Default auto-compaction threshold for [`EventSessionManager`] (0.25.0
/// E1/H-s1). Kept bounded so an unintentionally long session cannot grow the
/// event log without limit; subscribers can raise/lower it or disable it via
/// `with_auto_compaction`.
const DEFAULT_AUTO_COMPACTION_TURNS: usize = 1024;

impl EventSessionManager {
    /// Creates a manager over the given event store (branch `"main"`).
    pub fn new(store: Arc<dyn EventStore>) -> Self {
        Self {
            store,
            branch: MAIN_BRANCH.to_string(),
            memory: None,
            memory_factory: None,
            session_memories: Arc::new(Mutex::new(HashMap::new())),
            memory_input_key: "input".to_string(),
            memory_output_key: "output".to_string(),
            max_context_turns: None,
            // E1/H-s1: a non-`None` default so an unconfigured manager still
            // bounds its event log (crash-safe-by-default). Compaction is a
            // deterministic snapshot — no LLM — so it is safe as a default.
            auto_compaction: Some(AutoCompaction {
                max_turns: DEFAULT_AUTO_COMPACTION_TURNS,
                max_age: None,
            }),
            checkpoint: None,
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Operates on a non-main branch (a fork target).
    pub fn with_branch(mut self, branch: impl Into<String>) -> Self {
        self.branch = branch.into();
        self
    }

    /// Attaches a memory component (see the deprecated manager's P2-1).
    ///
    /// Legacy shared-instance wiring: the *same* memory backs every session,
    /// so sessions share context. Prefer [`Self::with_memory_factory`] when
    /// memory must be isolated per session (H-M1).
    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// 0.22.0 H-M1: attaches a per-session memory **factory**. Each session
    /// receives a fresh memory instance on first use, so its history stays
    /// isolated across sessions (no cross-session contamination). Falls back
    /// to the shared `with_memory` instance when the factory is absent.
    pub fn with_memory_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Arc<Mutex<dyn BaseMemory>> + Send + Sync + 'static,
    {
        self.memory_factory = Some(Arc::new(factory));
        self
    }

    /// Aligns custom memory input/output keys (defaults `"input"`/`"output"`).
    pub fn with_memory_keys(
        mut self,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        self.memory_input_key = input_key.into();
        self.memory_output_key = output_key.into();
        self
    }

    /// Turn-based context window for memory-less chats.
    pub fn with_max_context_turns(mut self, n: usize) -> Self {
        self.max_context_turns = Some(n);
        self
    }

    /// Enables automatic compaction after each turn.
    pub fn with_auto_compaction(mut self, policy: AutoCompaction) -> Self {
        self.auto_compaction = Some(policy);
        self
    }

    /// Attaches a durable checkpoint store (T2), enabling
    /// [`Self::save_checkpoint`] and the fast path of [`Self::recover_session`].
    ///
    /// Without one, recovery replays the whole log — always correct, just never
    /// cheap. [`super::checkpoint::FileCheckpoint`] is the built-in durable
    /// implementation; a Mongo/Redis-backed [`SessionCheckpoint`] plugs in the
    /// same way.
    pub fn with_checkpoint(mut self, checkpoint: Arc<dyn SessionCheckpoint>) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// The attached checkpoint store, if any.
    pub fn checkpoint(&self) -> Option<&Arc<dyn SessionCheckpoint>> {
        self.checkpoint.as_ref()
    }

    /// 0.22.0 C7 fix: gets the striped lock for a session id (lazily created
    /// if absent). Forked managers share the map so a branch and its trunk
    /// (same session id) stay mutually exclusive too.
    async fn session_lock(&self, session_id: &str) -> Arc<Mutex<()>> {
        let mut map = self.locks.lock().await;
        map.entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    async fn next_id(&self, session_id: &str) -> Result<u64, SessionError> {
        Ok(self.store.latest_id(session_id, &self.branch).await? + 1)
    }

    /// 0.22.0 H-M1: resolves the memory for a session. With a factory attached,
    /// each session gets (and reuses) its own fresh instance; otherwise the
    /// legacy shared instance is returned.
    async fn session_memory(&self, session_id: &str) -> Option<Arc<Mutex<dyn BaseMemory>>> {
        if let Some(factory) = &self.memory_factory {
            let mut map = self.session_memories.lock().await;
            Some(
                map.entry(session_id.to_string())
                    .or_insert_with(|| factory())
                    .clone(),
            )
        } else {
            self.memory.clone()
        }
    }

    async fn turn_count(&self, session_id: &str) -> Result<(u64, u64), SessionError> {
        // Returns (next_id, number of user-message events = turn count).
        let events = self.store.read(session_id, &self.branch, None).await?;
        let next_id = events.last().map(|e| e.id + 1).unwrap_or(1);
        let turns = events
            .iter()
            .filter(|e| matches!(e.payload, EventPayload::UserMessage { .. }))
            .count() as u64;
        Ok((next_id, turns))
    }

    async fn append(
        &self,
        session_id: &str,
        turn_index: u64,
        payload: EventPayload,
    ) -> Result<u64, SessionError> {
        let id = self.next_id(session_id).await?;
        self.store
            .append(&SessionEvent::now(
                id,
                session_id,
                turn_index,
                self.branch.clone(),
                payload,
            ))
            .await?;
        Ok(id)
    }

    /// Creates a session (events namespace): appends an empty-marker `Metadata`
    /// event so the session exists in the log, returns its id.
    pub async fn create_session(&self) -> Result<String, SessionError> {
        let id = uuid::Uuid::now_v7().to_string();
        self.append(
            &id,
            0,
            EventPayload::Metadata {
                key: "created".into(),
                value: serde_json::json!({}),
            },
        )
        .await?;
        Ok(id)
    }

    /// E1/H-s1: deletes a session's history and reaps its lazy bookkeeping.
    ///
    /// Removes every stored event for the session (all branches) and drops the
    /// per-session memory + striped lock entries so the manager does not
    /// accumulate them for a session that no longer exists. This closes the
    /// lifecycle loop that `create_session` opens — deleted sessions don't
    /// leave a stale log or orphaned per-session state behind.
    pub async fn delete_session(&self, session_id: &str) -> Result<(), SessionError> {
        let history_exists = !self
            .store
            .read(session_id, &self.branch, None)
            .await?
            .is_empty();
        self.store.clear_session(session_id).await?;
        // T2: drop the durable snapshot too. The event log is gone, so a
        // surviving checkpoint would resurrect the deleted session's history
        // the next time `recover_session` ran.
        if let Some(cp) = &self.checkpoint {
            cp.remove(session_id, &self.branch).await?;
        }
        // Reap lazily-created per-session state (best-effort; the entries may
        // never have been created if the session was only created, not chatted).
        self.session_memories.lock().await.remove(session_id);
        self.locks.lock().await.remove(session_id);
        if history_exists {
            Ok(())
        } else {
            // Nothing was stored for this session — treat as not-found so a
            // caller can distinguish "deleted an existing session" from "no such session".
            Err(SessionError::NotFound(session_id.to_string()))
        }
    }

    /// Reaps a session's lazily-created per-session cache (memory instance + striped lock)
    /// without touching its stored history.
    ///
    /// With an attached [`Self::with_memory_factory`], each session's memory instance is created
    /// on first use and is otherwise held in the manager forever — `delete_session` is the only
    /// entry point that drops it, yet a long-lived manager accumulates one entry per ever-visited
    /// session even after they become idle. `release_session_cache` gives an idle-reaper a valve:
    /// it frees those resident entries while keeping the session and its stored events alive. The
    /// next `chat()` on the session lazily recreates a fresh memory instance (resetting any
    /// in-memory context, per the ordinary per-session-memory semantics). Idempotent — no-op for
    /// sessions that never accumulated cache entries.
    pub async fn release_session_cache(&self, session_id: &str) {
        self.session_memories.lock().await.remove(session_id);
        self.locks.lock().await.remove(session_id);
    }

    /// Full history of a session (compat view: projected messages).
    pub async fn history(&self, session_id: &str) -> Result<Vec<Message>, SessionError> {
        let events = self.store.read(session_id, &self.branch, None).await?;
        if events.is_empty() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        Ok(project(&events)?.messages)
    }

    /// Replays a session into a legacy mutable [`crate::Session`] view.
    pub async fn replay_session(
        &self,
        session_id: &str,
    ) -> Result<crate::session::Session, SessionError> {
        let events = self.store.read(session_id, &self.branch, None).await?;
        if events.is_empty() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        replay_active_session(&events)
    }

    /// Forks a session: copies the branch prefix into `dst_branch` and returns
    /// a manager pinned to it (callers keep the original untouched).
    pub async fn fork_session(
        &self,
        session_id: &str,
        dst_branch: &str,
        until_id: Option<u64>,
    ) -> Result<EventSessionManager, SessionError> {
        self.store
            .fork(session_id, &self.branch, dst_branch, until_id)
            .await?;
        let mut forked = EventSessionManager::new(self.store.clone());
        forked.branch = dst_branch.to_string();
        forked.memory = self.memory.clone();
        // H-M1: a fork continues the same per-session memory wiring (factory
        // and the lazily-created instances) so trunk/fork share each session's
        // context, exactly as a single manager would.
        forked.memory_factory = self.memory_factory.clone();
        forked.session_memories = self.session_memories.clone();
        forked.memory_input_key = self.memory_input_key.clone();
        forked.memory_output_key = self.memory_output_key.clone();
        forked.max_context_turns = self.max_context_turns;
        forked.auto_compaction = self.auto_compaction;
        // T2: a fork is a durable session in its own right, so it inherits the
        // checkpoint store (snapshots are keyed by branch, so trunk and fork
        // never overwrite each other).
        forked.checkpoint = self.checkpoint.clone();
        // C7: share the per-session striped locks between the trunk manager
        // and the fork so concurrent chat on both cannot race the same log.
        forked.locks = self.locks.clone();
        Ok(forked)
    }

    /// Chats within a session: append user message → LLM → append reply →
    /// persist. Same contract as the deprecated manager's `chat`.
    pub async fn chat<L: BaseChatModel>(
        &self,
        session_id: &str,
        llm: &L,
        user_message: String,
    ) -> Result<String, SessionError>
    where
        L::Error: std::fmt::Display,
    {
        // C7 fix: serialize the whole turn against concurrent chat()/record_tool_use()
        // on the same session — without the lock, two requests compute the same
        // next id and the idempotent append silently drops one user message.
        let session_lock = self.session_lock(session_id).await;
        let _session_guard = session_lock.lock().await;

        let (next_id, turn_count) = self.turn_count(session_id).await?;
        if next_id == 1 {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        let turn_index = turn_count;

        // E2/M-s1: the user message is NOT persisted up front. If the LLM call
        // below fails, persisting it first would leave an orphan user message
        // in the log (a message with no reply), which the next `chat()` would
        // project as dangling context. Instead the current user message is
        // folded into the LLM context explicitly, and both the UserMessage and
        // AssistantMessage are appended only after the LLM responds. (Legacy
        // manager parity: write-after-success.)

        // 1. Build the LLM context. H-M1: resolve this session's own memory
        // (per-session factory instance, or the legacy shared one).
        let session_memory = self.session_memory(session_id).await;
        let response = if let Some(memory) = &session_memory {
            let history_messages = {
                let mem = memory.lock().await;
                let inputs = HashMap::from([(self.memory_input_key.clone(), user_message.clone())]);
                let vars = mem
                    .load_memory_variables(&inputs)
                    .await
                    .map_err(|e| SessionError::Memory(format!("failed to load memory: {e}")))?;
                lc_memory::memory_variables_to_messages(&vars)
            };
            let mut messages = history_messages;
            messages.push(Message::human(&user_message));
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::Llm(e.to_string()))?
        } else {
            let events = self.store.read(session_id, &self.branch, None).await?;
            let mut messages = project(&events)?.messages;
            if let Some(n) = self.max_context_turns {
                messages = last_n_turns_messages(&events, n)?;
            }
            // fold the current user message in (not yet persisted)
            messages.push(Message::human(&user_message));
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::Llm(e.to_string()))?
        };

        // 2. Commit memory *before* the event-log appends (§3.2): memory save
        // is gated on the LLM succeeding, and the log turn is appended only
        // after memory persists. The old "log first, memory second" order meant
        // a failed `save_context` left a committed, replayable log turn that a
        // caller retried into a *duplicate* (each append is a fresh id; there is
        // no content-level de-duplication). The residual risk — memory committed
        // but a later append fails — is far rarer and self-heals: the next load
        // simply feeds both from memory, and the log turn re-appends once.
        let content = response.content.clone();
        if let Some(memory) = &session_memory {
            let mut mem = memory.lock().await;
            let inputs = HashMap::from([(self.memory_input_key.clone(), user_message.clone())]);
            let outputs = HashMap::from([(self.memory_output_key.clone(), content.clone())]);
            mem.save_context(&inputs, &outputs)
                .await
                .map_err(|e| SessionError::Memory(format!("failed to save memory: {e}")))?;
        }

        // 3. Persist the turn now that both the LLM and memory succeeded: user
        // message first (opens the turn), then the reply.
        self.append(
            session_id,
            turn_index,
            EventPayload::UserMessage {
                content: user_message,
            },
        )
        .await?;
        self.append(
            session_id,
            turn_index,
            EventPayload::AssistantMessage {
                content: content.clone(),
            },
        )
        .await?;

        // 4. Auto-compaction (deterministic snapshot, no LLM), shared with the
        // idle-session `gc()` path.
        if let Some(policy) = self.auto_compaction {
            self.compact_and_prune(session_id, policy.max_turns).await?;
        }

        Ok(content)
    }

    /// Compaction step shared by `chat()` (count-triggered) and [`Self::gc`]
    /// (age-triggered).
    ///
    /// Appends the deterministic Snapshot standing in for everything beyond the
    /// most recent `keep_turns` turns, then physically prunes the summarized
    /// prefix (§3.1): the snapshot is the durable replacement for those dropped
    /// turns, so their raw events are removed and the log stays *bounded*
    /// instead of growing without limit. Returns `true` when the log changed.
    ///
    /// Callers hold the session's striped lock.
    async fn compact_and_prune(
        &self,
        session_id: &str,
        keep_turns: usize,
    ) -> Result<bool, SessionError> {
        let events = self.store.read(session_id, &self.branch, None).await?;
        let next_id = events.last().map(|e| e.id + 1).unwrap_or(1);
        let Some(snapshot) = super::replay::compact_to_snapshot(next_id, &events, keep_turns)?
        else {
            return Ok(false);
        };
        let prune_before = super::replay::compaction_prune_boundary(&events, keep_turns)?;
        self.store.append(&snapshot).await?;
        if let Some(before_id) = prune_before {
            self.store
                .delete_before(session_id, &self.branch, before_id)
                .await?;
        }
        Ok(true)
    }

    /// Reclaims an idle session's event log (T2: physical GC).
    ///
    /// `chat()` only compacts while someone is talking to the session, and only
    /// once the turn count crosses `max_turns`. A session that went quiet — a
    /// support thread with three turns from last year — therefore keeps every
    /// event it ever wrote. `gc()` applies the same deterministic snapshot +
    /// prune to a session nobody is talking to, tightening the turn budget by
    /// [`AutoCompaction::max_age`] so turns older than the bound are compacted
    /// away even when the count is low.
    ///
    /// A scheduled reaper can sweep idle sessions with this; it takes the
    /// session's striped lock, so it cannot race a concurrent `chat()`.
    ///
    /// Returns `true` when the log was compacted, `false` when there was nothing
    /// to do (no policy attached, or the session is already within budget).
    /// Errors with [`SessionError::NotFound`] for a session with no events.
    pub async fn gc(&self, session_id: &str) -> Result<bool, SessionError> {
        let session_lock = self.session_lock(session_id).await;
        let _session_guard = session_lock.lock().await;

        let Some(policy) = self.auto_compaction else {
            return Ok(false);
        };
        let events = self.store.read(session_id, &self.branch, None).await?;
        if events.is_empty() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        // With an age bound, the effective budget is the tighter of the two:
        // at most `max_turns` turns, and only those that ended after the cutoff.
        let keep_turns = match policy.max_age {
            Some(max_age) => {
                let now_ms = chrono::Utc::now().timestamp_millis();
                age_bounded_turns(&project(&events)?.turns, policy.max_turns, max_age, now_ms)
            }
            None => policy.max_turns,
        };
        self.compact_and_prune(session_id, keep_turns).await
    }

    /// Materializes the session's current projection into the attached
    /// checkpoint (T2).
    ///
    /// Deliberately explicit rather than automatic: the projection is
    /// O(history), so writing one on every turn would be pure write
    /// amplification, and `chat()` should not grow a new failure mode for a
    /// step it does not need. Call it on whatever cadence fits — after every N
    /// turns, when a session goes idle, or from a periodic task.
    ///
    /// Returns `true` when a checkpoint was written, `false` when no checkpoint
    /// store is attached (nothing to write to). Errors with
    /// [`SessionError::NotFound`] for a session with no events.
    pub async fn save_checkpoint(&self, session_id: &str) -> Result<bool, SessionError> {
        let Some(cp) = &self.checkpoint else {
            return Ok(false);
        };
        let events = self.store.read(session_id, &self.branch, None).await?;
        if events.is_empty() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        let projection = project(&events)?;
        cp.save(session_id, &self.branch, projection.last_id, &projection)
            .await?;
        Ok(true)
    }

    /// Rebuilds a session's projection, using the checkpoint to skip most of
    /// the replay (T2).
    ///
    /// Fast path: the latest snapshot for `(session_id, branch)` is loaded and
    /// only the events appended since its `up_to_id` are replayed, then merged
    /// ([`merge_projections`]). The result is identical to projecting the whole
    /// log — that is the checkpoint's contract. Without an attached checkpoint,
    /// without a stored snapshot, or when the snapshot is behind a compaction
    /// that rewrote the log's shape, this degrades to a full replay, which is
    /// always correct.
    ///
    /// Errors with [`SessionError::NotFound`] for a session with no events.
    pub async fn recover_session(
        &self,
        session_id: &str,
    ) -> Result<ProjectedSession, SessionError> {
        let events = self.store.read(session_id, &self.branch, None).await?;
        if events.is_empty() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        if let Some(cp) = &self.checkpoint {
            if let Some((up_to_id, base)) = cp.latest(session_id, &self.branch).await? {
                // A snapshot ahead of the log (the log was pruned or cleared
                // after it was written) cannot be trusted as a prefix.
                if up_to_id <= events.last().map(|e| e.id).unwrap_or(0) {
                    let tail: Vec<SessionEvent> =
                        events.iter().filter(|e| e.id > up_to_id).cloned().collect();
                    if tail.is_empty() {
                        return Ok(base);
                    }
                    return Ok(merge_projections(base, project(&tail)?));
                }
            }
        }
        project(&events)
    }

    /// Appends an explicit tool call/result pair (for agent-style loops that
    /// want their tool activity inside the session log). Both events share
    /// the current turn; pairing is by `call_id`.
    pub async fn record_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        tool_input: serde_json::Value,
        call_id: &str,
        observation: &str,
    ) -> Result<(), SessionError> {
        // C7 fix: same serialization as chat() — the Call/Result pair must
        // not interleave with another turn's events under concurrency.
        let session_lock = self.session_lock(session_id).await;
        let _session_guard = session_lock.lock().await;

        // L12: parity with chat()/history()/replay_session() — a session with no events
        // does not exist, so recording tool activity into it must error rather than
        // silently creating an orphaned tool log for a never-created session.
        let (next_id, turn_count) = self.turn_count(session_id).await?;
        if next_id == 1 {
            return Err(SessionError::NotFound(session_id.to_string()));
        }
        self.append(
            session_id,
            turn_count,
            EventPayload::ToolCall {
                tool: tool.to_string(),
                tool_input,
                call_id: call_id.to_string(),
            },
        )
        .await?;
        self.append(
            session_id,
            turn_count,
            EventPayload::ToolResult {
                call_id: call_id.to_string(),
                observation: observation.to_string(),
            },
        )
        .await?;
        Ok(())
    }
}

/// Messages of the most recent turns (projected from the log).
///
/// E2/M-s1 window semantics: `n` = number of *completed* turns kept. Since the
/// current in-flight user message is persisted only after the LLM succeeds, it
/// is not in `events`; `chat()` appends it to the returned slice. So window 1 =
/// the previous completed turn's messages, and combined with the caller's human
/// message that mirrors the deprecated manager's message-count window (parity:
/// `2n` completed messages + 1 current, same as the legacy pre-write behavior).
///
/// 0.22.0 audit fix: projection errors are **propagated** instead of silently
/// returning an empty list (an empty context sent to the LLM is an API 400).
fn last_n_turns_messages(events: &[SessionEvent], n: usize) -> Result<Vec<Message>, SessionError> {
    let p = project(events)?;
    let total = p.turns.len();
    let skip_turns = total.saturating_sub(n);
    let boundary_turn = p.turns.get(skip_turns);
    match boundary_turn {
        Some(first_kept) => {
            let first_kept_id = first_kept.events.first().map(|e| e.id).unwrap_or(0);
            let mut out: Vec<Message> = Vec::new();
            for e in events {
                if e.id >= first_kept_id {
                    match &e.payload {
                        EventPayload::UserMessage { content } => {
                            out.push(Message::human(content));
                        }
                        EventPayload::AssistantMessage { content } => {
                            out.push(Message::ai(content));
                        }
                        EventPayload::Snapshot { summary, .. } => {
                            // Context metadata, not user speech (same fix as the projector).
                            out.push(Message::system(format!("[context summary] {summary}")));
                        }
                        // T2: the windowed path used to drop tool activity
                        // entirely, so a windowed replay made the model think
                        // nothing had ever been called. Same labeled context
                        // lines as `project()` (a raw tool message would need a
                        // preceding assistant `tool_calls`, which a replayed log
                        // does not have).
                        EventPayload::ToolCall { .. }
                        | EventPayload::ToolResult { .. }
                        | EventPayload::ToolEdit { .. }
                        | EventPayload::ToolUndo { .. } => {
                            if let Some(msg) = super::replay::tool_context_message(&e.payload) {
                                out.push(msg);
                            }
                        }
                        EventPayload::Metadata { .. } => {}
                    }
                }
            }
            Ok(out)
        }
        None => Ok(p.messages),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::checkpoint::{FileCheckpoint, NoopCheckpoint, SessionCheckpoint};
    use crate::events::replay::assert_no_orphan_tool_results;
    use crate::events::store::MemoryEventStore;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, LLMResult, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_memory::MemoryError;
    use std::pin::Pin;

    fn manager() -> EventSessionManager {
        EventSessionManager::new(Arc::new(MemoryEventStore::new()))
    }

    #[derive(Debug)]
    struct MockError(String);
    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl std::error::Error for MockError {}

    #[derive(Clone)]
    struct MockLlm {
        response: String,
        received: Arc<Mutex<Vec<Vec<Message>>>>,
    }
    impl MockLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.into(),
                received: Arc::new(Mutex::new(Vec::new())),
            }
        }
        async fn received(&self) -> Vec<Vec<Message>> {
            self.received.lock().await.clone()
        }
    }
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockLlm {
        fn model_name(&self) -> &str {
            "mock"
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
        }
        fn with_temperature(self, _t: f32) -> Self
        where
            Self: Sized,
        {
            self
        }
        fn with_max_tokens(self, _m: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }
    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockLlm {
        type Error = MockError;
        async fn invoke(
            &self,
            input: Vec<Message>,
            _c: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.received.lock().await.push(input);
            Ok(LLMResult {
                content: self.response.clone(),
                model: "mock".into(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }
    #[async_trait]
    impl BaseChatModel for MockLlm {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }
        async fn stream_chat(
            &self,
            _m: Vec<Message>,
            _c: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream not exercised here")
        }
    }

    /// LLM that always fails — used to prove E2/M-s1: a failed `chat()` must
    /// not leave an orphan user message polluting the event log.
    struct FailingLlm;
    impl FailingLlm {
        fn new() -> Self {
            Self
        }
    }
    impl BaseLanguageModel<Vec<Message>, LLMResult> for FailingLlm {
        fn model_name(&self) -> &str {
            "failing-mock"
        }
        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
        }
        fn with_temperature(self, _t: f32) -> Self
        where
            Self: Sized,
        {
            self
        }
        fn with_max_tokens(self, _m: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }
    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for FailingLlm {
        type Error = MockError;
        async fn invoke(
            &self,
            _input: Vec<Message>,
            _c: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockError("boom".into()))
        }
    }
    #[async_trait]
    impl BaseChatModel for FailingLlm {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }
        async fn stream_chat(
            &self,
            _m: Vec<Message>,
            _c: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream not exercised here")
        }
    }

    #[derive(Clone)]
    struct RecordingMemory {
        history: String,
        saved: Arc<Mutex<Vec<(String, String)>>>,
    }
    impl RecordingMemory {
        fn new(history: &str) -> Self {
            Self {
                history: history.into(),
                saved: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }
    #[async_trait]
    impl BaseMemory for RecordingMemory {
        fn memory_variables(&self) -> Vec<&str> {
            vec!["history"]
        }
        async fn load_memory_variables(
            &self,
            _i: &HashMap<String, String>,
        ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
            let mut vars = HashMap::new();
            vars.insert(
                "history".into(),
                serde_json::Value::String(self.history.clone()),
            );
            Ok(vars)
        }
        async fn save_context(
            &mut self,
            i: &HashMap<String, String>,
            o: &HashMap<String, String>,
        ) -> Result<(), MemoryError> {
            self.saved.lock().await.push((
                i.get("input").cloned().unwrap_or_default(),
                o.get("output").cloned().unwrap_or_default(),
            ));
            Ok(())
        }
        async fn clear(&mut self) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    /// Chat accumulates history across turns via the event log (E2 parity
    /// with the deprecated manager's accumulation test).
    #[tokio::test]
    async fn chat_accumulates_history_across_turns() {
        let llm = MockLlm::new("回复");
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        assert_eq!(received[1].len(), 3, "second turn sees accumulated history");
        assert_eq!(received[1][0].content, "第一句");
        assert_eq!(received[1][1].content, "回复");
        assert_eq!(received[1][2].content, "第二句");
    }

    /// Memory wiring: context = memory history + current message; save_context called.
    #[tokio::test]
    async fn chat_with_memory() {
        let llm = MockLlm::new("你好,我是 AI");
        let rec = Arc::new(Mutex::new(RecordingMemory::new("Human: 在吗\nAI: 在")));
        let mgr = manager().with_memory(rec.clone());
        let id = mgr.create_session().await.unwrap();

        let reply = mgr.chat(&id, &llm, "你好".to_string()).await.unwrap();
        assert_eq!(reply, "你好,我是 AI");

        let received = llm.received().await;
        assert_eq!(received[0].len(), 2);
        assert_eq!(received[0][1].content, "你好");

        let saved = rec.lock().await.saved.lock().await.clone();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0], ("你好".into(), "你好,我是 AI".into()));
    }

    /// Turn-based context window: window 1 = the previous turn verbatim
    /// (user + AI) plus the current user message — turn granularity keeps
    /// pairs intact, unlike a message-count window.
    #[tokio::test]
    async fn chat_respects_turn_window() {
        let llm = MockLlm::new("回复");
        let mgr = manager().with_max_context_turns(1);
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        assert_eq!(
            received[1].len(),
            3,
            "window 1 = previous turn verbatim + current user message"
        );
        assert_eq!(received[1][0].content, "第一句");
        assert_eq!(received[1][1].content, "回复");
        assert_eq!(received[1][2].content, "第二句");
    }

    /// E2/M-s1 regression: a failed LLM call must NOT leave an orphan
    /// user message in the log; otherwise the next `chat()` projects the
    /// unconsumed message as dangling context.
    #[tokio::test]
    async fn failed_chat_leaves_no_orphan_user_message() {
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone());
        let id = mgr.create_session().await.unwrap();

        let failing = FailingLlm::new();
        let err = mgr.chat(&id, &failing, "忘记这句".to_string()).await;
        assert!(
            matches!(err, Err(SessionError::Llm(_))),
            "expected LLM failure"
        );

        // The user message must not have been persisted: the log holds only
        // the session metadata event, no UserMessage payload.
        let events = store.read(&id, "main", None).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e.payload, EventPayload::UserMessage { .. }))
                .count(),
            0,
            "failed chat() must not persist the user message"
        );

        // And a subsequent successful chat() must not surface the dropped text:
        // its context contains only the current user message, nothing lingering.
        let llm = MockLlm::new("回复");
        mgr.chat(&id, &llm, "这次成功".to_string()).await.unwrap();
        let received = llm.received().await;
        assert_eq!(
            received.len(),
            1,
            "the failing chat never reached a MockLlm"
        );
        assert_eq!(
            received[0].len(),
            1,
            "only the successful user message is in context"
        );
        assert_eq!(received[0][0].content, "这次成功");
    }

    /// Per-session cache eviction: `release_session_cache` frees a session's lazily-created
    /// memory + lock entries so a long-lived manager does not hold them resident forever,
    /// while the session and its stored history survive.
    #[tokio::test]
    async fn release_session_cache_evicts_lazy_state_but_keeps_session() {
        let store = Arc::new(MemoryEventStore::new());
        let mut mgr = EventSessionManager::new(store.clone());
        mgr.memory_factory = Some(Arc::new(|| {
            Arc::new(tokio::sync::Mutex::new(
                lc_memory::ConversationBufferMemory::new(),
            )) as Arc<tokio::sync::Mutex<dyn lc_memory::BaseMemory>>
        }));
        let id = mgr.create_session().await.unwrap();

        // First use lazily creates the per-session memory + striped lock entries.
        assert!(mgr.session_memory(&id).await.is_some());
        mgr.session_lock(&id).await;
        assert_eq!(
            mgr.session_memories.lock().await.len(),
            1,
            "memory cache populated on first use"
        );
        assert_eq!(
            mgr.locks.lock().await.len(),
            1,
            "striped lock created on first use"
        );

        // Eviction releases the resident entries and is idempotent.
        mgr.release_session_cache(&id).await;
        mgr.release_session_cache(&id).await;
        assert_eq!(
            mgr.session_memories.lock().await.len(),
            0,
            "memory cache released"
        );
        assert_eq!(mgr.locks.lock().await.len(), 0, "lock cache released");

        // The session survives — its stored history is still intact — and the next
        // use lazily recreates a fresh memory instance.
        assert!(
            mgr.history(&id).await.is_ok(),
            "history survives cache eviction"
        );
        assert!(mgr.session_memory(&id).await.is_some());
        assert_eq!(
            mgr.session_memories.lock().await.len(),
            1,
            "recreated lazily on next use"
        );
    }

    /// Auto-compaction: over the limit, a Snapshot is appended, the windowed
    /// context shrinks, and the summary preserves the dropped turns.
    #[tokio::test]
    async fn auto_compaction_appends_snapshot() {
        let llm = MockLlm::new("回复");
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone())
            .with_auto_compaction(AutoCompaction::new(2).unwrap());
        let id = mgr.create_session().await.unwrap();

        for t in 0..4 {
            mgr.chat(&id, &llm, format!("问题{t}")).await.unwrap();
        }

        let events = store.read(&id, "main", None).await.unwrap();
        assert!(events.iter().any(|e| matches!(
            e.payload,
            EventPayload::Snapshot { compacted_turns, .. } if compacted_turns >= 1
        )));
        // Snapshot events do not create orphan tool results (invariant holds).
        assert_eq!(assert_no_orphan_tool_results(&events).unwrap(), 0);
    }

    /// H-bounded: after auto-compaction the summarized prefix is physically
    /// removed, so the event log stays bounded (not just context-bounded) —
    /// it holds only the kept recent turns plus snapshots, not every turn ever
    /// chatted (§3.1).
    #[tokio::test]
    async fn auto_compaction_prunes_compacted_prefix() {
        let llm = MockLlm::new("回复");
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone())
            .with_auto_compaction(AutoCompaction::new(2).unwrap());
        let id = mgr.create_session().await.unwrap();

        // 8 turns ≥ max_turns=2 triggers compaction on every chat past the 2nd.
        for t in 0..8 {
            mgr.chat(&id, &llm, format!("问题{t}")).await.unwrap();
        }

        let events = store.read(&id, "main", None).await.unwrap();
        // Raw storage would be 8 turns * 2 events = 16+; a bounded log is well
        // under that (2 kept turns + snapshots).
        assert!(
            events.len() <= 8,
            "event log must stay bounded after compaction, got {} events",
            events.len()
        );
        // The very oldest content survives inside a snapshot summary...
        assert!(events.iter().any(|e| matches!(
            &e.payload,
            EventPayload::Snapshot { summary, .. } if summary.contains("问题0")
        )));
        // ...but its raw events are gone (turn 0's user message was pruned).
        assert!(!events.iter().any(|e| matches!(
            &e.payload,
            EventPayload::UserMessage { content } if content == "问题0"
        )));
        // Ids remain append-order monotonic, so idempotent appends still hold.
        let ids: Vec<u64> = events.iter().map(|e| e.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "ids stay monotonic after prefix pruning");
    }

    /// Memory whose save_context fails — proves §3.2: a failed memory save must
    /// NOT commit a log turn, otherwise a caller retry would append a duplicate.
    #[derive(Clone)]
    struct FailingMemory;
    #[async_trait]
    impl BaseMemory for FailingMemory {
        fn memory_variables(&self) -> Vec<&str> {
            vec!["history"]
        }
        async fn load_memory_variables(
            &self,
            _i: &HashMap<String, String>,
        ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
            Ok(HashMap::new())
        }
        async fn save_context(
            &mut self,
            _i: &HashMap<String, String>,
            _o: &HashMap<String, String>,
        ) -> Result<(), MemoryError> {
            Err(MemoryError::SaveError("boom".into()))
        }
        async fn clear(&mut self) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    /// §3.2 regression: when save_context fails the event log must not contain
    /// the turn — so a retry does not recreate a duplicate conversation.
    #[tokio::test]
    async fn failed_save_context_commits_no_log_turn() {
        let llm = MockLlm::new("回复");
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone())
            .with_memory(Arc::new(Mutex::new(FailingMemory)));
        let id = mgr.create_session().await.unwrap();

        let err = mgr.chat(&id, &llm, "会被丢弃吗".to_string()).await;
        assert!(
            matches!(err, Err(SessionError::Memory(_))),
            "a failing save_context must surface as a Memory error"
        );

        // No user/assistant payload was appended for the failed turn.
        let events = store.read(&id, "main", None).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(
                    e.payload,
                    EventPayload::UserMessage { .. } | EventPayload::AssistantMessage { .. }
                ))
                .count(),
            0,
            "failed save_context must not commit a log turn"
        );
    }

    /// Tool recording lands in the log as a paired call/result in one turn.
    #[tokio::test]
    async fn record_tool_use_pairs() {
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        mgr.record_tool_use(&id, "search", serde_json::json!({"q": "x"}), "c1", "found")
            .await
            .unwrap();

        let events = mgr.store.read(&id, "main", None).await.unwrap();
        assert_eq!(assert_no_orphan_tool_results(&events).unwrap(), 0);
    }

    /// L12: recording a tool call into a session that was never created must error
    /// (NotFound), matching chat()/history() — it must not silently create an orphan
    /// tool log for a nonexistent session.
    #[tokio::test]
    async fn record_tool_use_nonexistent_session_errors() {
        let mgr = manager();
        let err = mgr
            .record_tool_use(
                "nope",
                "search",
                serde_json::json!({"q": "x"}),
                "c1",
                "found",
            )
            .await
            .unwrap_err();
        assert!(matches!(err, SessionError::NotFound(_)));
    }

    /// Forked sessions diverge independently (manager pinned to the branch).
    #[tokio::test]
    async fn forked_manager_diverges() {
        let llm = MockLlm::new("回复");
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &llm, "主干".to_string()).await.unwrap();

        let forked = mgr.fork_session(&id, "experiment", None).await.unwrap();
        forked.chat(&id, &llm, "分支".to_string()).await.unwrap();

        assert_eq!(mgr.history(&id).await.unwrap().len(), 2);
        assert_eq!(forked.history(&id).await.unwrap().len(), 4);
    }

    /// chat on a nonexistent session errors like the deprecated manager.
    #[tokio::test]
    async fn chat_nonexistent_errors() {
        let mgr = manager();
        let err = mgr
            .chat("nope", &MockLlm::new("x"), "hi".to_string())
            .await
            .unwrap_err();
        assert!(matches!(err, SessionError::NotFound(_)));
    }

    /// H-M1: a memory *factory* gives each session its own instance — distinct
    /// sessions never share (and thus never contaminate) each other's memory.
    #[tokio::test]
    async fn memory_factory_isolates_sessions() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let llm = MockLlm::new("ok");
        let creations = Arc::new(AtomicUsize::new(0));
        let creations2 = creations.clone();
        let mgr = manager().with_memory_factory(move || {
            creations2.fetch_add(1, Ordering::SeqCst);
            Arc::new(Mutex::new(RecordingMemory::new("")))
        });

        let a = mgr.create_session().await.unwrap();
        let b = mgr.create_session().await.unwrap();

        mgr.chat(&a, &llm, "A问题".to_string()).await.unwrap();
        mgr.chat(&b, &llm, "B问题".to_string()).await.unwrap();
        // Two distinct sessions -> two distinct memory instances.
        assert_eq!(creations.load(Ordering::SeqCst), 2);

        // A repeat turn on session A reuses A's own instance (not a third one).
        mgr.chat(&a, &llm, "A再问".to_string()).await.unwrap();
        assert_eq!(creations.load(Ordering::SeqCst), 2);
    }

    /// Checkpoint placeholder wiring compiles and is a no-op.
    #[tokio::test]
    async fn checkpoint_noop_wiring() {
        let cp = NoopCheckpoint;
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &MockLlm::new("r"), "hi".to_string())
            .await
            .unwrap();
        let events = mgr.store.read(&id, "main", None).await.unwrap();
        let p = project(&events).unwrap();
        cp.save(&id, "main", p.last_id, &p).await.unwrap();
        assert!(cp.latest(&id, "main").await.unwrap().is_none());
    }

    // ---- T2: tool activity in the windowed context -------------------------

    /// T2: a windowed context keeps tool activity. The window used to drop tool
    /// events, so a session whose agent had run `search` looked to the model as
    /// if nothing had ever been called.
    #[tokio::test]
    async fn windowed_context_keeps_tool_activity() {
        let llm = MockLlm::new("回复");
        let mgr = manager().with_max_context_turns(1);
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.record_tool_use(
            &id,
            "search_docs",
            serde_json::json!({"q": "x"}),
            "c1",
            "found it",
        )
        .await
        .unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();

        let received = llm.received().await;
        let window = received.last().unwrap();
        let contents: Vec<&str> = window.iter().map(|m| m.content.as_str()).collect();
        assert!(
            contents.contains(&"[tool call] search_docs"),
            "the tool name must survive the window, got {contents:?}"
        );
        assert!(
            contents.iter().any(|c| c.starts_with("[tool result]")),
            "the observation must survive the window, got {contents:?}"
        );
    }

    // ---- T2: idle-session GC ----------------------------------------------

    /// Stamps an event with an explicit timestamp.
    fn at(mut e: SessionEvent, ts: i64) -> SessionEvent {
        e.ts = ts;
        e
    }

    /// Seeds `turns` completed turns into `store`, all timestamped `ts` ago.
    async fn seed_old_turns(
        store: &MemoryEventStore,
        id: &str,
        turns: u64,
        ts: i64,
        first_id: u64,
    ) {
        for t in 0..turns {
            store
                .append(&at(
                    SessionEvent::now(
                        first_id + t * 2,
                        id,
                        t,
                        "main",
                        EventPayload::UserMessage {
                            content: format!("问题{t}"),
                        },
                    ),
                    ts,
                ))
                .await
                .unwrap();
            store
                .append(&at(
                    SessionEvent::now(
                        first_id + t * 2 + 1,
                        id,
                        t,
                        "main",
                        EventPayload::AssistantMessage {
                            content: format!("回答{t}"),
                        },
                    ),
                    ts,
                ))
                .await
                .unwrap();
        }
    }

    /// T2: `gc()` reclaims an *idle* session — one whose turn count never
    /// crosses the count threshold, so `chat()`-driven compaction never fires
    /// and its events would otherwise stay in the log forever.
    #[tokio::test]
    async fn gc_reclaims_an_idle_session_by_age() {
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone()).with_auto_compaction(
            AutoCompaction::new(8)
                .unwrap()
                .with_max_age(Duration::from_secs(3600)),
        );
        let id = mgr.create_session().await.unwrap();
        let long_ago = chrono::Utc::now().timestamp_millis() - 10 * 24 * 3600 * 1000;
        seed_old_turns(&store, &id, 3, long_ago, 2).await;

        // 3 turns is far below max_turns = 8, so nothing was compacted yet...
        let before = store.read(&id, "main", None).await.unwrap();
        assert!(!before
            .iter()
            .any(|e| matches!(e.payload, EventPayload::Snapshot { .. })));

        // ...but the age bound makes every turn eligible.
        assert!(
            mgr.gc(&id).await.unwrap(),
            "gc must compact an idle session"
        );

        let after = store.read(&id, "main", None).await.unwrap();
        assert!(
            after
                .iter()
                .any(|e| matches!(&e.payload, EventPayload::Snapshot { summary, .. } if summary.contains("问题0"))),
            "the dropped turns must be preserved in the snapshot summary"
        );
        assert!(
            !after.iter().any(|e| matches!(&e.payload, EventPayload::UserMessage { content } if content == "问题0")),
            "the raw events of the summarized prefix must be pruned"
        );
        assert!(after.len() < before.len(), "the log must actually shrink");
    }

    /// Without an age bound, `gc()` leaves a low-turn session alone — the
    /// historical count-only behavior is unchanged.
    #[tokio::test]
    async fn gc_without_age_bound_keeps_a_low_turn_session() {
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone())
            .with_auto_compaction(AutoCompaction::new(8).unwrap());
        let id = mgr.create_session().await.unwrap();
        let long_ago = chrono::Utc::now().timestamp_millis() - 10 * 24 * 3600 * 1000;
        seed_old_turns(&store, &id, 3, long_ago, 2).await;

        assert!(
            !mgr.gc(&id).await.unwrap(),
            "nothing to compact under max_turns"
        );
        // A count-triggered gc over the threshold does compact.
        let mgr = EventSessionManager::new(store.clone())
            .with_auto_compaction(AutoCompaction::new(1).unwrap());
        assert!(mgr.gc(&id).await.unwrap());
    }

    /// `gc()` on a session that does not exist errors rather than creating one,
    /// and a manager with compaction disabled is a no-op.
    #[tokio::test]
    async fn gc_edge_cases() {
        let mgr = manager();
        assert!(matches!(
            mgr.gc("nope").await.unwrap_err(),
            SessionError::NotFound(_)
        ));

        let id = mgr.create_session().await.unwrap();
        // `create_session` only writes metadata; nothing to compact.
        assert!(!mgr.gc(&id).await.unwrap());

        // A manager with compaction disabled never reclaims anything.
        let mut disabled = EventSessionManager::new(Arc::new(MemoryEventStore::new()));
        disabled.auto_compaction = None;
        let id2 = disabled.create_session().await.unwrap();
        assert!(!disabled.gc(&id2).await.unwrap());
    }

    // ---- T2: durable checkpoint + recovery --------------------------------

    fn scratch_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lc-sessions-mgr-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// T2 end-to-end: a snapshot written before a restart lets the next process
    /// recover the session by replaying only the tail — and the result is
    /// identical to projecting the whole log.
    #[tokio::test]
    async fn recover_session_uses_the_checkpoint_and_merges_the_tail() {
        let dir = scratch_dir();
        let store = Arc::new(MemoryEventStore::new());
        let llm = MockLlm::new("回复");

        let id = {
            let mgr = EventSessionManager::new(store.clone())
                .with_checkpoint(Arc::new(FileCheckpoint::new(&dir)));
            let id = mgr.create_session().await.unwrap();
            mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
            mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();
            assert!(
                mgr.save_checkpoint(&id).await.unwrap(),
                "a checkpoint is written"
            );
            // Two more turns land *after* the snapshot.
            mgr.chat(&id, &llm, "第三句".to_string()).await.unwrap();
            mgr.chat(&id, &llm, "第四句".to_string()).await.unwrap();
            id
        };

        // Restart: a fresh manager over the same store + checkpoint directory.
        let cp = Arc::new(FileCheckpoint::new(&dir));
        let mgr = EventSessionManager::new(store.clone()).with_checkpoint(cp.clone());
        let events = store.read(&id, "main", None).await.unwrap();
        let full = project(&events).unwrap();

        let (up_to_id, _) = cp.latest(&id, "main").await.unwrap().unwrap();
        assert!(
            up_to_id < full.last_id,
            "the snapshot must be behind the log, so recovery really replays a tail"
        );

        let recovered = mgr.recover_session(&id).await.unwrap();
        assert_eq!(
            recovered, full,
            "checkpoint + tail must equal the full projection"
        );
        assert_eq!(
            recovered.messages.len(),
            8,
            "4 turns = 4 user + 4 assistant messages"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Recovery without a checkpoint (or with one that has no snapshot yet) is a
    /// full replay — degraded, never wrong.
    #[tokio::test]
    async fn recover_session_falls_back_to_full_replay() {
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone());
        let llm = MockLlm::new("回复");
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &llm, "你好".to_string()).await.unwrap();

        let full = project(&store.read(&id, "main", None).await.unwrap()).unwrap();
        assert_eq!(mgr.recover_session(&id).await.unwrap(), full);

        // A checkpoint store with nothing in it behaves the same way.
        let dir = scratch_dir();
        let mgr = EventSessionManager::new(store.clone())
            .with_checkpoint(Arc::new(FileCheckpoint::new(&dir)));
        assert_eq!(mgr.recover_session(&id).await.unwrap(), full);

        assert!(matches!(
            mgr.recover_session("nope").await.unwrap_err(),
            SessionError::NotFound(_)
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stale snapshot (the log was cleared out from under it) must not be
    /// trusted as a prefix — recovery falls back to replaying the log.
    #[tokio::test]
    async fn recover_session_ignores_a_snapshot_ahead_of_the_log() {
        let dir = scratch_dir();
        let store = Arc::new(MemoryEventStore::new());
        let cp = Arc::new(FileCheckpoint::new(&dir));
        let mgr = EventSessionManager::new(store.clone()).with_checkpoint(cp.clone());
        let llm = MockLlm::new("回复");
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &llm, "你好".to_string()).await.unwrap();

        // A snapshot claiming an id the log does not reach (e.g. the log was
        // rebuilt): recovery must not silently return the stale prefix.
        let stale = project(&store.read(&id, "main", None).await.unwrap()).unwrap();
        cp.save(&id, "main", 9_999, &stale).await.unwrap();

        let full = project(&store.read(&id, "main", None).await.unwrap()).unwrap();
        assert_eq!(mgr.recover_session(&id).await.unwrap(), full);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `save_checkpoint` is a no-op (not an error) when no store is attached,
    /// and errors for a session with no events once one is.
    #[tokio::test]
    async fn save_checkpoint_edge_cases() {
        let dir = scratch_dir();
        let store = Arc::new(MemoryEventStore::new());
        let mgr = EventSessionManager::new(store.clone());
        assert!(mgr.checkpoint().is_none());

        // No store attached → `false` for any session, existing or not: there is
        // nowhere to write, so session existence is not even consulted.
        let id = mgr.create_session().await.unwrap();
        assert!(!mgr.save_checkpoint(&id).await.unwrap());
        assert!(!mgr.save_checkpoint("nope").await.unwrap());

        // With a store attached, a session with no events is NotFound.
        let mgr = EventSessionManager::new(store.clone())
            .with_checkpoint(Arc::new(FileCheckpoint::new(&dir)));
        assert!(mgr.checkpoint().is_some());
        assert!(mgr.save_checkpoint(&id).await.unwrap());
        assert!(matches!(
            mgr.save_checkpoint("nope").await.unwrap_err(),
            SessionError::NotFound(_)
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Deleting a session drops its snapshot too: otherwise the next recovery
    /// would resurrect the deleted session's history.
    #[tokio::test]
    async fn delete_session_drops_the_checkpoint() {
        let dir = scratch_dir();
        let store = Arc::new(MemoryEventStore::new());
        let cp = Arc::new(FileCheckpoint::new(&dir));
        let mgr = EventSessionManager::new(store.clone()).with_checkpoint(cp.clone());
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &MockLlm::new("r"), "hi".to_string())
            .await
            .unwrap();
        mgr.save_checkpoint(&id).await.unwrap();
        assert!(cp.latest(&id, "main").await.unwrap().is_some());

        mgr.delete_session(&id).await.unwrap();
        assert!(
            cp.latest(&id, "main").await.unwrap().is_none(),
            "a deleted session must not leave a resurrectable snapshot"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A fork inherits the checkpoint store and keys snapshots by branch, so
    /// trunk and fork never overwrite each other's snapshot.
    #[tokio::test]
    async fn fork_checkpoints_are_branch_isolated() {
        let dir = scratch_dir();
        let store = Arc::new(MemoryEventStore::new());
        let cp = Arc::new(FileCheckpoint::new(&dir));
        let mgr = EventSessionManager::new(store.clone()).with_checkpoint(cp.clone());
        let llm = MockLlm::new("回复");
        let id = mgr.create_session().await.unwrap();
        mgr.chat(&id, &llm, "主干".to_string()).await.unwrap();

        let forked = mgr.fork_session(&id, "experiment", None).await.unwrap();
        assert!(forked.checkpoint().is_some(), "the fork inherits the store");
        forked.chat(&id, &llm, "分支".to_string()).await.unwrap();

        mgr.save_checkpoint(&id).await.unwrap();
        forked.save_checkpoint(&id).await.unwrap();

        let (main_id, main_p) = cp.latest(&id, "main").await.unwrap().unwrap();
        let (fork_id, fork_p) = cp.latest(&id, "experiment").await.unwrap().unwrap();
        assert_eq!(main_p.messages.len(), 2, "trunk snapshot is the trunk's");
        assert_eq!(fork_p.messages.len(), 4, "fork snapshot is the fork's");
        assert!(fork_id > main_id);

        std::fs::remove_dir_all(&dir).ok();
    }
}
