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

use lc_core::BaseChatModel;
use lc_memory::BaseMemory;
use lc_schema::Message;
use tokio::sync::Mutex;

use super::model::{EventPayload, SessionEvent};
use super::replay::{project, replay_active_session};
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
}

impl AutoCompaction {
    /// Validates the turn floor.
    pub fn new(max_turns: usize) -> Result<Self, SessionError> {
        if max_turns == 0 {
            return Err(SessionError::StoreError(
                "AutoCompaction::max_turns must be >= 1".into(),
            ));
        }
        Ok(Self { max_turns })
    }
}

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
    memory_factory: Option<Arc<dyn Fn() -> Arc<Mutex<dyn BaseMemory>> + Send + Sync + 'static>>,
    /// Lazily-created per-session memory instances, keyed by session id.
    session_memories: Arc<Mutex<HashMap<String, Arc<Mutex<dyn BaseMemory>>>>>,

    /// Turn-based context window: with `Some(n)`, a `chat()` without memory
    /// feeds only the most recent `n` turns' messages; `None` = full history.
    max_context_turns: Option<usize>,

    /// Automatic compaction policy (`None` = never compact).
    auto_compaction: Option<AutoCompaction>,

    /// 0.22.0 C7 fix: per-session-id striped lock serializing the whole
    /// turn_count → append → LLM → append → compaction sequence of `chat()`
    /// (and `record_tool_use`). Without it two concurrent requests compute
    /// the same next event id and the EventStore's idempotent append silently
    /// drops the loser's user message. Same discipline as the deprecated
    /// manager (Q2); the outer map's Mutex guards only the map itself.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

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
            auto_compaction: None,
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

        // 1. Record the user message (opens the turn).
        self.append(
            session_id,
            turn_index,
            EventPayload::UserMessage {
                content: user_message.clone(),
            },
        )
        .await?;

        // 2. Build the LLM context. H-M1: resolve this session's own memory
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
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::Llm(e.to_string()))?
        };

        // 3. Record the reply.
        let content = response.content.clone();
        self.append(
            session_id,
            turn_index,
            EventPayload::AssistantMessage {
                content: content.clone(),
            },
        )
        .await?;

        // 4. Memory save_context (same discipline as the deprecated manager).
        if let Some(memory) = &session_memory {
            let mut mem = memory.lock().await;
            let inputs = HashMap::from([(self.memory_input_key.clone(), user_message)]);
            let outputs = HashMap::from([(self.memory_output_key.clone(), content.clone())]);
            mem.save_context(&inputs, &outputs)
                .await
                .map_err(|e| SessionError::Memory(format!("failed to save memory: {e}")))?;
        }

        // 5. Auto-compaction (deterministic snapshot, no LLM).
        if let Some(policy) = self.auto_compaction {
            let events = self.store.read(session_id, &self.branch, None).await?;
            let next_id = events.last().map(|e| e.id + 1).unwrap_or(1);
            if let Some(snapshot) =
                super::replay::compact_to_snapshot(next_id, &events, policy.max_turns)?
            {
                self.store.append(&snapshot).await?;
            }
        }

        Ok(content)
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

        let (_, turn_count) = self.turn_count(session_id).await?;
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
/// Window semantics: `n` = number of *completed* turns kept before the
/// Window semantics: `n` = number of *completed* turns kept before the
/// in-flight turn, plus the in-flight turn's own messages. So after appending
/// the current user message, window 1 = previous turn's messages + the current
/// user message (mirrors the deprecated manager's message-count window).
///
/// 0.22.0 audit fix: projection errors are **propagated** instead of silently
/// returning an empty list (an empty context sent to the LLM is an API 400).
fn last_n_turns_messages(events: &[SessionEvent], n: usize) -> Result<Vec<Message>, SessionError> {
    let p = project(events)?;
    let total = p.turns.len();
    let skip_turns = total.saturating_sub(n + 1);
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
                        _ => {}
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
    use crate::events::checkpoint::{NoopCheckpoint, SessionCheckpoint};
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
}
