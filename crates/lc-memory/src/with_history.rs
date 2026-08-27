// lc-memory/src/with_history.rs
//! LCEL-composable "LLM wrapper with memory".
//!
//! Packages the "read memory → compose user input → call LLM → write back memory"
//! composition into a single `Runnable<String, LLMResult>`, so "LLM + memory" can
//! go straight into an LCEL pipeline (pipe into chains, batch, stream, etc.) without
//! hand-writing memory glue in business code.
//!
//! # Two memory sources
//!
//! - `new(llm, memory)` — injects a concrete memory object (single session; original behavior).
//! - `with_session_history(llm, factory)` — injects a **session callback** (mirrors Python's
//!   `RunnableWithMessageHistory(llm, get_session_history)`): each call picks a slot from
//!   `config.configurable["session_id"]`; the same session shares history, different sessions
//!   never cross; a missing `session_id` returns `LcelError::Chain`.
//!
//! # Semantics
//!
//! `invoke(user_input)` runs in order:
//! 1. select the memory by mode (Shared takes it directly; Sessions picks/creates a slot by session_id);
//! 2. read history from memory, convert to messages (`memory_variables_to_messages`);
//! 3. append the user input as a Human message;
//! 4. hand to `llm.chat` (optional `RunnableConfig` passthrough);
//! 5. write "user input / model answer" back to memory;
//! 6. return the full `LLMResult`.
//!
//! LLM errors enter the pipeline error via `L::Error: Into<LcelError>`; memory read/write
//! errors converge to `LcelError::Chain`.
//!
//! # Generics
//!
//! `L` is any model implementing `BaseChatModel` (native Provider / `LLMClient` both work),
//! as long as its error type can convert into `LcelError` (`LLMClient` satisfies it naturally;
//! native Providers see `From<...> for LcelError` in lc-providers). Memory is held as a trait
//! object; any `BaseMemory` (Buffer / Window / Summary / SummaryBuffer, etc.) works.

use crate::base::{memory_variables_to_messages, BaseMemory};
use crate::buffer::ConversationBufferMemory;
use async_trait::async_trait;
use lc_core::language_models::{BaseChatModel, LLMResult};
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use lc_schema::Message;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Memory handle: a shared mutable reference to any `BaseMemory`.
pub type SharedMemory = Arc<Mutex<Box<dyn BaseMemory>>>;

/// Default cap for the session cache: prevents runaway/malicious session_id growth from filling memory (M2a).
const DEFAULT_MAX_SESSIONS: usize = 100;

/// LLM wrapper with memory, participating in LCEL composition as a single Runnable.
pub struct RunnableWithMessageHistory<L> {
    llm: Arc<L>,
    mode: HistoryMode,
}

/// Memory source: a shared single slot (SHARED) or per-session slots (SESSIONS).
enum HistoryMode {
    /// A single memory object injected at construction; all calls share it.
    Shared(SharedMemory),
    /// Per-slot by `configurable.session_id`: the factory builds new slots, the cache reuses existing ones.
    Sessions {
        factory: Arc<dyn Fn(&str) -> SharedMemory + Send + Sync>,
        /// Slot cache: the same session reuses the same memory (its lock also serializes concurrent invokes on the same slot).
        cache: Mutex<SessionCache>,
        /// Cache cap: evicts the oldest session when exceeded, preventing a memory DoS from unbounded session_id growth (M2a).
        max_sessions: usize,
        /// Placeholder memory used only by the `memory()` accessor (in Sessions mode the real slot is not unique).
        default: SharedMemory,
    },
}

/// Session slot cache: `slots` looks up by session_id, `order` records insertion order for cap-based eviction.
struct SessionCache {
    slots: HashMap<String, SharedMemory>,
    order: VecDeque<String>,
}

impl<L> RunnableWithMessageHistory<L> {
    /// Builds the wrapper from an LLM + a single memory object (all calls share this memory).
    pub fn new(llm: L, memory: impl BaseMemory + 'static) -> Self {
        Self {
            llm: Arc::new(llm),
            mode: HistoryMode::Shared(Arc::new(Mutex::new(Box::new(memory)))),
        }
    }

    /// Builds the wrapper from an LLM + a session callback (mirrors Python's
    /// `RunnableWithMessageHistory(llm, get_session_history)`).
    ///
    /// `factory(session_id)` builds a memory object for one session slot; each call selects the
    /// slot by `config.configurable["session_id"]`, and the same session reuses its built slot.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipe = RunnableWithMessageHistory::with_session_history(llm, |session_id| {
    ///     Arc::new(Mutex::new(Box::new(
    ///         ConversationBufferMemory::new().with_return_messages(true),
    ///     ) as Box<dyn BaseMemory>))
    /// });
    /// let cfg = RunnableConfig::new().with_configurable("session_id", json!("s1"));
    /// pipe.invoke("我叫小明".into(), Some(cfg)).await?;
    /// ```
    pub fn with_session_history<F>(llm: L, factory: F) -> Self
    where
        F: Fn(&str) -> SharedMemory + Send + Sync + 'static,
    {
        Self {
            llm: Arc::new(llm),
            mode: HistoryMode::Sessions {
                factory: Arc::new(factory),
                cache: Mutex::new(SessionCache {
                    slots: HashMap::new(),
                    order: VecDeque::new(),
                }),
                max_sessions: DEFAULT_MAX_SESSIONS,
                default: Arc::new(Mutex::new(Box::new(ConversationBufferMemory::new()))),
            },
        }
    }

    /// Sets the session-cache cap (Sessions mode). When exceeded, a new session evicts the
    /// oldest slot, preventing a memory DoS from unbounded session_id growth (M2a).
    /// Defaults to `DEFAULT_MAX_SESSIONS`.
    pub fn with_max_sessions(mut self, max: usize) -> Self {
        if let HistoryMode::Sessions { max_sessions, .. } = &mut self.mode {
            *max_sessions = max.max(1);
        }
        self
    }

    /// Exposes the internal memory handle for reading saved history (debugging, display,
    /// verifying write-back, etc.).
    ///
    /// In Sessions mode the slot is not unique, so a placeholder memory is returned (it does
    /// not participate in pipeline reads/writes); to inspect real history, read from the
    /// pipeline internals or the business-side memory object.
    pub fn memory(&self) -> SharedMemory {
        match &self.mode {
            HistoryMode::Shared(m) => m.clone(),
            HistoryMode::Sessions { default, .. } => default.clone(),
        }
    }

    /// Selects the memory slot this call will use, by mode.
    async fn select_memory(
        &self,
        config: &Option<RunnableConfig>,
    ) -> Result<SharedMemory, LcelError> {
        match &self.mode {
            HistoryMode::Shared(m) => Ok(m.clone()),
            HistoryMode::Sessions {
                factory,
                cache,
                max_sessions,
                ..
            } => {
                let session_id = config
                    .as_ref()
                    .and_then(|c| c.configurable_value("session_id"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        LcelError::Chain(
                            "RunnableWithMessageHistory (session mode) is missing configurable.session_id"
                                .to_string(),
                        )
                    })?;
                let mut cache = cache.lock().await;
                if let Some(memory) = cache.slots.get(session_id) {
                    return Ok(memory.clone());
                }
                // M2a: the cache has a cap — evict the oldest session first, then build the new slot.
                if cache.slots.len() >= *max_sessions {
                    if let Some(oldest) = cache.order.pop_front() {
                        cache.slots.remove(&oldest);
                    }
                }
                let memory = factory(session_id);
                cache.slots.insert(session_id.to_string(), memory.clone());
                cache.order.push_back(session_id.to_string());
                Ok(memory)
            }
        }
    }
}

#[async_trait]
impl<L> Runnable<String, LLMResult> for RunnableWithMessageHistory<L>
where
    L: BaseChatModel + 'static,
    L::Error: Into<LcelError>,
{
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        config: Option<RunnableConfig>,
    ) -> Result<LLMResult, LcelError> {
        // 0. select the memory slot (Sessions mode reads configurable.session_id)
        let memory = self.select_memory(&config).await?;

        // M2b: hold the lock across the whole "read memory → call LLM → write back", serializing
        // concurrent invokes on the same memory slot. The old implementation released the lock
        // right after reading, so two concurrent invokes both read stale history and overwrote
        // each other's write-back, losing history. The cost is that same-slot calls serialize,
        // but memory-backed conversation should be serial anyway.
        let mut memory = memory.lock().await;

        // 1. read memory → convert to messages
        let mut messages = {
            let vars = memory
                .load_memory_variables(&HashMap::new())
                .await
                .map_err(|e| LcelError::Chain(format!("load memory: {e}")))?;
            memory_variables_to_messages(&vars)
        };

        // 2. append the user input
        messages.push(Message::human(&input));

        // 3. call the LLM (holding the lock: same-slot calls serialize, avoiding concurrent history loss)
        let result = self.llm.chat(messages, config).await.map_err(Into::into)?;

        // 4. write back to memory: on failure the model answer is not discarded (otherwise the
        //    caller retrying on Err would re-invoke the LLM); log a warn to expose the memory-layer degradation
        let inputs = HashMap::from([("input".to_string(), input)]);
        let outputs = HashMap::from([("output".to_string(), result.content.clone())]);
        if let Err(e) = memory.save_context(&inputs, &outputs).await {
            log::warn!("memory save failed (model answer still returned): {e}");
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::Stream;
    use lc_core::language_models::{BaseChatModel, BaseLanguageModel, StreamChunk};
    use lc_core::runnables::RunnableConfig;
    use lc_schema::MessageType;
    use serde_json::json;
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;

    /// Test LLM: records every received message and wraps the last user message as the reply.
    struct TestChatModel {
        seen: Arc<StdMutex<Vec<Vec<Message>>>>,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for TestChatModel {
        type Error = LcelError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.seen.lock().unwrap().push(input.clone());
            let last = input.last().map(|m| m.content.clone()).unwrap_or_default();
            Ok(LLMResult {
                content: format!("reply to: {last}"),
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for TestChatModel {
        fn model_name(&self) -> &str {
            "test-llm"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for TestChatModel {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LcelError>> + Send>>, LcelError>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    /// Session callback: builds an empty Buffer memory per session (return_messages = true).
    fn session_factory(_session_id: &str) -> SharedMemory {
        Arc::new(Mutex::new(
            Box::new(ConversationBufferMemory::new().with_return_messages(true))
                as Box<dyn BaseMemory>,
        ))
    }

    /// Read memory → LLM → write back: on the second call, the LLM should see the first turn's full conversation.
    #[tokio::test]
    async fn reads_memory_writes_back_round_trip() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        // return_messages = true: history returns as a message array, making per-turn message composition easy to assert
        let memory = ConversationBufferMemory::new().with_return_messages(true);
        let pipe = RunnableWithMessageHistory::new(llm, memory);

        // Act turn 1: no history
        let r1 = pipe.invoke("我叫什么名字".to_string(), None).await.unwrap();
        assert_eq!(r1.content, "reply to: 我叫什么名字");

        // Act turn 2: history should have been written back
        let r2 = pipe.invoke("再问一次".to_string(), None).await.unwrap();
        assert_eq!(r2.content, "reply to: 再问一次");

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2, "model should be called twice");
        // Turn 1: only the user message
        assert_eq!(calls[0].len(), 1);
        assert_eq!(calls[0][0].content, "我叫什么名字");
        // Turn 2: user + ai + new user (memory written back)
        assert_eq!(calls[1].len(), 3);
        assert_eq!(calls[1][0].content, "我叫什么名字");
        assert!(matches!(calls[1][0].message_type, MessageType::Human));
        assert!(matches!(calls[1][1].message_type, MessageType::AI));
        assert_eq!(calls[1][2].content, "再问一次");
    }

    /// Memory write-back persists inside the wrapper: a new wrapper reusing the same memory type still has the history.
    #[tokio::test]
    async fn memory_accumulates_across_invocations() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let memory = ConversationBufferMemory::new().with_return_messages(true);
        let pipe = RunnableWithMessageHistory::new(llm, memory);

        // Act three consecutive turns
        for turn in ["你好", "你在吗", "再见"] {
            pipe.invoke(turn.to_string(), None).await.unwrap();
        }

        // Assert turn 3 should see the full four messages of turns 1-2
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[2].len(), 5); // 2 turns * 2 messages + the current user message
        assert_eq!(calls[2][4].content, "再见");
    }

    /// Session mode: two calls with the same session_id share history.
    #[tokio::test]
    async fn session_history_same_session_shares_memory() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);
        let cfg = RunnableConfig::new().with_configurable("session_id", json!("s1"));

        // Act two turns on the same session
        let r1 = pipe
            .invoke("我叫什么名字".to_string(), Some(cfg.clone()))
            .await
            .unwrap();
        assert_eq!(r1.content, "reply to: 我叫什么名字");
        let r2 = pipe
            .invoke("再问一次".to_string(), Some(cfg))
            .await
            .unwrap();
        assert_eq!(r2.content, "reply to: 再问一次");

        // Assert turn 2 sees turn 1's full conversation (user + ai + user)
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].len(), 3);
        assert_eq!(calls[1][0].content, "我叫什么名字");
        assert!(matches!(calls[1][1].message_type, MessageType::AI));
    }

    /// Session mode: different session_ids never cross-contaminate.
    #[tokio::test]
    async fn session_history_different_sessions_isolated() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);
        let cfg_s1 = RunnableConfig::new().with_configurable("session_id", json!("s1"));
        let cfg_s2 = RunnableConfig::new().with_configurable("session_id", json!("s2"));

        // Act s1 two turns + s2 one turn
        pipe.invoke("我是 s1".to_string(), Some(cfg_s1.clone()))
            .await
            .unwrap();
        pipe.invoke("还在 s1".to_string(), Some(cfg_s1))
            .await
            .unwrap();
        pipe.invoke("我是 s2".to_string(), Some(cfg_s2))
            .await
            .unwrap();

        // Assert s2's first turn has no history (1 message); s1's second turn has history (3 messages)
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[1].len(), 3, "s1 second turn should see history");
        assert_eq!(calls[2].len(), 1, "s2 first turn should have no history");
    }

    /// Session mode: missing configurable.session_id → LcelError::Chain.
    #[tokio::test]
    async fn session_history_missing_session_id_errors() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory);

        // Act no config / no session_id
        let err = pipe.invoke("你好".to_string(), None).await.unwrap_err();
        assert!(matches!(err, LcelError::Chain(_)));

        let cfg_no_sid = RunnableConfig::new();
        let err = pipe
            .invoke("你好".to_string(), Some(cfg_no_sid))
            .await
            .unwrap_err();
        assert!(matches!(err, LcelError::Chain(_)));
    }

    /// Blockable test LLM: the first chat notifies `entered` and blocks on `release`, used to
    /// create a deterministic "lock held, model in-flight" window in concurrency tests (M2b).
    struct BlockingChatModel {
        seen: Arc<StdMutex<Vec<Vec<Message>>>>,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        blocked: Arc<StdMutex<bool>>,
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for BlockingChatModel {
        type Error = LcelError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.seen.lock().unwrap().push(input.clone());
            let should_block = {
                let mut blocked = self.blocked.lock().unwrap();
                if !*blocked {
                    *blocked = true;
                    true
                } else {
                    false
                }
            };
            self.entered.notify_one();
            if should_block {
                self.release.notified().await;
            }
            let last = input.last().map(|m| m.content.clone()).unwrap_or_default();
            Ok(LLMResult {
                content: format!("reply to: {last}"),
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for BlockingChatModel {
        fn model_name(&self) -> &str {
            "blocking-test-llm"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len()
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for BlockingChatModel {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, LcelError> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LcelError>> + Send>>, LcelError>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    /// The session cache has a cap: the oldest session is evicted when exceeded, preventing a memory DoS from unbounded session_id growth (M2a).
    #[tokio::test]
    async fn session_cache_evicts_oldest_when_over_capacity() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let llm = TestChatModel { seen: seen.clone() };
        let pipe = RunnableWithMessageHistory::with_session_history(llm, session_factory)
            .with_max_sessions(2);

        let cfg_s1 = RunnableConfig::new().with_configurable("session_id", json!("s1"));
        let cfg_s2 = RunnableConfig::new().with_configurable("session_id", json!("s2"));
        let cfg_s3 = RunnableConfig::new().with_configurable("session_id", json!("s3"));

        // Act s1 and s2 fill the cache; s3 triggers eviction of the oldest, s1.
        pipe.invoke("s1-turn1".to_string(), Some(cfg_s1.clone()))
            .await
            .unwrap();
        pipe.invoke("s2-turn1".to_string(), Some(cfg_s2))
            .await
            .unwrap();
        pipe.invoke("s3-turn1".to_string(), Some(cfg_s3))
            .await
            .unwrap();
        // s1 was evicted → re-entering s1 starts a fresh session.
        pipe.invoke("s1-turn2".to_string(), Some(cfg_s1))
            .await
            .unwrap();

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 4);
        assert_eq!(
            calls[3].len(),
            1,
            "M2a: re-entering s1 after eviction should be a fresh session (no history)"
        );
    }

    /// Concurrent invokes on the same memory slot must serialize: hold the lock across the whole
    /// read→LLM→write, so concurrent calls cannot both read stale history and overwrite each
    /// other's write-back, losing context (M2b).
    #[tokio::test]
    async fn concurrent_same_session_invokes_do_not_lose_history() {
        // Arrange
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let blocked = Arc::new(StdMutex::new(false));
        let llm = BlockingChatModel {
            seen: seen.clone(),
            entered: entered.clone(),
            release: release.clone(),
            blocked,
        };
        let pipe = Arc::new(RunnableWithMessageHistory::new(
            llm,
            ConversationBufferMemory::new().with_return_messages(true),
        ));

        // Act first invoke: blocks after entering the model call — the memory lock must be held meanwhile.
        let p1 = pipe.clone();
        let h1 = tokio::spawn(async move { p1.invoke("第一轮".to_string(), None).await });
        entered.notified().await;

        // Second invoke: if read→LLM→write were not lock-held end-to-end, this call would read empty history (lost context).
        let p2 = pipe.clone();
        let h2 = tokio::spawn(async move { p2.invoke("第二轮".to_string(), None).await });

        // Release the first turn: the lock is only released after write-back, so the second turn reads the first turn's full conversation.
        release.notify_one();
        let r1 = h1.await.unwrap().unwrap();
        let r2 = h2.await.unwrap().unwrap();
        assert_eq!(r1.content, "reply to: 第一轮");
        assert_eq!(r2.content, "reply to: 第二轮");

        // Assert
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1].len(),
            3,
            "M2b: second turn should see the full first-turn conversation (user+ai+user), not empty history"
        );
        assert_eq!(calls[1][0].content, "第一轮");
        assert_eq!(calls[1][2].content, "第二轮");
    }
}
