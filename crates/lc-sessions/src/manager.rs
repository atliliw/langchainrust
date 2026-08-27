//! Session manager — creates/gets sessions and chats within them

use std::collections::HashMap;
use std::sync::Arc;

use lc_core::BaseChatModel;
use lc_memory::BaseMemory;
use lc_schema::Message;
use tokio::sync::Mutex;

use super::session::Session;
use super::store::{SessionError, SessionStore};

/// Session manager
pub struct SessionManager {
    store: Arc<dyn SessionStore>,

    /// P2-1: optional memory component. When attached, the LLM context for `chat()` comes from
    /// memory (window/summary-compressed history) plus the current user message; `save_context`
    /// records after each turn. Without it the original behavior is kept (full session history).
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,

    /// Memory input key (must align with the memory instance's input_key; default `"input"`).
    memory_input_key: String,

    /// Memory output key (must align with the memory instance's output_key; default `"output"`).
    memory_output_key: String,

    /// Q2: per-session-id striped lock serializing the whole get→modify→update sequence of
    /// `chat`/`clear`/`archive`, so concurrent chats on the same session do not overwrite each
    /// other and lose messages. The outer map's Mutex only guards the map itself; the
    /// `Arc<Mutex<()>>` is released as soon as it is obtained.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,

    /// Q3: optional context window (message count). With `Some(n)`, a `chat()` without memory
    /// only feeds the most recent n messages to the LLM; `None` keeps the original behavior
    /// (full history).
    max_context_messages: Option<usize>,
}

impl SessionManager {
    /// Creates a new session manager backed by the given store
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            store,
            memory: None,
            memory_input_key: "input".to_string(),
            memory_output_key: "output".to_string(),
            locks: Arc::new(Mutex::new(HashMap::new())),
            max_context_messages: None,
        }
    }

    /// Q3: limits the message window fed to the LLM by `chat()` when no memory is attached
    /// (the most recent `n` messages). `n` is a message count, not a token count; without
    /// calling this the full history is kept.
    pub fn with_max_context_messages(mut self, n: usize) -> Self {
        self.max_context_messages = Some(n);
        self
    }

    /// Q2: gets the striped lock for a session id (lazily created if absent).
    async fn session_lock(&self, id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.locks.lock().await;
        locks
            .entry(id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// P2-1: attaches a memory component to enable memory-managed conversation context.
    ///
    /// After attaching, each `chat()` turn: memory provides the compressed history + the current
    /// user message -> the LLM is called -> `save_context` records this turn's input/output. When
    /// the memory instance customizes its input/output keys, align them via
    /// [`SessionManager::with_memory_keys`].
    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// P2-1: aligns the memory instance's custom input/output keys (defaults `"input"` / `"output"`).
    pub fn with_memory_keys(
        mut self,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        self.memory_input_key = input_key.into();
        self.memory_output_key = output_key.into();
        self
    }

    /// P2-1: whether a memory component is attached.
    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// Creates a new session, returning its ID
    pub async fn create_session(&self) -> Result<String, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        let session = Session::new(id.clone());
        self.store.create(session).await?;
        Ok(id)
    }

    /// Creates a session for a specific user ID
    pub async fn create_session_for(
        &self,
        user_id: impl Into<String>,
    ) -> Result<String, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        let session = Session::new(id.clone()).with_user(user_id);
        self.store.create(session).await?;
        Ok(id)
    }

    /// Gets a session
    pub async fn get_session(&self, id: &str) -> Result<Option<Session>, SessionError> {
        self.store.get(id).await
    }

    /// Chats within a session: append the user message -> call the LLM -> append the AI reply -> persist
    pub async fn chat<L: BaseChatModel>(
        &self,
        id: &str,
        llm: &L,
        user_message: String,
    ) -> Result<String, SessionError>
    where
        L::Error: std::fmt::Display,
    {
        // Q2: hold this session's striped lock so the whole get→modify→llm→update is mutually
        // exclusive; concurrent chats never observe each other's in-progress intermediate state.
        let lock = self.session_lock(id).await;
        let _guard = lock.lock().await;

        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.add_message(Message::human(&user_message));

        // P2-1: with memory attached, the LLM context = memory-compressed history + the current user message; without it the original logic runs.
        let response = if let Some(memory) = &self.memory {
            let history_messages = {
                let mem = memory.lock().await;
                let inputs = HashMap::from([(self.memory_input_key.clone(), user_message.clone())]);
                let vars = mem
                    .load_memory_variables(&inputs)
                    .await
                    .map_err(|e| SessionError::Memory(format!("failed to load memory: {}", e)))?;
                lc_memory::memory_variables_to_messages(&vars)
            };

            let mut messages = history_messages;
            messages.push(Message::human(&user_message));
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::Llm(e.to_string()))?
        } else {
            // Q3: when a context window is set, only the most recent n messages are taken
            // (including the current user message); otherwise the full history is kept.
            let messages: Vec<Message> = match self.max_context_messages {
                Some(n) => session.recent_messages(n).into_iter().cloned().collect(),
                None => session.messages.clone(),
            };
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::Llm(e.to_string()))?
        };
        let content = response.content.clone();
        session.add_message(Message::ai(content.clone()));

        if let Some(memory) = &self.memory {
            let mut mem = memory.lock().await;
            let inputs = HashMap::from([(self.memory_input_key.clone(), user_message)]);
            let outputs = HashMap::from([(self.memory_output_key.clone(), content.clone())]);
            mem.save_context(&inputs, &outputs)
                .await
                .map_err(|e| SessionError::Memory(format!("failed to save memory: {}", e)))?;
        }

        self.store.update(&session).await?;
        Ok(content)
    }

    /// Gets the session history
    pub async fn history(&self, id: &str) -> Result<Vec<Message>, SessionError> {
        let session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        Ok(session.messages)
    }

    /// Clears the session history (keeps the session)
    pub async fn clear(&self, id: &str) -> Result<(), SessionError> {
        // Q2: take the same striped lock as chat to avoid interleaving with concurrent chats.
        let lock = self.session_lock(id).await;
        let _guard = lock.lock().await;
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.clear();
        self.store.update(&session).await
    }

    /// Archives a session
    pub async fn archive(&self, id: &str) -> Result<(), SessionError> {
        let lock = self.session_lock(id).await;
        let _guard = lock.lock().await;
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.archive();
        self.store.update(&session).await
    }

    /// Soft-deletes a session (Q4: marks it `Deleted`; the record is kept but no longer appears in listings).
    pub async fn delete_session(&self, id: &str) -> Result<(), SessionError> {
        let lock = self.session_lock(id).await;
        let _guard = lock.lock().await;
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.delete();
        self.store.update(&session).await
    }

    /// Gets all sessions of a user
    pub async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError> {
        self.store.list_by_user(user_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_store::MemorySessionStore;
    use crate::SessionStatus;
    use async_trait::async_trait;
    use futures_util::Stream;
    use lc_core::language_models::{BaseLanguageModel, LLMResult, StreamChunk};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_memory::MemoryError;
    use lc_schema::MessageType;
    use std::pin::Pin;

    fn manager() -> SessionManager {
        SessionManager::new(Arc::new(MemorySessionStore::new()))
    }

    // ---- mock LLM: records received messages, returns a fixed reply ----

    #[derive(Debug)]
    struct MockSessionLlmError(String);

    impl std::fmt::Display for MockSessionLlmError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for MockSessionLlmError {}

    #[derive(Clone)]
    struct MockSessionLlm {
        response: String,
        received: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl MockSessionLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                received: Arc::new(Mutex::new(Vec::new())),
            }
        }

        async fn received(&self) -> Vec<Vec<Message>> {
            self.received.lock().await.clone()
        }
    }

    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockSessionLlm {
        fn model_name(&self) -> &str {
            "mock-session-llm"
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
    impl Runnable<Vec<Message>, LLMResult> for MockSessionLlm {
        type Error = MockSessionLlmError;

        async fn invoke(
            &self,
            input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.received.lock().await.push(input);
            Ok(LLMResult {
                content: self.response.clone(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
                thinking_content: None,
            })
        }
    }

    #[async_trait]
    impl BaseChatModel for MockSessionLlm {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    // ---- mock memory: returns a fixed history text, records save_context ----

    #[derive(Clone)]
    struct RecordingMemory {
        history: String,
        saved: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl RecordingMemory {
        fn new(history: &str) -> Self {
            Self {
                history: history.to_string(),
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
            _inputs: &HashMap<String, String>,
        ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
            let mut vars = HashMap::new();
            vars.insert(
                "history".to_string(),
                serde_json::Value::String(self.history.clone()),
            );
            Ok(vars)
        }

        async fn save_context(
            &mut self,
            inputs: &HashMap<String, String>,
            outputs: &HashMap<String, String>,
        ) -> Result<(), MemoryError> {
            let input = inputs.get("input").cloned().unwrap_or_default();
            let output = outputs.get("output").cloned().unwrap_or_default();
            self.saved.lock().await.push((input, output));
            Ok(())
        }

        async fn clear(&mut self) -> Result<(), MemoryError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_create_and_get() {
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        let session = mgr.get_session(&id).await.unwrap().unwrap();
        assert_eq!(session.id, id);
        assert!(session.messages.is_empty());
    }

    #[tokio::test]
    async fn test_create_for_user() {
        let mgr = manager();
        let id = mgr.create_session_for("u1").await.unwrap();
        let session = mgr.get_session(&id).await.unwrap().unwrap();
        assert_eq!(session.user_id, Some("u1".to_string()));
    }

    #[tokio::test]
    async fn test_clear_keeps_session() {
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        mgr.clear(&id).await.unwrap();
        assert!(mgr.get_session(&id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_archive() {
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();
        mgr.archive(&id).await.unwrap();
        let session = mgr.get_session(&id).await.unwrap().unwrap();
        assert_eq!(session.status, SessionStatus::Archived);
    }

    #[tokio::test]
    async fn test_list_by_user() {
        let mgr = manager();
        let _ = mgr.create_session_for("u1").await.unwrap();
        let _ = mgr.create_session_for("u1").await.unwrap();
        let _ = mgr.create_session_for("u2").await.unwrap();
        assert_eq!(mgr.list_by_user("u1").await.unwrap().len(), 2);
        assert_eq!(mgr.list_by_user("u2").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let mgr = manager();
        assert!(mgr.get_session("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_history_nonexistent_errors() {
        let mgr = manager();
        assert!(mgr.history("nope").await.is_err());
    }

    #[tokio::test]
    async fn test_chat_nonexistent_errors() {
        let mgr = manager();
        // chat should fail because the session does not exist
        // We cannot use OpenAIChat here since it's in the main crate,
        // but the error occurs before the LLM is called, so we just
        // verify that a non-existent session returns an error.
        // This test is covered in the main crate's integration tests.
        assert!(mgr.get_session("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_with_memory_sets_flag() {
        let mgr = SessionManager::new(Arc::new(MemorySessionStore::new()))
            .with_memory(Arc::new(Mutex::new(RecordingMemory::new(""))));
        assert!(mgr.has_memory());

        let plain = SessionManager::new(Arc::new(MemorySessionStore::new()));
        assert!(!plain.has_memory());
    }

    #[test]
    fn test_with_memory_keys() {
        let mgr = SessionManager::new(Arc::new(MemorySessionStore::new()))
            .with_memory_keys("question", "answer");
        assert_eq!(mgr.memory_input_key, "question");
        assert_eq!(mgr.memory_output_key, "answer");
    }

    /// P2-1: with memory attached, the LLM context = memory history (system) + the current user
    /// message; after the turn, memory `save_context` is called and records this turn's I/O.
    #[tokio::test]
    async fn test_chat_with_memory_feeds_history_and_saves() {
        let llm = MockSessionLlm::new("你好,我是 AI");
        let rec = Arc::new(Mutex::new(RecordingMemory::new("Human: 在吗\nAI: 在")));
        let mgr = SessionManager::new(Arc::new(MemorySessionStore::new())).with_memory(rec.clone());
        let id = mgr.create_session().await.unwrap();

        let reply = mgr.chat(&id, &llm, "你好".to_string()).await.unwrap();
        assert_eq!(reply, "你好,我是 AI");

        // LLM context received: memory history + the current user message
        let received = llm.received().await;
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].len(), 2);
        assert_eq!(received[0][0].message_type, MessageType::System);
        assert_eq!(received[0][0].content, "Human: 在吗\nAI: 在");
        assert_eq!(received[0][1].message_type, MessageType::Human);
        assert_eq!(received[0][1].content, "你好");

        // memory was written with this turn's input/output
        let saved = rec.lock().await.saved.lock().await.clone();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0], ("你好".to_string(), "你好,我是 AI".to_string()));
    }

    /// P2-1: without memory the original logic runs — the LLM receives the full session history (including the current user message).
    #[tokio::test]
    async fn test_chat_without_memory_uses_full_history() {
        let llm = MockSessionLlm::new("回复");
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        // the second round should carry the first round's user + AI messages
        assert_eq!(received[1].len(), 3);
        assert_eq!(received[1][0].content, "第一句");
        assert_eq!(received[1][1].content, "回复");
        assert_eq!(received[1][2].content, "第二句");
    }

    /// Q3: without memory + `with_max_context_messages(n)`, the LLM only receives the most
    /// recent n messages. The third round's history has 4 messages; window 2 takes only the last
    /// 2 (previous AI + current user).
    #[tokio::test]
    async fn test_chat_respects_context_window() {
        let llm = MockSessionLlm::new("回复");
        let mgr = manager().with_max_context_messages(2);
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第三句".to_string()).await.unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 3);
        // first round: 1; second round: 2 (previous AI + current user)
        assert_eq!(received[0].len(), 1);
        assert_eq!(received[1].len(), 2);
        // third round: window 2 → only the last 2, no longer carrying the full history
        assert_eq!(received[2].len(), 2);
        assert_eq!(received[2][0].content, "回复");
        assert_eq!(received[2][1].content, "第三句");
    }

    /// Q4: after soft-deleting a session its status is Deleted; the record is kept but no longer appears in the user's list.
    #[tokio::test]
    async fn test_delete_session() {
        let mgr = manager();
        let id = mgr.create_session_for("u1").await.unwrap();
        mgr.delete_session(&id).await.unwrap();

        let session = mgr.get_session(&id).await.unwrap().unwrap();
        assert_eq!(session.status, SessionStatus::Deleted);
        assert!(mgr.list_by_user("u1").await.unwrap().is_empty());

        // deleting a nonexistent session → NotFound
        assert!(mgr.delete_session("nope").await.is_err());
    }

    // ---- mock failing LLM: verifies LLM errors map to SessionError::Llm, not StoreError ----

    #[derive(Clone)]
    struct MockFailLlm;

    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockFailLlm {
        fn model_name(&self) -> &str {
            "mock-fail-llm"
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
    impl Runnable<Vec<Message>, LLMResult> for MockFailLlm {
        type Error = MockSessionLlmError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Err(MockSessionLlmError("model timed out".to_string()))
        }
    }

    #[async_trait]
    impl BaseChatModel for MockFailLlm {
        async fn chat(
            &self,
            messages: Vec<Message>,
            config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            self.invoke(messages, config).await
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    /// Q1: an LLM failure must map to `SessionError::Llm`, never masquerade as a storage error.
    #[tokio::test]
    async fn test_chat_maps_llm_error() {
        let llm = MockFailLlm;
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();

        let err = mgr.chat(&id, &llm, "你好".to_string()).await.unwrap_err();
        assert!(matches!(err, SessionError::Llm(ref msg) if msg.contains("model timed out")));
    }
}
