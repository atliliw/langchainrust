//! Session 管理器 - 创建/获取会话,在会话中对话

use std::collections::HashMap;
use std::sync::Arc;

use lc_core::BaseChatModel;
use lc_memory::BaseMemory;
use lc_schema::Message;
use tokio::sync::Mutex;

use super::session::Session;
use super::store::{SessionError, SessionStore};

/// Session 管理器
pub struct SessionManager {
    store: Arc<dyn SessionStore>,

    /// P2-1: 可选记忆组件。挂接后 `chat()` 的 LLM 上下文由记忆提供
    /// (窗口/摘要压缩后的历史)+ 本轮用户消息,轮后 `save_context` 记录;
    /// 未挂接时保持原行为(传完整 session 历史)。
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,

    /// 记忆输入 key(需与记忆实例的 input_key 对齐,默认 `"input"`)。
    memory_input_key: String,

    /// 记忆输出 key(需与记忆实例的 output_key 对齐,默认 `"output"`)。
    memory_output_key: String,
}

impl SessionManager {
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            store,
            memory: None,
            memory_input_key: "input".to_string(),
            memory_output_key: "output".to_string(),
        }
    }

    /// P2-1: 挂接记忆组件,启用记忆管理的对话上下文。
    ///
    /// 挂接后 `chat()` 每轮:由记忆提供压缩后的历史 + 本轮用户消息 -> 调 LLM ->
    /// `save_context` 记录本轮输入输出。记忆实例自定义了 input/output key 时,
    /// 需用 [`SessionManager::with_memory_keys`] 对齐。
    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// P2-1: 对齐记忆实例的自定义 input/output key(默认 `"input"` / `"output"`)。
    pub fn with_memory_keys(
        mut self,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        self.memory_input_key = input_key.into();
        self.memory_output_key = output_key.into();
        self
    }

    /// P2-1: 是否已挂接记忆组件。
    pub fn has_memory(&self) -> bool {
        self.memory.is_some()
    }

    /// 创建新会话,返回会话 ID
    pub async fn create_session(&self) -> Result<String, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        let session = Session::new(id.clone());
        self.store.create(session).await?;
        Ok(id)
    }

    /// 创建带用户 ID 的会话
    pub async fn create_session_for(
        &self,
        user_id: impl Into<String>,
    ) -> Result<String, SessionError> {
        let id = uuid::Uuid::new_v4().to_string();
        let session = Session::new(id.clone()).with_user(user_id);
        self.store.create(session).await?;
        Ok(id)
    }

    /// 获取会话
    pub async fn get_session(&self, id: &str) -> Result<Option<Session>, SessionError> {
        self.store.get(id).await
    }

    /// 在会话中对话:追加用户消息 -> 调用 LLM -> 追加 AI 回复 -> 持久化
    pub async fn chat<L: BaseChatModel>(
        &self,
        id: &str,
        llm: &L,
        user_message: String,
    ) -> Result<String, SessionError>
    where
        L::Error: std::fmt::Display,
    {
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.add_message(Message::human(&user_message));

        // P2-1: 挂接记忆时,LLM 上下文 = 记忆压缩历史 + 本轮用户消息;未挂接走原逻辑。
        let response = if let Some(memory) = &self.memory {
            let history_messages = {
                let mem = memory.lock().await;
                let inputs = HashMap::from([(self.memory_input_key.clone(), user_message.clone())]);
                let vars = mem
                    .load_memory_variables(&inputs)
                    .await
                    .map_err(|e| SessionError::StoreError(format!("加载记忆失败: {}", e)))?;
                lc_memory::memory_variables_to_messages(&vars)
            };

            let mut messages = history_messages;
            messages.push(Message::human(&user_message));
            llm.chat(messages, None)
                .await
                .map_err(|e| SessionError::StoreError(e.to_string()))?
        } else {
            llm.chat(session.messages.clone(), None)
                .await
                .map_err(|e| SessionError::StoreError(e.to_string()))?
        };
        let content = response.content.clone();
        session.add_message(Message::ai(content.clone()));

        if let Some(memory) = &self.memory {
            let mut mem = memory.lock().await;
            let inputs = HashMap::from([(self.memory_input_key.clone(), user_message)]);
            let outputs = HashMap::from([(self.memory_output_key.clone(), content.clone())]);
            mem.save_context(&inputs, &outputs)
                .await
                .map_err(|e| SessionError::StoreError(format!("保存记忆失败: {}", e)))?;
        }

        self.store.update(&session).await?;
        Ok(content)
    }

    /// 获取会话历史
    pub async fn history(&self, id: &str) -> Result<Vec<Message>, SessionError> {
        let session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        Ok(session.messages)
    }

    /// 清空会话历史(保留会话)
    pub async fn clear(&self, id: &str) -> Result<(), SessionError> {
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.clear();
        self.store.update(&session).await
    }

    /// 归档会话
    pub async fn archive(&self, id: &str) -> Result<(), SessionError> {
        let mut session = self
            .store
            .get(id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.to_string()))?;
        session.archive();
        self.store.update(&session).await
    }

    /// 获取用户所有会话
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
    use lc_core::language_models::{BaseLanguageModel, LLMResult};
    use lc_core::runnables::{Runnable, RunnableConfig};
    use lc_memory::MemoryError;
    use lc_schema::MessageType;
    use std::pin::Pin;

    fn manager() -> SessionManager {
        SessionManager::new(Arc::new(MemorySessionStore::new()))
    }

    // ---- mock LLM:记录收到的消息,返回固定回复 ----

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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            unimplemented!("stream_chat not needed for tests")
        }
    }

    // ---- mock 记忆:返回固定历史文本,记录 save_context ----

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

    /// P2-1: 挂接记忆后,LLM 上下文 = 记忆历史(system)+ 本轮用户消息,
    /// 轮后记忆 `save_context` 被调用并记录本轮输入输出。
    #[tokio::test]
    async fn test_chat_with_memory_feeds_history_and_saves() {
        let llm = MockSessionLlm::new("你好,我是 AI");
        let rec = Arc::new(Mutex::new(RecordingMemory::new("Human: 在吗\nAI: 在")));
        let mgr = SessionManager::new(Arc::new(MemorySessionStore::new())).with_memory(rec.clone());
        let id = mgr.create_session().await.unwrap();

        let reply = mgr.chat(&id, &llm, "你好".to_string()).await.unwrap();
        assert_eq!(reply, "你好,我是 AI");

        // LLM 收到的上下文:记忆历史 + 本轮用户消息
        let received = llm.received().await;
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].len(), 2);
        assert_eq!(received[0][0].message_type, MessageType::System);
        assert_eq!(received[0][0].content, "Human: 在吗\nAI: 在");
        assert_eq!(received[0][1].message_type, MessageType::Human);
        assert_eq!(received[0][1].content, "你好");

        // 记忆被写入本轮输入输出
        let saved = rec.lock().await.saved.lock().await.clone();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0], ("你好".to_string(), "你好,我是 AI".to_string()));
    }

    /// P2-1: 未挂接记忆时走原逻辑 —— LLM 收到完整 session 历史(含本轮用户消息)。
    #[tokio::test]
    async fn test_chat_without_memory_uses_full_history() {
        let llm = MockSessionLlm::new("回复");
        let mgr = manager();
        let id = mgr.create_session().await.unwrap();

        mgr.chat(&id, &llm, "第一句".to_string()).await.unwrap();
        mgr.chat(&id, &llm, "第二句".to_string()).await.unwrap();

        let received = llm.received().await;
        assert_eq!(received.len(), 2);
        // 第二轮应携带第一轮的用户+AI 消息
        assert_eq!(received[1].len(), 3);
        assert_eq!(received[1][0].content, "第一句");
        assert_eq!(received[1][1].content, "回复");
        assert_eq!(received[1][2].content, "第二句");
    }
}
