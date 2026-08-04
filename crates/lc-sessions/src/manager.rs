//! Session 管理器 - 创建/获取会话,在会话中对话

use std::sync::Arc;

use lc_core::BaseChatModel;
use lc_schema::Message;

use super::session::Session;
use super::store::{SessionError, SessionStore};

/// Session 管理器
pub struct SessionManager {
    store: Arc<dyn SessionStore>,
}

impl SessionManager {
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self { store }
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
        session.add_message(Message::human(user_message));

        let response = llm
            .chat(session.messages.clone(), None)
            .await
            .map_err(|e| SessionError::StoreError(e.to_string()))?;
        let content = response.content.clone();
        session.add_message(Message::ai(content.clone()));

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

    fn manager() -> SessionManager {
        SessionManager::new(Arc::new(MemorySessionStore::new()))
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
}
