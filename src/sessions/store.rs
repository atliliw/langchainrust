//! Session 存储 trait 与错误类型

use async_trait::async_trait;

use super::session::Session;

/// Session 错误
#[derive(Debug)]
pub enum SessionError {
    /// Session 不存在
    NotFound(String),
    /// 存储操作错误
    StoreError(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::NotFound(id) => write!(f, "Session 不存在: {}", id),
            SessionError::StoreError(msg) => write!(f, "Session 存储错误: {}", msg),
        }
    }
}

impl std::error::Error for SessionError {}

/// Session 存储 trait
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// 创建会话,返回会话 ID
    async fn create(&self, session: Session) -> Result<String, SessionError>;
    /// 获取会话
    async fn get(&self, id: &str) -> Result<Option<Session>, SessionError>;
    /// 更新会话
    async fn update(&self, session: &Session) -> Result<(), SessionError>;
    /// 删除会话
    async fn delete(&self, id: &str) -> Result<(), SessionError>;
    /// 获取用户所有会话
    async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError>;
}
