//! Session storage trait and error types

use async_trait::async_trait;

use super::session::Session;

/// Session error
#[derive(Debug)]
#[non_exhaustive]
pub enum SessionError {
    /// Session not found
    NotFound(String),
    /// Storage operation error
    StoreError(String),
    /// LLM call error (Q1: an LLM failure must not masquerade as a storage error)
    Llm(String),
    /// Memory component error
    Memory(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::NotFound(id) => write!(f, "session not found: {}", id),
            SessionError::StoreError(msg) => write!(f, "session storage error: {}", msg),
            SessionError::Llm(msg) => write!(f, "LLM call error: {}", msg),
            SessionError::Memory(msg) => write!(f, "memory component error: {}", msg),
        }
    }
}

impl std::error::Error for SessionError {}

/// Session storage trait
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Creates a session, returning its ID
    async fn create(&self, session: Session) -> Result<String, SessionError>;
    /// Gets a session
    async fn get(&self, id: &str) -> Result<Option<Session>, SessionError>;
    /// Updates a session
    async fn update(&self, session: &Session) -> Result<(), SessionError>;
    /// Deletes a session
    async fn delete(&self, id: &str) -> Result<(), SessionError>;
    /// Gets all sessions of a user
    async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError>;
}
