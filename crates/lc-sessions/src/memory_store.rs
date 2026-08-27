//! In-memory session storage

use std::collections::HashMap;
use tokio::sync::Mutex;

use async_trait::async_trait;

use super::session::{Session, SessionStatus};
use super::store::{SessionError, SessionStore};

/// In-memory session storage (for tests and single-process scenarios)
pub struct MemorySessionStore {
    sessions: Mutex<HashMap<String, Session>>,
}

impl MemorySessionStore {
    /// Creates a new in-memory session store
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for MemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStore for MemorySessionStore {
    async fn create(&self, session: Session) -> Result<String, SessionError> {
        let id = session.id.clone();
        self.sessions.lock().await.insert(id.clone(), session);
        Ok(id)
    }

    async fn get(&self, id: &str) -> Result<Option<Session>, SessionError> {
        Ok(self.sessions.lock().await.get(id).cloned())
    }

    async fn update(&self, session: &Session) -> Result<(), SessionError> {
        let mut sessions = self.sessions.lock().await;
        // Q4: update must target an existing session — an unconditional insert would mistake
        // "session missing" for "overwrite succeeded". Updating a nonexistent session must
        // explicitly return NotFound.
        if !sessions.contains_key(&session.id) {
            return Err(SessionError::NotFound(session.id.clone()));
        }
        sessions.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<(), SessionError> {
        self.sessions.lock().await.remove(id);
        Ok(())
    }

    async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError> {
        Ok(self
            .sessions
            .lock()
            .await
            .values()
            .filter(|s| s.user_id.as_deref() == Some(user_id) && s.status != SessionStatus::Deleted)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_schema::Message;

    fn sample_session(id: &str, user: &str) -> Session {
        let mut s = Session::new(id).with_user(user);
        s.add_message(Message::human("hi"));
        s
    }

    #[tokio::test]
    async fn test_crud() {
        let store = MemorySessionStore::new();
        let session = sample_session("s1", "u1");

        // create
        let id = store.create(session).await.unwrap();
        assert_eq!(id, "s1");

        // get
        let got = store.get("s1").await.unwrap().unwrap();
        assert_eq!(got.id, "s1");
        assert_eq!(got.messages.len(), 1);

        // update
        let mut s = got;
        s.add_message(Message::ai("hello"));
        store.update(&s).await.unwrap();
        assert_eq!(store.get("s1").await.unwrap().unwrap().messages.len(), 2);

        // list_by_user
        let list = store.list_by_user("u1").await.unwrap();
        assert_eq!(list.len(), 1);

        // delete
        store.delete("s1").await.unwrap();
        assert!(store.get("s1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let store = MemorySessionStore::new();
        assert!(store.get("nope").await.unwrap().is_none());
    }

    /// Q4: updating a nonexistent session must return NotFound, not silently behave like a successful insert.
    #[tokio::test]
    async fn test_update_nonexistent_errors() {
        let store = MemorySessionStore::new();
        let session = sample_session("s1", "u1");
        let err = store.update(&session).await.unwrap_err();
        assert!(matches!(err, SessionError::NotFound(id) if id == "s1"));
    }

    /// Q4: a soft-deleted session does not appear in list_by_user, but the record is still retrievable via get (for audit/recovery).
    #[tokio::test]
    async fn test_deleted_session_hidden_from_list_but_kept_in_store() {
        let store = MemorySessionStore::new();
        store.create(sample_session("s1", "u1")).await.unwrap();

        let mut s = store.get("s1").await.unwrap().unwrap();
        s.delete();
        store.update(&s).await.unwrap();

        assert!(store.list_by_user("u1").await.unwrap().is_empty());
        assert_eq!(
            store.get("s1").await.unwrap().unwrap().status,
            SessionStatus::Deleted
        );
    }

    #[tokio::test]
    async fn test_list_by_user_filtered() {
        let store = MemorySessionStore::new();
        store.create(sample_session("s1", "u1")).await.unwrap();
        store.create(sample_session("s2", "u2")).await.unwrap();
        assert_eq!(store.list_by_user("u1").await.unwrap().len(), 1);
        assert_eq!(store.list_by_user("u2").await.unwrap().len(), 1);
        assert_eq!(store.list_by_user("u3").await.unwrap().len(), 0);
    }
}
