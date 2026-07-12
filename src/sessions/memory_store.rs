//! 内存 Session 存储

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use super::session::Session;
use super::store::{SessionError, SessionStore};

/// 内存 Session 存储(用于测试与单进程场景)
pub struct MemorySessionStore {
    sessions: Mutex<HashMap<String, Session>>,
}

impl MemorySessionStore {
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
        self.sessions
            .lock()
            .unwrap()
            .insert(id.clone(), session);
        Ok(id)
    }

    async fn get(&self, id: &str) -> Result<Option<Session>, SessionError> {
        Ok(self.sessions.lock().unwrap().get(id).cloned())
    }

    async fn update(&self, session: &Session) -> Result<(), SessionError> {
        self.sessions
            .lock()
            .unwrap()
            .insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<(), SessionError> {
        self.sessions.lock().unwrap().remove(id);
        Ok(())
    }

    async fn list_by_user(&self, user_id: &str) -> Result<Vec<Session>, SessionError> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .values()
            .filter(|s| s.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Message;

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
