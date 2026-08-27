//! Session definition: multi-turn conversation lifecycle management

use chrono::{DateTime, Utc};
use lc_schema::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// Active
    Active,
    /// Archived
    Archived,
    /// Deleted
    Deleted,
}

/// Session — the full lifecycle of a multi-turn conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Session ID
    pub id: String,
    /// Associated user ID (optional)
    pub user_id: Option<String>,
    /// Conversation message list
    pub messages: Vec<Message>,
    /// Session metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// Session status
    pub status: SessionStatus,
}

impl Session {
    /// Creates a new session
    pub fn new(id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            user_id: None,
            messages: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            status: SessionStatus::Active,
        }
    }

    /// Associates a user with the session
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Adds a message and refreshes the update time
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// The most recent N messages (in chronological order)
    pub fn recent_messages(&self, n: usize) -> Vec<&Message> {
        let len = self.messages.len();
        if len <= n {
            self.messages.iter().collect()
        } else {
            self.messages[len - n..].iter().collect()
        }
    }

    /// Clears the session messages and refreshes the update time
    pub fn clear(&mut self) {
        self.messages.clear();
        self.updated_at = Utc::now();
    }

    /// Archives the session and refreshes the update time
    pub fn archive(&mut self) {
        self.status = SessionStatus::Archived;
        self.updated_at = Utc::now();
    }

    /// Soft-deletes: marks the status `Deleted`, keeping the record for audit/recovery.
    /// (Q4: `Deleted` was previously never set anywhere in the repo; the delete flow closes the
    /// status-machine loop.)
    pub fn delete(&mut self) {
        self.status = SessionStatus::Deleted;
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new() {
        let s = Session::new("s1");
        assert_eq!(s.id, "s1");
        assert!(s.messages.is_empty());
        assert_eq!(s.status, SessionStatus::Active);
    }

    #[test]
    fn test_session_add_message() {
        let mut s = Session::new("s1");
        s.add_message(Message::human("hi"));
        s.add_message(Message::ai("hello"));
        assert_eq!(s.messages.len(), 2);
    }

    #[test]
    fn test_session_recent_messages() {
        let mut s = Session::new("s1");
        for i in 0..5 {
            s.add_message(Message::human(format!("msg{}", i)));
        }
        let recent = s.recent_messages(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, "msg2");
        assert_eq!(recent[2].content, "msg4");
    }

    #[test]
    fn test_session_with_user() {
        let s = Session::new("s1").with_user("user1");
        assert_eq!(s.user_id, Some("user1".to_string()));
    }

    #[test]
    fn test_session_archive() {
        let mut s = Session::new("s1");
        s.archive();
        assert_eq!(s.status, SessionStatus::Archived);
    }

    #[test]
    fn test_session_delete() {
        let mut s = Session::new("s1");
        s.delete();
        assert_eq!(s.status, SessionStatus::Deleted);
    }

    #[test]
    fn test_session_clear() {
        let mut s = Session::new("s1");
        s.add_message(Message::human("hi"));
        s.clear();
        assert!(s.messages.is_empty());
    }
}
