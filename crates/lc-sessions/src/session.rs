//! Session 定义:多轮对话生命周期管理

use chrono::{DateTime, Utc};
use lc_schema::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// 会话状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    Active,
    Archived,
    Deleted,
}

/// 会话 - 一段多轮对话的完整生命周期
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_id: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: SessionStatus,
}

impl Session {
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

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// 最近 N 条消息(按时间顺序)
    pub fn recent_messages(&self, n: usize) -> Vec<&Message> {
        let len = self.messages.len();
        if len <= n {
            self.messages.iter().collect()
        } else {
            self.messages[len - n..].iter().collect()
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.updated_at = Utc::now();
    }

    pub fn archive(&mut self) {
        self.status = SessionStatus::Archived;
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
    fn test_session_clear() {
        let mut s = Session::new("s1");
        s.add_message(Message::human("hi"));
        s.clear();
        assert!(s.messages.is_empty());
    }
}
