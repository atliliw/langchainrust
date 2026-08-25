//! Session 定义:多轮对话生命周期管理

use chrono::{DateTime, Utc};
use lc_schema::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// 会话状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// 活跃状态
    Active,
    /// 已归档状态
    Archived,
    /// 已删除状态
    Deleted,
}

/// 会话 - 一段多轮对话的完整生命周期
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// 会话 ID
    pub id: String,
    /// 关联用户 ID(可选)
    pub user_id: Option<String>,
    /// 对话消息列表
    pub messages: Vec<Message>,
    /// 会话元数据
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 会话状态
    pub status: SessionStatus,
}

impl Session {
    /// 创建新的会话
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

    /// 设置会话关联用户
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// 添加一条消息并更新时间
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

    /// 清空会话消息并更新时间
    pub fn clear(&mut self) {
        self.messages.clear();
        self.updated_at = Utc::now();
    }

    /// 将会话归档并更新时间
    pub fn archive(&mut self) {
        self.status = SessionStatus::Archived;
        self.updated_at = Utc::now();
    }

    /// 软删除:置为 `Deleted` 状态,保留记录以便审计/恢复。
    /// (Q4:`Deleted` 此前全仓库无人置位,补上删除流程让状态机闭环。)
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
