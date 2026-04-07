// src/memory/base.rs
//! Memory 基础 trait

use async_trait::async_trait;
use crate::schema::Message;
use std::collections::HashMap;

/// Memory 错误类型
#[derive(Debug)]
pub enum MemoryError {
    /// 加载错误
    LoadError(String),
    
    /// 保存错误
    SaveError(String),
    
    /// 清空错误
    ClearError(String),
    
    /// 其他错误
    Other(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::LoadError(msg) => write!(f, "加载记忆失败: {}", msg),
            MemoryError::SaveError(msg) => write!(f, "保存记忆失败: {}", msg),
            MemoryError::ClearError(msg) => write!(f, "清空记忆失败: {}", msg),
            MemoryError::Other(msg) => write!(f, "Memory 错误: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Base Memory trait
/// 
/// 所有 Memory 类型的基础接口。
#[async_trait]
pub trait BaseMemory: Send + Sync {
    /// 获取记忆变量名
    /// 
    /// 返回 memory 中存储的所有变量键。
    fn memory_variables(&self) -> Vec<&str>;
    
    /// 加载记忆变量
    /// 
    /// # 参数
    /// * `inputs` - 当前输入
    /// 
    /// # 返回
    /// 记忆变量字典
    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError>;
    
    /// 保存上下文
    /// 
    /// # 参数
    /// * `inputs` - 用户输入
    /// * `outputs` - 系统输出
    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError>;
    
    /// 清空记忆
    async fn clear(&mut self) -> Result<(), MemoryError>;
}

/// Base Chat Memory trait
/// 
/// 专门用于聊天场景的 Memory。
pub trait BaseChatMemory: BaseMemory {
    /// 获取聊天消息列表
    fn messages(&self) -> &Vec<Message>;
    
    /// 添加消息
    fn add_message(&mut self, message: Message);
    
    /// 添加用户消息
    fn add_user_message(&mut self, content: &str) {
        self.add_message(Message::human(content));
    }
    
    /// 添加 AI 消息
    fn add_ai_message(&mut self, content: &str) {
        self.add_message(Message::ai(content));
    }
}

/// 聊天消息缓冲区
/// 
/// 简单的消息存储，用于 ConversationBufferMemory。
#[derive(Debug, Clone)]
pub struct ChatMessageHistory {
    /// 消息列表
    messages: Vec<Message>,
}

impl ChatMessageHistory {
    /// 创建空的历史记录
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
    
    /// 从已有消息创建
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }
    
    /// 添加消息
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
    
    /// 添加用户消息
    pub fn add_user_message(&mut self, content: &str) {
        self.add_message(Message::human(content));
    }
    
    /// 添加 AI 消息
    pub fn add_ai_message(&mut self, content: &str) {
        self.add_message(Message::ai(content));
    }
    
    /// 添加系统消息
    pub fn add_system_message(&mut self, content: &str) {
        self.add_message(Message::system(content));
    }
    
    /// 获取所有消息
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    
    /// 清空消息
    pub fn clear(&mut self) {
        self.messages.clear();
    }
    
    /// 消息数量
    pub fn len(&self) -> usize {
        self.messages.len()
    }
    
    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

impl std::fmt::Display for ChatMessageHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let formatted: String = self.messages
            .iter()
            .map(|msg| {
                let role = match msg.message_type {
                    crate::schema::MessageType::Human => "Human",
                    crate::schema::MessageType::AI => "AI",
                    crate::schema::MessageType::System => "System",
                    crate::schema::MessageType::Tool { .. } => "Tool",
                };
                format!("{}: {}", role, msg.content)
            })
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{}", formatted)
    }
}

impl Default for ChatMessageHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chat_message_history() {
        let mut history = ChatMessageHistory::new();
        
        history.add_user_message("你好");
        history.add_ai_message("你好！有什么我可以帮助你的吗？");
        history.add_user_message("介绍一下自己");
        
        assert_eq!(history.len(), 3);
        assert!(!history.is_empty());
    }
    
    #[test]
    fn test_chat_message_history_to_string() {
        let mut history = ChatMessageHistory::new();
        
        history.add_user_message("你好");
        history.add_ai_message("你好！");
        
        let str = history.to_string();
        assert!(str.contains("Human: 你好"));
        assert!(str.contains("AI: 你好！"));
    }
    
    #[test]
    fn test_chat_message_history_clear() {
        let mut history = ChatMessageHistory::new();
        
        history.add_user_message("测试");
        assert_eq!(history.len(), 1);
        
        history.clear();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
    }
}