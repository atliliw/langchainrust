// src/memory/window.rs
//! Conversation Buffer Window Memory
//!
//! 带窗口的对话记忆，只保留最近 k 轮对话。

use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;

use super::base::{BaseMemory, MemoryError, ChatMessageHistory};
use crate::schema::Message;

/// Conversation Buffer Window Memory
/// 
/// 只保留最近 k 轮对话，避免上下文过长。
/// 
/// # 示例
/// ```ignore
/// use langchainrust::ConversationBufferWindowMemory;
/// 
/// // 只保留最近 2 轮对话
/// let mut memory = ConversationBufferWindowMemory::new(2);
/// ```
#[derive(Debug)]
pub struct ConversationBufferWindowMemory {
    /// 聊天历史
    chat_memory: ChatMessageHistory,
    
    /// 窗口大小（保留最近 k 轮对话，默认 5）
    k: usize,
    
    /// 输入键名
    input_key: String,
    
    /// 输出键名
    output_key: String,
    
    /// 记忆变量名
    memory_key: String,
    
    /// 是否返回消息对象
    return_messages: bool,
}

impl ConversationBufferWindowMemory {
    /// 创建新的窗口记忆
    /// 
    /// # 参数
    /// * `k` - 保留最近 k 轮对话（每轮包含用户消息和 AI 消息）
    pub fn new(k: usize) -> Self {
        Self {
            chat_memory: ChatMessageHistory::new(),
            k,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            return_messages: false,
        }
    }
    
    /// 设置输入键名
    pub fn with_input_key(mut self, key: String) -> Self {
        self.input_key = key;
        self
    }
    
    /// 设置输出键名
    pub fn with_output_key(mut self, key: String) -> Self {
        self.output_key = key;
        self
    }
    
    /// 设置记忆变量名
    pub fn with_memory_key(mut self, key: String) -> Self {
        self.memory_key = key;
        self
    }
    
    /// 设置是否返回消息对象
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }
    
    /// 获取聊天历史
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }
    
    /// 获取窗口大小
    pub fn k(&self) -> usize {
        self.k
    }
    
    /// 获取窗口内的消息
    /// 
    /// 只保留最近 k 轮（2*k 条消息）
    fn get_window_messages(&self) -> Vec<Message> {
        let messages = self.chat_memory.messages();
        let total = messages.len();
        
        // 每轮包含 2 条消息（用户 + AI）
        let max_messages = self.k * 2;
        
        if total <= max_messages {
            messages.to_vec()
        } else {
            messages[total - max_messages..].to_vec()
        }
    }
    
    /// 转换为字符串
    fn buffer_as_string(&self) -> String {
        self.get_window_messages()
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
            .join("\n")
    }
}

impl Default for ConversationBufferWindowMemory {
    fn default() -> Self {
        Self::new(5)
    }
}

#[async_trait]
impl BaseMemory for ConversationBufferWindowMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }
    
    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();
        
        if self.return_messages {
            let messages: Vec<Value> = self.get_window_messages()
                .into_iter()
                .map(|msg| {
                    serde_json::to_value(&msg)
                        .unwrap_or(Value::Null)
                })
                .collect();
            result.insert(self.memory_key.clone(), Value::Array(messages));
        } else {
            result.insert(
                self.memory_key.clone(),
                Value::String(self.buffer_as_string())
            );
        }
        
        Ok(result)
    }
    
    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        // 保存用户输入
        if let Some(input) = inputs.get(&self.input_key) {
            self.chat_memory.add_user_message(input);
        }
        
        // 保存 AI 输出
        if let Some(output) = outputs.get(&self.output_key) {
            self.chat_memory.add_ai_message(output);
        }
        
        Ok(())
    }
    
    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.chat_memory.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_window_memory() {
        let mut memory = ConversationBufferWindowMemory::new(2);
        
        // 添加 3 轮对话（共 6 条消息）
        for i in 1..=3 {
            let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
            let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
            memory.save_context(&inputs, &outputs).await.unwrap();
        }
        
        // 完整历史有 6 条消息
        assert_eq!(memory.chat_memory().len(), 6);
        
        // 但只返回最近 2 轮（4 条消息）
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();
        
        // 应该包含问题2、答案2、问题3、答案3
        assert!(!history.contains("问题1"));
        assert!(!history.contains("答案1"));
        assert!(history.contains("问题2"));
        assert!(history.contains("答案3"));
    }
    
    #[tokio::test]
    async fn test_window_memory_smaller_than_k() {
        let mut memory = ConversationBufferWindowMemory::new(5);
        
        // 只添加 2 轮对话
        for i in 1..=2 {
            let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
            let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
            memory.save_context(&inputs, &outputs).await.unwrap();
        }
        
        // 应该返回全部 4 条消息
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();
        
        assert!(history.contains("问题1"));
        assert!(history.contains("问题2"));
    }
    
    #[tokio::test]
    async fn test_window_memory_clear() {
        let mut memory = ConversationBufferWindowMemory::new(2);
        
        let inputs = HashMap::from([("input".to_string(), "测试".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "收到".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();
        
        assert_eq!(memory.chat_memory().len(), 2);
        
        memory.clear().await.unwrap();
        assert_eq!(memory.chat_memory().len(), 0);
    }
}