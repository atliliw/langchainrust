// src/memory/buffer.rs
//! Conversation Buffer Memory
//!
//! 简单的对话缓冲区记忆，保存所有对话历史。

use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;

use super::base::{BaseMemory, MemoryError, ChatMessageHistory};
use crate::schema::Message;

/// Conversation Buffer Memory
/// 
/// 将所有对话历史保存在内存中。
/// 
/// # 示例
/// ```ignore
/// use langchainrust::ConversationBufferMemory;
/// 
/// let mut memory = ConversationBufferMemory::new();
/// 
/// // 保存对话
/// let inputs = HashMap::from([("input".to_string(), "你好".to_string())]);
/// let outputs = HashMap::from([("output".to_string(), "你好！".to_string())]);
/// memory.save_context(&inputs, &outputs).await?;
/// 
/// // 加载记忆
/// let memory_vars = memory.load_memory_variables(&HashMap::new()).await?;
/// println!("{:?}", memory_vars.get("history"));
/// ```
#[derive(Debug)]
pub struct ConversationBufferMemory {
    /// 聊天历史
    chat_memory: ChatMessageHistory,
    
    /// 输入键名（默认: "input"）
    input_key: String,
    
    /// 输出键名（默认: "output"）
    output_key: String,
    
    /// 记忆变量名（默认: "history"）
    memory_key: String,
    
    /// 是否返回消息对象
    return_messages: bool,
}

impl ConversationBufferMemory {
    /// 创建新的对话缓冲区记忆
    pub fn new() -> Self {
        Self {
            chat_memory: ChatMessageHistory::new(),
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
    
    /// 从已有历史创建
    pub fn from_chat_memory(chat_memory: ChatMessageHistory) -> Self {
        Self {
            chat_memory,
            ..Self::new()
        }
    }
    
    /// 获取聊天历史
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }
    
    /// 获取可变的聊天历史
    pub fn chat_memory_mut(&mut self) -> &mut ChatMessageHistory {
        &mut self.chat_memory
    }
    
    /// 转换历史为字符串
    fn buffer_as_string(&self) -> String {
        self.chat_memory.to_string()
    }
    
    /// 转换历史为消息列表
    fn buffer_as_messages(&self) -> Vec<Message> {
        self.chat_memory.messages().to_vec()
    }
}

impl Default for ConversationBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseMemory for ConversationBufferMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }
    
    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();
        
        if self.return_messages {
            // 返回消息列表
            let messages: Vec<Value> = self.buffer_as_messages()
                .into_iter()
                .map(|msg| {
                    serde_json::to_value(&msg)
                        .unwrap_or(Value::Null)
                })
                .collect();
            result.insert(self.memory_key.clone(), Value::Array(messages));
        } else {
            // 返回字符串
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
    async fn test_conversation_buffer_memory() {
        let mut memory = ConversationBufferMemory::new();
        
        // 保存对话
        let inputs = HashMap::from([("input".to_string(), "你好".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "你好！有什么可以帮助你的？".to_string())]);
        
        memory.save_context(&inputs, &outputs).await.unwrap();
        
        // 加载记忆
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        
        assert!(memory_vars.contains_key("history"));
        let history = memory_vars.get("history").unwrap();
        assert!(history.as_str().unwrap().contains("Human: 你好"));
        assert!(history.as_str().unwrap().contains("AI: 你好"));
    }
    
    #[tokio::test]
    async fn test_conversation_buffer_memory_multiple() {
        let mut memory = ConversationBufferMemory::new();
        
        // 第一轮对话
        let inputs1 = HashMap::from([("input".to_string(), "我叫张三".to_string())]);
        let outputs1 = HashMap::from([("output".to_string(), "你好张三！".to_string())]);
        memory.save_context(&inputs1, &outputs1).await.unwrap();
        
        // 第二轮对话
        let inputs2 = HashMap::from([("input".to_string(), "我叫什么？".to_string())]);
        let outputs2 = HashMap::from([("output".to_string(), "你叫张三".to_string())]);
        memory.save_context(&inputs2, &outputs2).await.unwrap();
        
        // 检查历史
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap().as_str().unwrap();
        
        assert!(history.contains("张三"));
        assert!(memory.chat_memory().len() == 4); // 2 轮 * 2 条消息
    }
    
    #[tokio::test]
    async fn test_conversation_buffer_memory_clear() {
        let mut memory = ConversationBufferMemory::new();
        
        // 保存对话
        let inputs = HashMap::from([("input".to_string(), "测试".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "收到".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();
        
        assert_eq!(memory.chat_memory().len(), 2);
        
        // 清空
        memory.clear().await.unwrap();
        assert_eq!(memory.chat_memory().len(), 0);
    }
    
    #[tokio::test]
    async fn test_conversation_buffer_memory_return_messages() {
        let mut memory = ConversationBufferMemory::new()
            .with_return_messages(true);
        
        let inputs = HashMap::from([("input".to_string(), "你好".to_string())]);
        let outputs = HashMap::from([("output".to_string(), "你好！".to_string())]);
        memory.save_context(&inputs, &outputs).await.unwrap();
        
        let memory_vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = memory_vars.get("history").unwrap();
        
        // 应该返回消息数组
        assert!(history.is_array());
        let messages = history.as_array().unwrap();
        assert_eq!(messages.len(), 2);
    }
}