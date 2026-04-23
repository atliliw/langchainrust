// src/memory/summary_buffer.rs
//! Conversation Summary Buffer Memory
//!
//! 结合摘要和完整对话，平衡 token 消耗和对话质量。

use async_trait::async_trait;
use std::collections::HashMap;
use serde_json::Value;
use tokio::sync::Mutex;

use super::base::{BaseMemory, MemoryError, ChatMessageHistory};
use crate::language_models::OpenAIChat;
use crate::schema::Message;
use crate::Runnable;

const DEFAULT_SUMMARY_PROMPT: &str = "逐步总结对话内容，将新内容添加到之前的摘要中。

当前摘要：
{summary}

新增对话：
{new_lines}

新摘要：";

/// Conversation Summary Buffer Memory
///
/// 结合摘要和完整对话：
/// - 保留最近 k 轮完整对话（确保流畅性）
/// - 对旧对话进行摘要（节省 token）
///
/// # 示例
/// ```ignore
/// use langchainrust::{ConversationSummaryBufferMemory, OpenAIChat};
///
/// let llm = OpenAIChat::new(config);
/// let memory = ConversationSummaryBufferMemory::new(llm, 5); // 保留最近 5 轮
///
/// // 20 轮对话后：
/// // - 前 15 轮 → 摘要
/// // - 最近 5 轮 → 完整对话
/// ```
pub struct ConversationSummaryBufferMemory {
    llm: OpenAIChat,
    
    buffer: Mutex<String>,
    chat_memory: ChatMessageHistory,
    
    max_token_limit: usize,
    
    input_key: String,
    output_key: String,
    memory_key: String,
    
    summary_prompt: String,
    return_messages: bool,
}

impl ConversationSummaryBufferMemory {
    pub fn new(llm: OpenAIChat, max_token_limit: usize) -> Self {
        Self {
            llm,
            buffer: Mutex::new(String::new()),
            chat_memory: ChatMessageHistory::new(),
            max_token_limit,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            memory_key: "history".to_string(),
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            return_messages: false,
        }
    }
    
    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }
    
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }
    
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }
    
    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }
    
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }
    
    pub fn chat_memory(&self) -> &ChatMessageHistory {
        &self.chat_memory
    }
    
    pub fn chat_memory_mut(&mut self) -> &mut ChatMessageHistory {
        &mut self.chat_memory
    }
    
    pub fn max_token_limit(&self) -> usize {
        self.max_token_limit
    }
    
    pub async fn buffer(&self) -> String {
        self.buffer.lock().await.clone()
    }
    
    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }
    
    fn prune_messages(&self, messages: &[Message]) -> Vec<Message> {
        let total_tokens = messages.iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum::<usize>();
        
        if total_tokens <= self.max_token_limit {
            return messages.to_vec();
        }
        
        let mut kept_messages = Vec::new();
        let mut current_tokens = 0;
        
        for msg in messages.iter().rev() {
            let msg_tokens = Self::estimate_tokens(&msg.content);
            if current_tokens + msg_tokens <= self.max_token_limit {
                kept_messages.push(msg.clone());
                current_tokens += msg_tokens;
            } else {
                break;
            }
        }
        
        kept_messages.reverse();
        kept_messages
    }
    
    async fn predict_new_summary(&self, new_lines: &str) -> Result<String, MemoryError> {
        let buffer = self.buffer.lock().await.clone();
        
        let prompt = self.summary_prompt
            .replace("{summary}", &buffer)
            .replace("{new_lines}", new_lines);
        
        let messages = vec![Message::human(&prompt)];
        
        let result = self.llm.invoke(messages, None).await
            .map_err(|e| MemoryError::SaveError(format!("LLM 摘要失败: {}", e)))?;
        
        Ok(result.content)
    }
}

#[async_trait]
impl BaseMemory for ConversationSummaryBufferMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec![&self.memory_key]
    }
    
    async fn load_memory_variables(
        &self,
        _inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, Value>, MemoryError> {
        let mut result = HashMap::new();
        
        let buffer = self.buffer.lock().await.clone();
        let messages = self.chat_memory.messages();
        let pruned = self.prune_messages(messages);
        
        if self.return_messages {
            let mut all_messages = Vec::new();
            
            if !buffer.is_empty() {
                all_messages.push(Message::system(&buffer));
            }
            
            all_messages.extend(pruned);
            
            let messages_value: Vec<Value> = all_messages.iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect();
            
            result.insert(self.memory_key.clone(), Value::Array(messages_value));
        } else {
            let mut history = String::new();
            
            if !buffer.is_empty() {
                history.push_str(&format!("摘要: {}\n\n", buffer));
            }
            
            for msg in &pruned {
                let role = match msg.message_type {
                    crate::schema::MessageType::Human => "Human",
                    crate::schema::MessageType::AI => "AI",
                    crate::schema::MessageType::System => "System",
                    crate::schema::MessageType::Tool { .. } => "Tool",
                };
                history.push_str(&format!("{}: {}\n", role, msg.content));
            }
            
            result.insert(self.memory_key.clone(), Value::String(history));
        }
        
        Ok(result)
    }
    
    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        let empty = String::new();
        let input = inputs.get(&self.input_key).unwrap_or(&empty);
        let output = outputs.get(&self.output_key).unwrap_or(&empty);
        
        self.chat_memory.add_user_message(input);
        self.chat_memory.add_ai_message(output);
        
        let messages = self.chat_memory.messages();
        let total_tokens = messages.iter()
            .map(|m| Self::estimate_tokens(&m.content))
            .sum::<usize>();
        
        if total_tokens > self.max_token_limit {
            let pruned = self.prune_messages(messages);
            
            let pruned_count = pruned.len();
            
            if messages.len() > pruned_count {
                let messages_to_summarize: Vec<&Message> = messages.iter()
                    .take(messages.len() - pruned_count)
                    .collect();
                
                if !messages_to_summarize.is_empty() {
                    let new_lines: String = messages_to_summarize.iter()
                        .map(|m| {
                            let role = match m.message_type {
                                crate::schema::MessageType::Human => "Human",
                                crate::schema::MessageType::AI => "AI",
                                crate::schema::MessageType::System => "System",
                                crate::schema::MessageType::Tool { .. } => "Tool",
                            };
                            format!("{}: {}", role, m.content)
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    
                    let new_summary = self.predict_new_summary(&new_lines).await?;
                    
                    let mut buffer = self.buffer.lock().await;
                    *buffer = new_summary;
                }
                
                self.chat_memory.clear();
                for msg in pruned {
                    if matches!(msg.message_type, crate::schema::MessageType::Human) {
                        self.chat_memory.add_user_message(&msg.content);
                    } else if matches!(msg.message_type, crate::schema::MessageType::AI) {
                        self.chat_memory.add_ai_message(&msg.content);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn clear(&mut self) -> Result<(), MemoryError> {
        let mut buffer = self.buffer.lock().await;
        *buffer = String::new();
        self.chat_memory.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpenAIConfig;
    
    fn create_test_config() -> OpenAIConfig {
        OpenAIConfig::default()
    }
    
    #[test]
    fn test_new() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationSummaryBufferMemory::new(llm, 1000);
        
        assert_eq!(memory.memory_variables(), vec!["history"]);
        assert_eq!(memory.max_token_limit(), 1000);
    }
    
    #[test]
    fn test_with_options() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationSummaryBufferMemory::new(llm, 500)
            .with_input_key("question")
            .with_output_key("answer")
            .with_memory_key("context")
            .with_return_messages(true);
        
        assert_eq!(memory.input_key, "question");
        assert_eq!(memory.output_key, "answer");
        assert_eq!(memory.memory_key, "context");
        assert!(memory.return_messages);
    }
    
    #[test]
    fn test_estimate_tokens() {
        let text1 = "Hello";
        let text2 = "Hello World";
        let text3 = "这是一段中文文本";
        
        assert!(ConversationSummaryBufferMemory::estimate_tokens(text1) > 0);
        assert!(ConversationSummaryBufferMemory::estimate_tokens(text2) > ConversationSummaryBufferMemory::estimate_tokens(text1));
        assert!(ConversationSummaryBufferMemory::estimate_tokens(text3) > 0);
    }
    
    #[test]
    fn test_prune_messages_within_limit() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationSummaryBufferMemory::new(llm, 1000);
        
        let messages = vec![
            Message::human("短消息1"),
            Message::ai("短回复1"),
        ];
        
        let pruned = memory.prune_messages(&messages);
        
        assert_eq!(pruned.len(), 2);
    }
    
    #[tokio::test]
    async fn test_buffer_initial_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationSummaryBufferMemory::new(llm, 1000);
        
        let buffer = memory.buffer().await;
        assert!(buffer.is_empty());
    }
    
    #[tokio::test]
    async fn test_load_memory_variables_empty() {
        let llm = OpenAIChat::new(create_test_config());
        let memory = ConversationSummaryBufferMemory::new(llm, 1000);
        
        let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        
        assert!(history.is_empty());
    }
    
    #[tokio::test]
    async fn test_clear() {
        let llm = OpenAIChat::new(create_test_config());
        let mut memory = ConversationSummaryBufferMemory::new(llm, 1000);
        
        memory.chat_memory.add_user_message("测试");
        memory.chat_memory.add_ai_message("回复");
        
        let mut buffer = memory.buffer.lock().await;
        *buffer = "测试摘要".to_string();
        drop(buffer);
        
        memory.clear().await.unwrap();
        
        assert!(memory.buffer().await.is_empty());
        assert_eq!(memory.chat_memory().len(), 0);
    }
}