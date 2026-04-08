//! 单元测试 - Memory

mod loaders_csv;
mod vectorstores;

use langchainrust::{ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory};
use langchainrust::memory::BaseMemory;
use langchainrust::schema::{Message, MessageType};
use std::collections::HashMap;

#[test]
fn test_chat_message_history_basic() {
    let mut history = ChatMessageHistory::new();
    
    history.add_message(Message::human("Hello!"));
    history.add_message(Message::ai("Hi there!"));
    history.add_message(Message::human("How are you?"));
    
    let messages = history.messages();
    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0].message_type, MessageType::Human));
    assert!(matches!(messages[1].message_type, MessageType::AI));
}

#[test]
fn test_chat_message_history_clear() {
    let mut history = ChatMessageHistory::new();
    
    history.add_message(Message::human("Hello!"));
    assert_eq!(history.messages().len(), 1);
    
    history.clear();
    assert_eq!(history.messages().len(), 0);
}

#[tokio::test]
async fn test_conversation_buffer_memory() {
    let mut memory = ConversationBufferMemory::new();
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), "Hello".to_string());
    
    let mut outputs = HashMap::new();
    outputs.insert("output".to_string(), "Hi there!".to_string());
    
    memory.save_context(&inputs, &outputs).await.unwrap();
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    assert!(loaded.contains_key("history"));
}

#[tokio::test]
async fn test_conversation_buffer_window_memory() {
    let mut memory = ConversationBufferWindowMemory::new(2);
    
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), "Q1".to_string());
    
    let mut outputs = HashMap::new();
    outputs.insert("output".to_string(), "A1".to_string());
    
    memory.save_context(&inputs, &outputs).await.unwrap();
    
    inputs.insert("input".to_string(), "Q2".to_string());
    outputs.insert("output".to_string(), "A2".to_string());
    memory.save_context(&inputs, &outputs).await.unwrap();
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    assert!(loaded.contains_key("history"));
}