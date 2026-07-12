//! 单元测试 - Memory

mod loaders_csv;
mod vectorstores;

use langchainrust::{ChatMessageHistory, ConversationBufferMemory, ConversationBufferWindowMemory};
use langchainrust::memory::BaseMemory;
use langchainrust::schema::{Message, MessageType};
use std::collections::HashMap;

#[test]
fn test_chat_message_history_basic() {
    println!("\n=== ChatMessageHistory 基础测试 ===");
    
    let mut history = ChatMessageHistory::new();
    
    history.add_message(Message::human("你好"));
    history.add_message(Message::ai("你好！有什么可以帮助你的？"));
    history.add_message(Message::human("介绍一下 Rust"));
    history.add_message(Message::ai("Rust 是一门系统编程语言"));
    
    println!("消息数量: {}", history.len());
    for (i, msg) in history.messages().iter().enumerate() {
        println!("  [{}] {:?}: {}", i, msg.message_type, msg.content);
    }
    
    println!("\n格式化输出:\n{}", history);
    
    assert_eq!(history.messages().len(), 4);
    assert!(matches!(history.messages()[0].message_type, MessageType::Human));
}

#[test]
fn test_chat_message_history_clear() {
    println!("\n=== ChatMessageHistory 清空测试 ===");
    
    let mut history = ChatMessageHistory::new();
    history.add_message(Message::human("测试"));
    
    println!("清空前: {} 条", history.messages().len());
    assert_eq!(history.messages().len(), 1);
    
    history.clear();
    println!("清空后: {} 条", history.messages().len());
    assert_eq!(history.messages().len(), 0);
}

#[tokio::test]
async fn test_conversation_buffer_memory() {
    println!("\n=== ConversationBufferMemory 测试 ===");
    
    let mut memory = ConversationBufferMemory::new();
    
    println!("添加 3 轴对话:");
    for i in 1..=3 {
        let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
        let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
        memory.save_context(&inputs, &outputs).await.unwrap();
        println!("  第{}轮完成, chat_memory={} 条", i, memory.chat_memory().len());
    }
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = loaded.get("history").unwrap().as_str().unwrap();
    
    println!("\n加载的历史:\n{}", history);
    
    assert!(loaded.contains_key("history"));
    assert!(history.contains("问题1"));
    assert!(history.contains("问题3"));
}

#[tokio::test]
async fn test_conversation_buffer_memory_return_messages() {
    println!("\n=== BufferMemory return_messages 模式 ===");
    
    let mut memory = ConversationBufferMemory::new()
        .with_return_messages(true);
    
    let inputs = HashMap::from([("input".to_string(), "测试问题".to_string())]);
    let outputs = HashMap::from([("output".to_string(), "测试回答".to_string())]);
    memory.save_context(&inputs, &outputs).await.unwrap();
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = loaded.get("history").unwrap();
    
    println!("返回类型: {:?}", history);
    
    assert!(history.is_array());
    let arr = history.as_array().unwrap();
    println!("数组长度: {}", arr.len());
    assert_eq!(arr.len(), 2);
}

#[tokio::test]
async fn test_conversation_buffer_window_memory() {
    println!("\n=== ConversationBufferWindowMemory 测试 ===");
    
    let k = 2;
    let mut memory = ConversationBufferWindowMemory::new(k);
    
    println!("窗口大小 k={} (保留最近{}轮={}条消息)", k, k, k * 2);
    
    println!("\n添加 4 轴对话:");
    for i in 1..=4 {
        let inputs = HashMap::from([("input".to_string(), format!("问题{}", i))]);
        let outputs = HashMap::from([("output".to_string(), format!("答案{}", i))]);
        memory.save_context(&inputs, &outputs).await.unwrap();
        
        println!("\n第{}轮后:", i);
        println!("  chat_memory 实际: {} 条", memory.chat_memory().len());
        
        let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
        let history = loaded.get("history").unwrap().as_str().unwrap();
        println!("  返回的 history:\n{}", history);
    }
    
    println!("\n=== 结论 ===");
    println!("完整历史 8 条，只返回最近 {} 条", k * 2);
    println!("问题1、2 被丢弃");
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = loaded.get("history").unwrap().as_str().unwrap();
    
    assert!(loaded.contains_key("history"));
    assert!(!history.contains("问题1"));
    assert!(history.contains("问题3"));
    assert!(history.contains("问题4"));
}

#[tokio::test]
async fn test_conversation_buffer_window_memory_small_window() {
    println!("\n=== WindowMemory 小窗口测试 (k=1) ===");
    
    let mut memory = ConversationBufferWindowMemory::new(1);
    
    for i in 1..=3 {
        let inputs = HashMap::from([("input".to_string(), format!("消息{}", i))]);
        let outputs = HashMap::from([("output".to_string(), format!("回复{}", i))]);
        memory.save_context(&inputs, &outputs).await.unwrap();
    }
    
    println!("chat_memory 实际: {} 条", memory.chat_memory().len());
    
    let loaded = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = loaded.get("history").unwrap().as_str().unwrap();
    
    println!("返回的 history:\n{}", history);
    
    assert!(history.contains("消息3"));
    assert!(!history.contains("消息1"));
}