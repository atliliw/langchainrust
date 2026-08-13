//! ConversationChain 单元测试

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::memory::BaseMemory;
use langchainrust::schema::{Message, MessageType};
use langchainrust::{BaseChain, ConversationBufferMemory, ConversationChain, OpenAIChat};
use std::collections::HashMap;

/// Load the messages a chain's memory currently holds, through the
/// `BaseMemory` trait (the concrete `chat_memory()` accessor no longer exists
/// on the trait object stored by the chain).
async fn load_history_messages(chain: &ConversationChain<OpenAIChat>) -> Vec<Message> {
    let mem = chain.memory().lock().await;
    let vars = mem.load_memory_variables(&HashMap::new()).await.unwrap();
    let mut messages = Vec::new();
    for value in vars.values() {
        if let Some(arr) = value.as_array() {
            for item in arr {
                if let Ok(m) = serde_json::from_value::<Message>(item.clone()) {
                    messages.push(m);
                }
            }
        }
    }
    messages
}

#[test]
fn test_conversation_chain_new() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory);

    assert_eq!(chain.input_keys(), vec!["input"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
    assert_eq!(chain.name(), "conversation_chain");
}

#[test]
fn test_conversation_chain_with_options() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();

    let chain = ConversationChain::new(llm, memory)
        .with_system_prompt("你是一个友好的助手")
        .with_input_key("question")
        .with_output_key("answer")
        .with_verbose(true);

    assert_eq!(chain.input_keys(), vec!["question"]);
    assert_eq!(chain.output_keys(), vec!["answer"]);
}

#[test]
fn test_conversation_chain_builder() {
    let llm = TestConfig::get().openai_chat();

    let chain = ConversationChain::builder(llm)
        .system_prompt("你是一个 Rust 专家")
        .input_key("query")
        .output_key("response")
        .verbose(true)
        .build();

    assert_eq!(chain.input_keys(), vec!["query"]);
    assert_eq!(chain.output_keys(), vec!["response"]);
}

#[tokio::test]
async fn test_clear_memory() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory);

    chain.clear_memory().await.unwrap();

    let messages = load_history_messages(&chain).await;
    assert_eq!(messages.len(), 0);
}

#[tokio::test]
async fn test_memory_integration() {
    let mut memory = ConversationBufferMemory::new();

    let inputs = HashMap::from([("input".to_string(), "你好".to_string())]);
    let outputs = HashMap::from([("output".to_string(), "你好！".to_string())]);

    memory.save_context(&inputs, &outputs).await.unwrap();

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = vars.get("history").unwrap().as_str().unwrap();

    assert!(history.contains("Human: 你好"));
    assert!(history.contains("AI: 你好！"));
}

#[test]
fn test_prepare_messages_structure() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory).with_system_prompt("测试系统提示");

    let history = vec![Message::human("第一轮问题"), Message::ai("第一轮回答")];

    let messages = chain.prepare_messages("第二轮问题", &history);

    assert_eq!(messages.len(), 4);

    assert!(matches!(messages[0].message_type, MessageType::System));
    assert!(matches!(messages[1].message_type, MessageType::Human));
    assert!(matches!(messages[2].message_type, MessageType::AI));
    assert!(matches!(messages[3].message_type, MessageType::Human));

    assert_eq!(messages[0].content, "测试系统提示");
    assert_eq!(messages[1].content, "第一轮问题");
    assert_eq!(messages[2].content, "第一轮回答");
    assert_eq!(messages[3].content, "第二轮问题");
}

#[test]
fn test_prepare_messages_without_system_prompt() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory);

    let history = vec![Message::human("历史问题"), Message::ai("历史回答")];

    let messages = chain.prepare_messages("当前问题", &history);

    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0].message_type, MessageType::Human));
    assert!(matches!(messages[1].message_type, MessageType::AI));
    assert!(matches!(messages[2].message_type, MessageType::Human));
}

#[test]
fn test_prepare_messages_empty_history() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory).with_system_prompt("你是一个助手");

    let messages = chain.prepare_messages("第一个问题", &[]);

    assert_eq!(messages.len(), 2);
    assert!(matches!(messages[0].message_type, MessageType::System));
    assert!(matches!(messages[1].message_type, MessageType::Human));
}

#[tokio::test]
async fn test_conversation_chain_invoke_structure() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();
    let chain = ConversationChain::new(llm, memory);

    let count_before = load_history_messages(&chain).await.len();
    assert_eq!(count_before, 0);
}

#[tokio::test]
async fn test_memory_with_multiple_contexts() {
    let mut memory = ConversationBufferMemory::new();

    let inputs1 = HashMap::from([("input".to_string(), "第一轮".to_string())]);
    let outputs1 = HashMap::from([("output".to_string(), "回答1".to_string())]);
    memory.save_context(&inputs1, &outputs1).await.unwrap();

    let inputs2 = HashMap::from([("input".to_string(), "第二轮".to_string())]);
    let outputs2 = HashMap::from([("output".to_string(), "回答2".to_string())]);
    memory.save_context(&inputs2, &outputs2).await.unwrap();

    assert_eq!(memory.chat_memory().len(), 4);

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = vars.get("history").unwrap().as_str().unwrap();

    assert!(history.contains("第一轮"));
    assert!(history.contains("回答1"));
    assert!(history.contains("第二轮"));
    assert!(history.contains("回答2"));
}

#[test]
fn test_conversation_chain_default_memory() {
    let llm = TestConfig::get().openai_chat();
    let chain = ConversationChain::builder(llm).build();

    assert_eq!(chain.input_keys(), vec!["input"]);
    assert_eq!(chain.output_keys(), vec!["output"]);
}

// ============================================================================
// 真实 LLM 调用测试
// ============================================================================

#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_llm_single_call() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();

    let chain = ConversationChain::new(llm, memory)
        .with_system_prompt("你是一个友好的助手，请简短回答")
        .with_verbose(true);

    println!("\n=== 单轮对话测试 ===");
    let result = chain.predict("你好，介绍一下自己").await.unwrap();

    println!("用户: 你好，介绍一下自己");
    println!("AI: {}", result);

    assert!(!result.is_empty());

    let messages = load_history_messages(&chain).await;
    assert_eq!(messages.len(), 2);
}

#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_llm_multi_turn_memory() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();

    let chain = ConversationChain::new(llm, memory)
        .with_system_prompt("你是一个友好的助手。请记住用户告诉你的信息。")
        .with_verbose(true);

    println!("\n=== 多轮对话记忆测试 ===");

    println!("\n--- 第一轮 ---");
    let result1 = chain.predict("你好，我叫张三，我喜欢编程").await.unwrap();
    println!("用户: 你好，我叫张三，我喜欢编程");
    println!("AI: {}", result1);
    assert!(!result1.is_empty());

    println!("\n--- 第二轮 ---");
    let result2 = chain.predict("我叫什么名字？我喜欢什么？").await.unwrap();
    println!("用户: 我叫什么名字？我喜欢什么？");
    println!("AI: {}", result2);
    assert!(!result2.is_empty());

    println!("\n--- 验证记忆 ---");
    let messages = load_history_messages(&chain).await;
    let history = messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    println!("历史记录:\n{}", history);

    assert!(history.contains("张三"), "记忆应包含名字");
    assert!(history.contains("编程"), "记忆应包含爱好");
    assert_eq!(messages.len(), 4);
}

#[tokio::test]
#[ignore = "需要 LLM API Key 且账户余额充足"]
async fn test_llm_clear_memory() {
    let llm = TestConfig::get().openai_chat();
    let memory = ConversationBufferMemory::new();

    let chain = ConversationChain::new(llm, memory).with_verbose(true);

    println!("\n=== 清空记忆测试 ===");

    println!("\n--- 第一轮 ---");
    let result1 = chain.predict("我叫李四").await.unwrap();
    println!("用户: 我叫李四");
    println!("AI: {}", result1);

    println!("\n--- 清空记忆 ---");
    chain.clear_memory().await.unwrap();

    let messages = load_history_messages(&chain).await;
    assert_eq!(messages.len(), 0);
    println!("记忆已清空");

    println!("\n--- 第二轮（清空后） ---");
    let result2 = chain.predict("我叫什么名字？").await.unwrap();
    println!("用户: 我叫什么名字？");
    println!("AI: {}", result2);
    println!("注意: AI 不应知道 '李四'");
}
