//! ConversationSummaryBufferMemory 测试
//!
//! 测试结合摘要和完整对话的 Memory 功能：
//! 1. Token 限制触发摘要 - 超过限制时自动摘要旧对话
//! 2. 保留最近对话 - 最近几轮完整保留
//! 3. 摘要 + 完整对话组合 - 加载时返回摘要 + 最近对话
//! 4. 清空功能 - 清空摘要和 chat_memory

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::{BaseMemory, ConversationSummaryBufferMemory};
use std::collections::HashMap;

/// 测试 token 估算
#[test]
fn test_token_estimation() {
    println!("\n=== Token 估算演示 ===");

    let cases = [
        ("短消息", "我叫张三", 3),
        ("中等消息", "我想了解一下 Rust 语言的特点", 9),
        (
            "长消息",
            "我想了解一下 Rust 语言的特点和优势，特别是在系统编程方面的应用场景",
            30,
        ),
    ];

    for (name, text, expected_approx) in cases {
        let tokens = text.len() / 4;
        println!(
            "{}: \"{}\" -> tokens={} (预期~{})",
            name, text, tokens, expected_approx
        );
        assert!(tokens > 0);
    }
}

/// 测试 prune 逻辑演示
#[test]
fn test_prune_logic_demo() {
    println!("\n=== prune_messages 逻辑演示 ===");

    let limit = 20;

    let messages = [
        ("M1", "第1轮问题内容很长很长很长很长很长"),
        ("M2", "第1轮回答内容也很长很长很长很长"),
        ("M3", "第2轮问题"),
        ("M4", "第2轮回答"),
        ("M5", "第3轮问题短"),
        ("M6", "第3轮回答短"),
    ];

    let total: usize = messages.iter().map(|(_, t)| t.len() / 4).sum();
    println!("全部消息 tokens: {} (limit = {})", total, limit);

    let mut kept = Vec::new();
    let mut current = 0;

    for (name, text) in messages.iter().rev() {
        let tokens = text.len() / 4;
        if current + tokens <= limit {
            kept.push((name, text, tokens));
            current += tokens;
        } else {
            break;
        }
    }

    kept.reverse();

    println!("保留的消息 (tokens={}):", current);
    for (name, _, tokens) in &kept {
        println!("  {} ({} tokens)", name, tokens);
    }

    let kept_count = kept.len();
    println!("被裁掉的消息 (送去 LLM 摘要):");
    for (name, text) in messages.iter().take(messages.len() - kept_count) {
        println!("  {} ({} tokens): \"{}\"", name, text.len() / 4, text);
    }

    assert!(current <= limit);
}

/// 测试清空功能
#[tokio::test]
async fn test_clear_summary_buffer() {
    let config = TestConfig::get();

    let mut memory = ConversationSummaryBufferMemory::new(config.openai_chat(), 100);

    println!("\n=== 测试：清空功能 ===");

    let inputs = HashMap::from([("input".to_string(), "测试消息".to_string())]);
    let outputs = HashMap::from([("output".to_string(), "测试回复".to_string())]);
    memory.save_context(&inputs, &outputs).await.unwrap();

    println!("清空前: 消息数={}", memory.chat_memory().len());

    memory.clear().await.unwrap();

    println!("清空后: 消息数={}", memory.chat_memory().len());

    assert!(memory.buffer().await.is_empty());
    assert_eq!(memory.chat_memory().len(), 0);
}

/// 测试自定义配置
#[tokio::test]
async fn test_custom_config() {
    let config = TestConfig::get();

    let mut memory = ConversationSummaryBufferMemory::new(config.openai_chat(), 100)
        .with_input_key("question")
        .with_output_key("answer")
        .with_memory_key("context")
        .with_return_messages(true);

    println!("\n=== 测试：自定义配置 ===");

    let inputs = HashMap::from([("question".to_string(), "测试".to_string())]);
    let outputs = HashMap::from([("answer".to_string(), "回复".to_string())]);

    memory.save_context(&inputs, &outputs).await.unwrap();

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();

    assert!(vars.contains_key("context"));
    println!("使用自定义键 'context'");
}

/// 测试 return_messages 模式
#[tokio::test]
async fn test_return_messages_mode() {
    let config = TestConfig::get();

    let mut memory =
        ConversationSummaryBufferMemory::new(config.openai_chat(), 100).with_return_messages(true);

    println!("\n=== 测试：return_messages 模式 ===");

    let inputs = HashMap::from([("input".to_string(), "问题".to_string())]);
    let outputs = HashMap::from([("output".to_string(), "回答".to_string())]);
    memory.save_context(&inputs, &outputs).await.unwrap();

    let vars = memory.load_memory_variables(&HashMap::new()).await.unwrap();
    let history = vars.get("history").unwrap();

    println!("返回类型: {:?}", history);
    assert!(history.is_array());

    if history.is_array() {
        let arr = history.as_array().unwrap();
        println!("数组长度: {}", arr.len());
        assert!(!arr.is_empty());
    }
}
