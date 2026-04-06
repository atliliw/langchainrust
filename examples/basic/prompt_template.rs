// examples/basic/prompt_template.rs
//! 基础示例 3: 提示词模板
//!
//! 运行: cargo run --example prompt_template
//!
//! 功能: 演示如何使用 PromptTemplate 和 ChatPromptTemplate

use langchainrust::prompts::{ChatPromptTemplate, PromptTemplate};
use langchainrust::schema::Message;
use std::collections::HashMap;

fn main() {
    println!("=== 基础示例 3: 提示词模板 ===\n");

    // ========== 1. PromptTemplate (简单字符串模板) ==========
    println!("--- 1. PromptTemplate 示例 ---\n");

    // 创建模板
    let template = PromptTemplate::new(
        "请用{style}的风格，解释什么是{topic}。\n\
         要求：不超过{length}字。",
    );

    // 填充变量
    let mut vars = HashMap::new();
    vars.insert("style", "通俗易懂");
    vars.insert("topic", "机器学习");
    vars.insert("length", "50");

    let prompt = template.format(&vars).unwrap();
    println!("生成的提示词:\n{}\n", prompt);

    // 另一个例子
    vars.insert("style", "专业严谨");
    vars.insert("topic", "量子计算");
    vars.insert("length", "100");

    let prompt2 = template.format(&vars).unwrap();
    println!("另一个提示词:\n{}\n", prompt2);

    // ========== 2. ChatPromptTemplate (聊天消息模板) ==========
    println!("--- 2. ChatPromptTemplate 示例 ---\n");

    // 创建聊天模板
    let chat_template = ChatPromptTemplate::new(vec![
        Message::system("你是一个{role}，专精于{domain}领域。"),
        Message::human("你好，我是{name}。"),
        Message::ai("你好{name}！我是你的{role}助手，有什么可以帮你的吗？"),
        Message::human("{question}"),
    ]);

    // 填充变量
    let mut chat_vars = HashMap::new();
    chat_vars.insert("role", "编程专家");
    chat_vars.insert("domain", "Rust 语言");
    chat_vars.insert("name", "小明");
    chat_vars.insert("question", "请解释 Rust 的所有权系统");

    let messages = chat_template.format(&chat_vars).unwrap();

    println!("生成的聊天消息:");
    for (i, msg) in messages.iter().enumerate() {
        let role = match msg.message_type {
            langchainrust::schema::MessageType::System => "System",
            langchainrust::schema::MessageType::Human => "Human",
            langchainrust::schema::MessageType::AI => "AI",
            langchainrust::schema::MessageType::Tool { .. } => "Tool",
        };
        println!(
            "  [{}] {}: {}",
            i + 1,
            role,
            msg.content.chars().take(50).collect::<String>()
        );
        if msg.content.len() > 50 {
            println!("       ...");
        }
    }

    println!("\n--- 3. 模板复用 ---\n");

    // 同一个模板，不同变量
    chat_vars.insert("name", "小红");
    chat_vars.insert("question", "如何在 Rust 中处理错误？");

    let messages2 = chat_template.format(&chat_vars).unwrap();
    println!("复用模板生成的新消息:");
    println!("  最后一条: {}", messages2.last().unwrap().content);

    println!("\n=== 示例完成 ===");
}
