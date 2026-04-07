// examples/intermediate/memory_conversation.rs
//! 中级示例 2: 记忆与多轮对话
//!
//! 运行: cargo run --example memory_conversation
//!
//! 功能: 演示如何使用 Memory 实现多轮对话

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ConversationBufferMemory, ChatMessageHistory, BaseMemory,
};
use langchainrust::schema::Message;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 中级示例 2: 记忆与多轮对话 ===\n");
    
    // 1. 创建 LLM
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.7),
        max_tokens: Some(500),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm = OpenAIChat::new(config);
    
    // 2. 创建聊天历史
    let mut chat_history = ChatMessageHistory::new();
    
    println!("开始多轮对话\n");
    println!("{}", "=".repeat(50));
    
    // 3. 对话循环
    let mut turn = 0;
    
    // 模拟几轮对话
    let test_inputs = vec![
        "你好，我叫小明",
        "我刚才告诉你我叫什么？",
        "我喜欢编程，特别是 Rust 语言",
        "我之前说我喜欢什么编程语言？",
    ];
    
    for user_input in test_inputs {
        turn += 1;
        println!("\n[轮次 {}]", turn);
        println!("用户: {}", user_input);
        
        // 构建消息
        let mut messages = vec![
            Message::system("你是一个友好的助手。请记住用户告诉你的信息。"),
        ];
        
        // 添加历史
        messages.extend(chat_history.messages().iter().cloned());
        
        // 添加当前问题
        messages.push(Message::human(user_input));
        
        // 调用 LLM
        match llm.chat(messages, None).await {
            Ok(response) => {
                println!("助手: {}", response.content);
                
                // 保存到历史
                chat_history.add_message(Message::human(user_input));
                chat_history.add_message(Message::ai(&response.content));
            }
            Err(e) => {
                eprintln!("错误: {}", e);
                eprintln!("提示: 请确保设置了正确的 OPENAI_API_KEY 环境变量");
                break;
            }
        }
    }
    
    // 4. 显示完整记忆
    println!("\n{}\n", "=".repeat(50));
    println!("完整对话历史:");
    for (i, msg) in chat_history.messages().iter().enumerate() {
        let role = match msg.message_type {
            langchainrust::schema::MessageType::Human => "用户",
            langchainrust::schema::MessageType::AI => "助手",
            langchainrust::schema::MessageType::System => "系统",
            langchainrust::schema::MessageType::Tool { .. } => "工具",
        };
        println!("  [{}] {}: {}", i + 1, role, msg.content.chars().take(50).collect::<String>());
    }
    
    // 5. 使用 ConversationBufferMemory (演示 API)
    println!("\n{}\n", "=".repeat(50));
    println!("演示 ConversationBufferMemory API:");
    
    let mut memory = ConversationBufferMemory::new()
        .with_return_messages(true);
    
    // 保存上下文
    let inputs = HashMap::from([
        ("input".to_string(), "你好".to_string()),
    ]);
    let outputs = HashMap::from([
        ("output".to_string(), "你好！很高兴见到你！".to_string()),
    ]);
    
    memory.save_context(&inputs, &outputs).await?;
    
    // 加载记忆
    let memory_vars = memory.load_memory_variables(&HashMap::new()).await?;
    println!("记忆变量: {:?}", memory_vars.keys().collect::<Vec<_>>());
    
    println!("\n=== 示例完成 ===");
    Ok(())
}