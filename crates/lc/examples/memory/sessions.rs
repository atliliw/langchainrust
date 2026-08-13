//! Sessions 会话示例(真实可运行,需 API Key)
//!
//! 展示 `SessionManager` + 窗口记忆(lc-memory)集成:
//! - 多会话独立管理(每个会话独立生命周期);
//! - P2-1 起 `SessionManager` 可挂接 `BaseMemory`,每轮对话由
//!   `ConversationBufferWindowMemory` 压缩历史后再喂给 LLM(而不是全量历史);
//! - 会话历史持久化到内存存储(`MemorySessionStore`)。
//!
//! # 运行
//! ```bash
//! OPENAI_API_KEY=sk-xxx cargo run --example memory_sessions
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选,默认官方)

use langchainrust::sessions::{MemorySessionStore, SessionManager};
use langchainrust::{ConversationBufferWindowMemory, MessageType, OpenAIChat, OpenAIConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    // 窗口记忆:保留最近 k 轮对话(2*k 条消息)。记忆的 input/output key
    // 默认都是 "input"/"output",与 SessionManager 默认对齐,无需额外配置。
    let memory = ConversationBufferWindowMemory::new(3);

    // P2-1: SessionManager 挂接记忆组件后,chat() 的 LLM 上下文由记忆提供。
    let manager = SessionManager::new(Arc::new(MemorySessionStore::new()))
        .with_memory(Arc::new(Mutex::new(memory)));

    let id = manager.create_session().await?;
    println!("会话已创建: {id}\n");

    // 多轮对话:LLM 上下文 = 窗口压缩后的历史 + 本轮用户消息
    let questions = [
        "请记住我叫张三。",
        "我刚才让你记住的名字是什么?",
        "上一轮我提到的人叫什么?",
    ];
    for (i, question) in questions.iter().enumerate() {
        println!("用户({}): {}", i + 1, question);
        let reply = manager.chat(&id, &llm, question.to_string()).await?;
        println!("AI: {}\n", reply);
    }

    // 会话历史:包含每轮的 human/ai 消息(与记忆各自独立)
    let history = manager.history(&id).await?;
    println!("会话历史共 {} 条消息:", history.len());
    for msg in &history {
        let role = match msg.message_type {
            MessageType::Human => "Human",
            MessageType::AI => "AI",
            MessageType::System => "System",
            MessageType::Tool { .. } => "Tool",
        };
        println!("  [{role}] {}", msg.content);
    }
    Ok(())
}
