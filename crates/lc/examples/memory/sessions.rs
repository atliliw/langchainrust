//! Sessions example (runnable for real, needs an API key)
//!
//! Shows `SessionManager` integrated with windowed memory (lc-memory):
//! - Multiple sessions managed independently (each session has its own lifecycle);
//! - Since P2-1, `SessionManager` can attach a `BaseMemory`: each turn the
//!   `ConversationBufferWindowMemory` compresses the history before it is fed to the
//!   LLM (instead of the full history);
//! - Session history is persisted in an in-memory store (`MemorySessionStore`).
//!
//! # Run
//! ```bash
//! OPENAI_API_KEY=sk-xxx cargo run --example memory_sessions
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional, default is the official endpoint)

use langchainrust::sessions::{MemorySessionStore, SessionManager};
use langchainrust::{ConversationBufferWindowMemory, MessageType, OpenAIChat, OpenAIConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("please set the OPENAI_API_KEY environment variable");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

    // Windowed memory: keeps the last k turns (2*k messages). The memory's input/output
    // keys default to "input"/"output", matching SessionManager's defaults, so no extra
    // configuration is needed.
    let memory = ConversationBufferWindowMemory::new(3);

    // P2-1: once SessionManager attaches a memory component, chat()'s LLM context
    // comes from the memory.
    let manager = SessionManager::new(Arc::new(MemorySessionStore::new()))
        .with_memory(Arc::new(Mutex::new(memory)));

    let id = manager.create_session().await?;
    println!("Session created: {id}\n");

    // Multi-turn conversation: LLM context = window-compressed history + this turn's user message
    let questions = [
        "Please remember my name is Alice.",
        "What is the name I asked you to remember just now?",
        "What was the name of the person I mentioned in the previous turn?",
    ];
    for (i, question) in questions.iter().enumerate() {
        println!("user({}): {}", i + 1, question);
        let reply = manager.chat(&id, &llm, question.to_string()).await?;
        println!("AI: {}\n", reply);
    }

    // Session history: contains every round's human/ai messages (independent of the memory)
    let history = manager.history(&id).await?;
    println!("Session history has {} messages:", history.len());
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
