//! What: event-sourced session with a durable file checkpoint, recovery and fork.
//! Run: cargo run -p langchainrust --example sessions_durable
//! Env: OPENAI_API_KEY (optional: OPENAI_MODEL)

use std::sync::Arc;

use langchainrust::sessions::{EventSessionManager, FileCheckpoint, MemoryEventStore};
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_dir = std::env::temp_dir().join("lc-sessions-durable-example");
    let store = Arc::new(MemoryEventStore::new());
    let manager = EventSessionManager::new(store)
        .with_checkpoint(Arc::new(FileCheckpoint::new(checkpoint_dir)));

    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
    let session_id = manager.create_session().await?;

    let reply = manager
        .chat(&session_id, &llm, "Hello, remember this turn.".to_string())
        .await?;
    println!("reply: {reply}");

    let history = manager.history(&session_id).await?;
    println!("history: {} messages", history.len());

    // Persist a snapshot, then recover from the checkpoint plus the event tail.
    manager.save_checkpoint(&session_id).await?;
    let recovered = manager.recover_session(&session_id).await?;
    println!("recovered: {} messages", recovered.messages.len());

    // Fork a branch from the current state without touching the trunk.
    let forked = manager.fork_session(&session_id, "experiment", None).await?;
    let forked_history = forked.history(&session_id).await?;
    println!("forked branch: {} messages", forked_history.len());

    Ok(())
}
