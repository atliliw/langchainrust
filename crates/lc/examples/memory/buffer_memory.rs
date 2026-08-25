//! Buffer memory example
//!
//! Shows ConversationBufferMemory storing a multi-turn conversation (no API key required).
//!
//! # Run
//! ```bash
//! cargo run --example memory_buffer_memory
//! ```

use langchainrust::memory::BaseMemory;
use langchainrust::ConversationBufferMemory;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = ConversationBufferMemory::new();

    for i in 1..=3 {
        let inputs = HashMap::from([("input".to_string(), format!("question {}", i))]);
        let outputs = HashMap::from([("output".to_string(), format!("answer {}", i))]);
        memory.save_context(&inputs, &outputs).await?;
        println!(
            "round {} saved, {} messages in total",
            i,
            memory.chat_memory().len()
        );
    }

    let loaded = memory.load_memory_variables(&HashMap::new()).await?;
    println!("\nLoaded history:\n{}", loaded.get("history").unwrap());
    Ok(())
}
