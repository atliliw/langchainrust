//! Handoffs example
//!
//! Shows how to use HandoffManager to switch between multiple agents.
//!
//! # Run
//! ```bash
//! cargo run --example agent_handoffs
//! ```

// This example only shows the usage guide; it does not need to import any LLM type

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Handoffs example ===\n");

    // Handoffs let you switch between multiple specialized agents
    println!("Handoffs workflow:");
    println!("1. The main agent receives the user's request");
    println!("2. It decides which specialization is needed");
    println!("3. It hands the conversation off to the specialized agent");
    println!("4. The specialized agent can hand control back to the main agent when done");

    println!("\nExample scenarios:");
    println!("- Support agent → Technical support agent → back to support");
    println!("- General assistant → Code expert → Documentation expert");
    println!("- Sales advisor → Product expert → After-sales service");

    println!("\nHandoffManager usage:");
    println!("  let manager = HandoffManager::new();");
    println!("  manager.register_agent(\"tech\", tech_executor)?;");
    println!("  manager.register_agent(\"docs\", docs_executor)?;");
    println!("  manager.set_primary(\"tech\")?;");
    println!("  let handoff_tools = manager.handoff_tools();");

    println!("\nNote: set the OPENAI_API_KEY environment variable to make real calls.");
    Ok(())
}
