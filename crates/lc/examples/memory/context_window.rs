//! ContextWindow example
//!
//! Shows how to manage long-context conversations with ContextWindow.
//!
//! # Run
//! ```bash
//! cargo run --example memory_context_window
//! ```

use langchainrust::{ContextWindow, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ContextWindow example ===\n");

    // 1. Truncate strategy: drop the oldest messages when over the limit
    //    ContextWindow<OpenAIChat> is the default generic; the Truncate strategy needs no LLM
    let cw: ContextWindow<langchainrust::OpenAIChat> = ContextWindow::new(4096)?;

    let messages = vec![
        Message::system("You are an assistant"),
        Message::human("First question"),
        Message::ai("First answer"),
        Message::human("Second question"),
        Message::ai("Second answer"),
        Message::human("Latest question"),
    ];

    let fitted = cw.fit(messages.clone()).await?;
    println!(
        "Truncate strategy: {} messages → {} messages (within 4096 tokens)",
        messages.len(),
        fitted.len()
    );

    // 2. Summarize strategy: compress old messages with the LLM when over the limit
    println!("\nSummarize strategy:");
    println!("  let llm = OpenAIChat::new(config);");
    println!("  let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm));");
    println!("  let fitted = cw.fit(messages).await?;");
    println!("\nWorkflow:");
    println!("  1. Count the total tokens of the messages");
    println!("  2. If over the limit, find the split point that keeps the newest messages");
    println!("  3. Use the LLM to compress the old messages into a summary");
    println!("  4. Return: system + [summary] + newest messages");

    // 3. Summarize with a custom prompt
    println!("\nCustom summary prompt:");
    println!("  let cw = ContextWindow::with_strategy(");
    println!("      4096,");
    println!(
        "      Strategy::summarize_with_prompt(llm, \"Summarize in English: {{conversation}}\\nSummary:\"),"
    );
    println!("  );");

    Ok(())
}
