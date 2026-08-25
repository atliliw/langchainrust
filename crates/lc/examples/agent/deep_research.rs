//! Deep Research Agent example
//!
//! Shows multi-round deep research with DeepResearchAgent:
//! split into subtopics → search → synthesize → find gaps → search again → cited report.
//!
//! # Run
//! ```bash
//! cargo run --example agent_deep_research
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::tools::DuckDuckGoSearchTool;
use langchainrust::{DeepResearchAgent, OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure the LLM
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

    // 2. Create the Deep Research Agent
    let agent = DeepResearchAgent::new(llm)
        .with_searcher(Box::new(DuckDuckGoSearchTool::new()))
        .with_max_rounds(3) // at most 3 search rounds
        .with_max_subtopics(5); // at most 5 subtopics

    // 3. Run the deep research
    let report = agent
        .research("Rust async runtime comparison: tokio vs async-std vs smol")
        .await?;

    // 4. Print the report
    println!("=== Deep Research Report ===\n");
    println!("{}", report.markdown);

    println!("\n--- Citations ---");
    for citation in &report.citations {
        println!(
            "[{}] {} {}",
            citation.index,
            citation.source,
            citation
                .url
                .as_ref()
                .map(|u| format!("({})", u))
                .unwrap_or_default()
        );
        println!("  {}", citation.snippet);
    }

    println!("\n--- Statistics ---");
    println!("Subtopics: {:?}", report.subtopics);
    println!("Research rounds: {}", report.rounds_completed);

    Ok(())
}
