//! GraphRAG example
//!
//! Shows GraphRAG knowledge-graph construction plus Local/Global/Hybrid queries.
//!
//! # Run
//! ```bash
//! cargo run --example rag_graph_rag
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::retrieval::graph_rag::{GraphRAG, GraphRAGConfig, QueryMode};
use langchainrust::{Document, OpenAIChat, OpenAIConfig};

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

    // 2. Create the GraphRAG
    let graph_rag = GraphRAG::new(llm).with_config(
        GraphRAGConfig::new()
            .with_max_entities_per_doc(10)
            .with_max_relations_per_doc(10),
    );

    // 3. Add documents (the LLM extracts entities and relations automatically)
    let docs = vec![
        Document::new("Alice is a professor at Tsinghua University, specializing in artificial intelligence."),
        Document::new("Bob is Alice's student, currently researching large language models."),
        Document::new("Charlie is also Alice's student, researching computer vision."),
    ];
    graph_rag.add_documents(&docs).await?;
    println!("Documents added, entities and relations extracted");

    // 4. Build communities (clusters of tightly related entities)
    graph_rag.build_communities().await?;
    println!("Community detection finished");

    // 5. Three query modes
    // Local: searches entity neighbors, good for specific questions
    let local_result = graph_rag
        .query("Who are Alice's students?", QueryMode::Local)
        .await?;
    println!("\n[Local query] Who are Alice's students?");
    println!("Answer: {}", local_result.answer);

    // Global: searches community summaries, good for high-level questions
    let global_result = graph_rag
        .query("What research areas does this knowledge base cover?", QueryMode::Global)
        .await?;
    println!("\n[Global query] Which research areas are covered?");
    println!("Answer: {}", global_result.answer);

    // Hybrid: combines Local + Global
    let hybrid_result = graph_rag
        .query("What is Alice's research group working on?", QueryMode::Hybrid)
        .await?;
    println!("\n[Hybrid query] What is Alice's research group working on?");
    println!("Answer: {}", hybrid_result.answer);

    Ok(())
}
