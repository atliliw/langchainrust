//! CorrectiveRAG example
//!
//! Shows CorrectiveRAGAgent's retrieve → grade → correct → hallucination-check flow.
//!
//! # Run
//! ```bash
//! cargo run --example rag_corrective_rag
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::embeddings::{Embeddings, MockEmbeddings};
use langchainrust::retrieval::SimilarityRetriever;
use langchainrust::tools::DuckDuckGoSearchTool;
use langchainrust::vector_stores::{InMemoryVectorStore, VectorStore};
use langchainrust::{CorrectiveRAGAgent, Document, OpenAIChat, OpenAIConfig};
use std::sync::Arc;

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

    // 2. Prepare the retriever (InMemoryVectorStore + MockEmbeddings, for demo purposes)
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(3));

    // Add documents to the vector store first
    let docs = vec![
        Document::new("Rust is a systems programming language developed by Mozilla, focused on safety and performance."),
        Document::new("Rust's core features include the ownership system, borrow checking, and zero-cost abstractions."),
        Document::new("Rust's package manager is called Cargo; it supports dependency management and build automation."),
    ];
    let doc_texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let doc_embeddings = embeddings.embed_documents(&doc_texts).await?;
    store.add_documents(docs, doc_embeddings).await?;

    let retriever = SimilarityRetriever::new(store, embeddings);

    // 3. Create the CRAG Agent
    let agent = CorrectiveRAGAgent::new(llm, retriever)
        .with_grade_threshold(0.5) // documents scored below 0.5 trigger a correction
        .with_web_fallback(Box::new(DuckDuckGoSearchTool::new())) // optional: web search fallback
        .with_hallucination_check(true); // optional: hallucination check

    // 4. Query
    let result = agent.invoke("What are the core features of Rust?").await?;

    println!("Answer: {}", result.answer);
    println!("Grounded in documents: {}", result.grounded);
    println!("Source document count: {}", result.sources.len());
    for (i, score) in result.grade_scores.iter().enumerate() {
        println!("  document[{}] score: {:.2}", i, score);
    }

    Ok(())
}
