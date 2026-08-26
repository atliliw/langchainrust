//! AdaptiveRAG example
//!
//! Shows AdaptiveRAG's adaptive retrieval decision:
//! NoRetrieval (chitchat) / SingleSearch (specific question) / MultiQuery (complex question).
//!
//! # Run
//! ```bash
//! cargo run --example rag_adaptive_rag
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)
//! - `OPENAI_BASE_URL`: API base URL (optional)

use langchainrust::embeddings::{Embeddings, MockEmbeddings};
use langchainrust::retrieval::SimilarityRetriever;
use langchainrust::vector_stores::{InMemoryVectorStore, VectorStore};
use langchainrust::{AdaptiveRAG, Document, OpenAIChat, OpenAIConfig};
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

    // 2. Prepare the retriever
    let store = Arc::new(InMemoryVectorStore::new());
    let embeddings = Arc::new(MockEmbeddings::new(3));

    let docs = vec![
        Document::new("Rust is a systems programming language, focused on safety and performance."),
        Document::new("Rust's ownership system avoids data races and null pointers."),
        Document::new("Tokio is the most popular async runtime for Rust."),
        Document::new(
            "async-std is another Rust async runtime with an API closer to the standard library.",
        ),
    ];
    let doc_texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let doc_embeddings = embeddings.embed_documents(&doc_texts).await?;
    store.add_documents(docs, doc_embeddings).await?;

    let retriever = SimilarityRetriever::new(store, embeddings);

    // 3. Create the AdaptiveRAG
    let rag = AdaptiveRAG::new(llm, retriever)
        .with_retrieve_k(4)
        .with_multi_query_count(3);

    // 4. Three query scenarios
    // Scenario 1: chitchat — the LLM decides no retrieval is needed
    let result = rag.invoke("Hi, how is the weather today?").await?;
    print_result("[NoRetrieval] chitchat", &result);

    // Scenario 2: specific question — single retrieval
    let result = rag.invoke("What is Rust's ownership system?").await?;
    print_result("[SingleSearch] specific question", &result);

    // Scenario 3: complex question — multi-angle retrieval
    let result = rag
        .invoke("Compare the scheduling models of Tokio and async-std")
        .await?;
    print_result("[MultiQuery] complex question", &result);

    Ok(())
}

fn print_result(label: &str, result: &langchainrust::AdaptiveRAGResult) {
    println!("\n{}", label);
    println!("  decision: {}", result.decision);
    println!("  answer: {}", result.answer);
    println!("  source document count: {}", result.sources.len());
}
