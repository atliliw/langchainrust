//! BM25 keyword search example
//!
//! Shows ChunkedBM25Retriever keyword search (no LLM or vector store required).
//!
//! # Run
//! ```bash
//! cargo run --example rag_bm25_search
//! ```

use langchainrust::retrieval::bm25::ChunkedBM25Retriever;
use langchainrust::retrieval::ChunkedDocumentStore;
use langchainrust::Document;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(ChunkedDocumentStore::new());
    let mut retriever = ChunkedBM25Retriever::new(store);

    retriever
        .add_documents(vec![
            Document::new("Rust is a systems programming language developed by Mozilla, focused on safety and performance.")
                .with_id("rust_intro"),
            Document::new("Rust's core features include the ownership system, borrow checking, and zero-cost abstractions.")
                .with_id("rust_features"),
            Document::new("Machine learning is the core technology of AI, letting computers learn from data.")
                .with_id("ml_def"),
        ])
        .unwrap();

    let results = retriever.search("Rust language features", 3);
    println!("Search results (sorted by relevance):");
    for (i, r) in results.iter().enumerate() {
        println!("  [{}] {}", i + 1, r.content());
    }
    Ok(())
}
