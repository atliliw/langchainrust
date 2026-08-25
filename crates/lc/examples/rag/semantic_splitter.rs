//! Semantic Splitter example
//!
//! Shows how to split text into semantic chunks with SemanticSplitter.
//!
//! # Run
//! ```bash
//! cargo run --example semantic_splitter
//! ```

use langchainrust::retrieval::TextSplitter;
use langchainrust::{MockEmbeddings, RecursiveCharacterSplitter, SemanticSplitter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Semantic Splitter example ===\n");

    let text = r#"Rust is a systems programming language. It focuses on memory safety and concurrency.
Rust's ownership system is its most distinctive feature. The compiler checks ownership rules at compile time, avoiding runtime errors.
Python is an interpreted language. It is known for being concise and readable, widely used in data science and AI.
LangChain is a framework. It helps developers build applications on top of large language models."#;

    // Create a semantic splitter (needs an embedding model, similarity threshold 0.5, max 200 chars per chunk)
    let embeddings = MockEmbeddings::new(4);
    let splitter = SemanticSplitter::new(embeddings, 0.5, 200);

    let chunks = splitter.split_text(text).await?;
    println!("The text is split into {} semantic chunks:\n", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("--- chunk {} ---", i + 1);
        println!("{}\n", chunk.trim());
    }

    // Comparison: a traditional recursive splitter
    let recursive = RecursiveCharacterSplitter::new(100, 20);
    let rec_chunks = recursive.split_text(text);
    println!(
        "\nComparison: the recursive splitter produces {} chunks (fixed length, may cut semantics)",
        rec_chunks.len()
    );

    Ok(())
}
