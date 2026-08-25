//! FileVectorStore example
//!
//! Shows how to use FileVectorStore for persistent vector storage.
//!
//! # Run
//! ```bash
//! cargo run --example file_vectorstore
//! ```

use langchainrust::{Document, FileVectorStore, MockEmbeddings, VectorStore};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FileVectorStore example ===\n");

    let path = PathBuf::from("./example_vectors.json");
    let dim = 4;

    // Create a file-backed vector store
    let store = FileVectorStore::new(path.clone(), dim).await?;
    println!("Created FileVectorStore: {:?}", path);
    println!("Vector dimension: {}", store.dimension());

    // Add documents
    let docs = vec![
        Document::new("Rust focuses on safety and performance").with_id("rust"),
        Document::new("Python is great for rapid development").with_id("python"),
        Document::new("Go is good for concurrent services").with_id("go"),
    ];

    let _embeddings = MockEmbeddings::new(dim);
    let emb: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    let ids = store.add_documents(docs, emb).await?;
    println!("Added {} documents: {:?}", ids.len(), ids);
    println!("Current document count: {}", store.count().await);

    // Semantic search
    let query = vec![0.9, 0.1, 0.0, 0.0]; // close to "Rust"
    let results = store.similarity_search(&query, 2).await?;
    println!("\nTop 2 search results:");
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.document.content);
    }

    // Persistence check
    println!("\nThe file is automatically persisted to disk.");
    println!("After a restart, create a new FileVectorStore(path, dim) to load the existing data.");

    // Cleanup
    store.clear().await?;
    println!("\nStorage cleared.");

    // Delete the example file
    let _ = std::fs::remove_file(&path);

    Ok(())
}
