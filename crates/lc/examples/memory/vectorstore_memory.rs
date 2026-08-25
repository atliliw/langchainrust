//! VectorStore Memory example
//!
//! Shows how to use VectorStoreRetrieverMemory for semantically retrieving past memories.
//!
//! # Run
//! ```bash
//! cargo run --example vectorstore_memory
//! ```

use langchainrust::vector_stores::Document;
use langchainrust::{Embeddings, InMemoryVectorStore, MockEmbeddings, VectorStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VectorStore Memory example ===\n");

    // Create an in-memory vector store
    let store = InMemoryVectorStore::new();
    let embeddings = MockEmbeddings::new(4);

    // Add documents to the vector store
    let docs = vec![
        Document::new("Rust is a systems programming language, focused on safety and performance")
            .with_id("1"),
        Document::new("Python is a scripting language, great for rapid development").with_id("2"),
        Document::new("LangChain is a framework for building LLM applications").with_id("3"),
    ];

    let mut emb_vecs = Vec::new();
    for d in &docs {
        let emb = embeddings.embed_query(&d.content).await?;
        emb_vecs.push(emb);
    }
    let ids = store.add_documents(docs, emb_vecs).await?;
    println!("Added {} documents: {:?}", ids.len(), ids);

    // Semantic search
    let query_emb = embeddings.embed_query("programming language").await?;
    let results = store.similarity_search(&query_emb, 2).await?;
    println!("\nTop 2 search results for 'programming language':");
    for r in &results {
        println!("  [{:.3}] {}", r.score, r.document.content);
    }

    println!("\nVectorStoreRetrieverMemory stores conversation history in a vector store,");
    println!("retrieving relevant memories by semantic similarity for smart long-term recall.");
    Ok(())
}
