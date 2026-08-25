//! Document loader example
//!
//! Shows TextLoader loading a text file and parsing its metadata (no API key required).
//!
//! # Run
//! ```bash
//! cargo run --example rag_document_loaders
//! # or specify a file
//! cargo run --example rag_document_loaders -- README.md
//! ```

use langchainrust::{DocumentLoader, TextLoader};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "README.md".to_string());
    let loader = TextLoader::new(PathBuf::from(path));
    let docs = loader.load().await?;

    println!("Loaded {} documents", docs.len());
    for (i, doc) in docs.iter().enumerate() {
        let preview: String = doc.content.chars().take(200).collect();
        println!("--- document {} ---", i + 1);
        println!("Content preview: {}", preview);
        println!("metadata: {:?}", doc.metadata);
    }
    Ok(())
}
