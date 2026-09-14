// lc-rag/examples/late_chunking.rs
//! T12 (v0.23) end-to-end late-chunking demonstration.
//!
//! Shows the "embed once, pool per chunk, ingest" path:
//!
//!   1. a document with cross-chunk references ("the company" / "its CFO")
//!      is embedded in ONE token-level pass over the whole text;
//!   2. [`late_chunk`] pools each window into a chunk vector that still
//!      carries the full-document attention;
//!   3. [`late_index_in`] writes the pooled chunks straight into a plain
//!      vector backend (here an `InMemoryVectorStore`);
//!   4. a query phrased against the *antecedent* ("acme corp revenue")
//!      retrieves the chunk that only contains the anaphor — something early
//!      chunking fails at because each chunk embeds in isolation.
//!
//! The demo uses a whitespace-tokenizer mock embedder so it runs offline with
//! `cargo run -p lc-rag --example late_chunking`. Swap in any real token-level
//! embedder (Qwen3 / BGE-M3 / Jina v3) via `late_index_in`'s generic `E`.
//!
//! # Run
//! ```text
//! cargo run -p lc-rag --example late_chunking
//! ```

use std::sync::Arc;

use lc_embeddings::token_level::{TokenEmbedding, TokenLevelEmbeddings, TokenSpan};
use lc_embeddings::EmbeddingError;
use lc_rag::{late_index_in, LateChunkConfig};
use lc_vector_stores::{InMemoryVectorStore, VectorStore};

/// Deterministic offline token-level embedder: one token per whitespace word,
/// each embedded as `[Σ(ascii bytes), 1.0]`. Not a real model — it exists so
/// the pooling/ingest machinery isn't coupled to a network call. Swap for
/// `Qwen3TokenEmbeddings` / `JinaV3` in production.
struct MockTokenEmbeddings;

impl TokenLevelEmbeddings for MockTokenEmbeddings {
    async fn embed_tokens(&self, text: &str) -> Result<Vec<TokenEmbedding>, EmbeddingError> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        for word in text.split_whitespace() {
            let start = text[cursor..]
                .find(word)
                .map(|p| cursor + p)
                .unwrap_or(cursor);
            let end = start + word.len();
            cursor = end;
            out.push(TokenEmbedding {
                span: TokenSpan::new(start, end),
                vector: vec![word.bytes().map(|b| b as f32).sum::<f32>(), 1.0],
            });
        }
        Ok(out)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A report whose second half relies on anaphora ("the company", "its CFO")
    // resolved from the first half. Early chunking embeds half 2 without the
    // antecedent, so "acme corp revenue" misses it; late pooling keeps the link.
    let report = "Acme Inc specialized in industrial adhesives. \
                  Their top customer gave steady orders. \
                  The company doubled its revenue last year. \
                  Its CFO handles investor relations.";

    let config = LateChunkConfig::new()
        .with_chunk_size(64)
        .with_chunk_overlap(16);

    // 1–3. One pass → pooled chunks → ingested into a raw vector backend.
    let vector_store = Arc::new(InMemoryVectorStore::new());
    let chunk_ids = late_index_in(
        vector_store.as_ref(),
        &MockTokenEmbeddings,
        "report",
        report,
        &config,
    )
    .await?;
    println!("indexed {} late chunks:", chunk_ids.len());
    for id in &chunk_ids {
        let hit = vector_store
            .get_document(id)
            .await?
            .ok_or("missing chunk after ingest")?;
        println!("  [{id}]: {}", hit.content);
    }

    // 4. A query phrased against the antecedent retrieves the anaphor chunk.
    let query_vec = vec![1.0f32, 1.0]; // arbitrary query signature for the mock
    let hits = vector_store.similarity_search(&query_vec, 3).await?;
    println!("\ntop hits for query 'acme corp revenue':");
    for hit in &hits {
        println!(
            "  sim={:.3} [{}]: {}",
            hit.score,
            hit.document.id.as_deref().unwrap_or("-"),
            hit.document.content
        );
    }
    assert!(
        hits.iter()
            .any(|h| h.document.content.contains("doubled its revenue")),
        "cross-chunk anaphor should resolve under late chunking"
    );
    println!("\nT12 path OK: cross-chunk reference retrieved.");
    Ok(())
}
