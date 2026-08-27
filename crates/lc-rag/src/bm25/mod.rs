// src/retrieval/bm25/mod.rs
//! BM25 retrieval module
//!
//! BM25 (Best Match 25) is a classic TF-IDF weighted retrieval algorithm,
//! suited to keyword search, long-document retrieval, and similar scenarios.

mod algorithm;
mod chunked;
mod retriever;
mod tokenizer;

pub use algorithm::{bm25_score, compute_idf, BM25Params};
pub use chunked::{
    AutoMergingConfig, ChunkedBM25Index, ChunkedBM25Retriever, ChunkedIndexData,
    ChunkedSearchResult,
};
pub use lc_vector_stores::document_store::ChunkDocument;
pub use retriever::BM25Retriever;
pub use tokenizer::Tokenizer;
