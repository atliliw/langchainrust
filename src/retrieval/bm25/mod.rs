// src/retrieval/bm25/mod.rs
//! BM25 检索模块
//!
//! BM25 (Best Match 25) 是一种经典的 TF-IDF 加权检索算法，
//! 适用于关键词搜索、长文档检索等场景。

mod algorithm;
mod chunked;
mod index;
mod retriever;
mod tokenizer;

pub use crate::vector_stores::document_store::ChunkDocument;
pub use algorithm::{bm25_score, compute_idf, BM25Params};
pub use chunked::{
    AutoMergingConfig, ChunkedBM25Index, ChunkedBM25Retriever, ChunkedIndexData,
    ChunkedSearchResult,
};
pub use index::BM25Index;
pub use retriever::BM25Retriever;
pub use tokenizer::Tokenizer;
