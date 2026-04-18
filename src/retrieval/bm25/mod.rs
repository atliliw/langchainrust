// src/retrieval/bm25/mod.rs
//! BM25 检索模块
//!
//! BM25 (Best Match 25) 是一种经典的 TF-IDF 加权检索算法，
//! 适用于关键词搜索、长文档检索等场景。

mod algorithm;
mod index;
mod tokenizer;
mod retriever;

pub use algorithm::{bm25_score, compute_idf, BM25Params};
pub use index::BM25Index;
pub use tokenizer::Tokenizer;
pub use retriever::BM25Retriever;