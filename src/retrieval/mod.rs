// src/retrieval/mod.rs
//! 检索模块
//!
//! 提供文档检索和文本分割功能。

mod retriever;
mod splitter;
mod loaders;
pub mod bm25;

pub use retriever::{Retriever, SimilarityRetriever, RetrieverTrait};
pub use splitter::{TextSplitter, RecursiveCharacterSplitter};
pub use loaders::{PDFLoader, CSVLoader, DocumentLoader, LoaderError};

// BM25 检索器
pub use bm25::{BM25Retriever, BM25Index, BM25Params, Tokenizer};

// 重新导出 vector_stores 和 embeddings 的类型
pub use crate::vector_stores::{Document, SearchResult, VectorStore, InMemoryVectorStore};
pub use crate::embeddings::{Embeddings, MockEmbeddings, OpenAIEmbeddings, cosine_similarity};