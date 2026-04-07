// src/retrieval/mod.rs
//! 检索模块
//!
//! 提供文档检索和文本分割功能。

mod retriever;
mod splitter;
mod loaders;

pub use retriever::{Retriever, SimilarityRetriever, RetrieverTrait};
pub use splitter::{TextSplitter, RecursiveCharacterSplitter};
pub use loaders::{PDFLoader, CSVLoader, DocumentLoader, LoaderError};

// 重新导出 vector_stores 和 embeddings 的类型
pub use crate::vector_stores::{Document, SearchResult, VectorStore, InMemoryVectorStore};
pub use crate::embeddings::{Embeddings, MockEmbeddings, OpenAIEmbeddings, cosine_similarity};