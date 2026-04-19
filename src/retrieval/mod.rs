// src/retrieval/mod.rs
//! 检索模块
//!
//! 提供文档检索和文本分割功能。

mod retriever;
mod splitter;
mod loaders;
pub mod bm25;
pub mod hybrid;
pub mod chunked_hybrid;
pub mod unified_hybrid;

pub use retriever::{Retriever, SimilarityRetriever, RetrieverTrait};
pub use splitter::{TextSplitter, RecursiveCharacterSplitter};
pub use loaders::{PDFLoader, CSVLoader, DocumentLoader, LoaderError};

pub use bm25::{BM25Retriever, BM25Index, BM25Params, Tokenizer, ChunkedBM25Retriever, ChunkedSearchResult, AutoMergingConfig};

pub use hybrid::{HybridRetriever, RetrievedDocument, RetrievalSource, reciprocal_rank_fusion};
pub use chunked_hybrid::ChunkedHybridRetriever;
pub use unified_hybrid::{UnifiedHybridIndex, HybridIndexConfig, HybridSearchResult};

pub use crate::vector_stores::{Document, SearchResult, VectorStore, InMemoryVectorStore};
pub use crate::vector_stores::{DocumentStore, InMemoryDocumentStore, ChunkedDocumentStore, ChunkDocument, ChunkedVectorStore};
pub use crate::embeddings::{Embeddings, MockEmbeddings, OpenAIEmbeddings, cosine_similarity};