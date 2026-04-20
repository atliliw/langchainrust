// src/retrieval/mod.rs
mod retriever;
mod splitter;
mod loaders;
pub mod bm25;
pub mod hybrid;
pub mod chunked_hybrid;
pub mod unified_hybrid;
pub mod multi_query;
pub mod hyde;
pub mod reranking;

pub use retriever::{Retriever, SimilarityRetriever, RetrieverTrait, RetrieverError};
pub use splitter::{TextSplitter, RecursiveCharacterSplitter};
pub use loaders::{PDFLoader, CSVLoader, TextLoader, JSONLoader, MarkdownLoader, DocumentLoader, LoaderError};

pub use bm25::{BM25Retriever, BM25Index, BM25Params, Tokenizer, ChunkedBM25Retriever, ChunkedSearchResult, AutoMergingConfig};

pub use hybrid::{HybridRetriever, RetrievedDocument, RetrievalSource, reciprocal_rank_fusion};
pub use chunked_hybrid::ChunkedHybridRetriever;
pub use unified_hybrid::{UnifiedHybridIndex, HybridIndexConfig, HybridSearchResult};

pub use multi_query::{MultiQueryRetriever, MultiQueryConfig, MultiQueryError, StaticQueryGenerator};

pub use hyde::{HyDERetriever, HyDEConfig, HyDEError};

pub use reranking::{Reranker, KeywordReranker, BM25Reranker, RerankingExecutor, RerankingConfig, RerankingError};

pub use crate::vector_stores::{Document, SearchResult, VectorStore, InMemoryVectorStore};
pub use crate::vector_stores::{DocumentStore, InMemoryDocumentStore, ChunkedDocumentStore, ChunkDocument, ChunkedVectorStore};
pub use crate::embeddings::{Embeddings, MockEmbeddings, OpenAIEmbeddings, cosine_similarity};