#![warn(missing_docs)]
// lc-rag/src/lib.rs
//! RAG (Retrieval-Augmented Generation) module for LangChainRust.
//!
//! Provides document loaders, text splitters, retrievers, BM25 search,
//! hybrid retrieval, GraphRAG, HyDE, multi-query, reranking, and
//! a full RAG pipeline builder.

pub mod adapter;
pub mod bm25;
pub mod graph_rag;
pub mod hybrid;
pub mod hyde;
pub mod loaders;
pub mod multi_query;
pub mod parent_document;
pub mod pipeline;
pub mod reranking;
pub mod retriever;
pub mod retriever_runnable;
pub mod self_query;
pub mod semantic_splitter;
pub mod splitter;
mod structured;
pub mod unified_hybrid;

pub use adapter::RagRunnable;
pub use parent_document::ParentDocumentRetriever;
pub use pipeline::{RAGPipeline, RAGPipelineBuilder, RAGQueryResult};
pub use retriever_runnable::RetrieverRunnable;

pub use loaders::{
    CSVLoader, DocumentLoader, DocxLoader, HTMLLoader, JSONLoader, LoaderError, MarkdownLoader,
    PDFLoader, SitemapLoader, TextLoader, WebScraperLoader,
};
pub use retriever::{Retriever, RetrieverError, RetrieverTrait, SimilarityRetriever};
pub use self_query::{SelfQueryArgs, SelfQueryRetriever};
pub use semantic_splitter::SemanticSplitter;
pub use splitter::{RecursiveCharacterSplitter, TextSplitter};

pub use bm25::{
    AutoMergingConfig, BM25Params, BM25Retriever, ChunkedBM25Retriever, ChunkedSearchResult,
    Tokenizer,
};

pub use hybrid::{filter_by_score, reciprocal_rank_fusion, RetrievalSource, RetrievedDocument};
pub use unified_hybrid::{HybridIndexConfig, HybridSearchResult, UnifiedHybridIndex};

pub use multi_query::{
    MultiQueryConfig, MultiQueryError, MultiQueryRetriever, StaticQueryGenerator,
};

pub use hyde::{HyDEConfig, HyDEError, HyDERetriever};

pub use reranking::{
    BM25Reranker, KeywordReranker, Reranker, RerankingConfig, RerankingError, RerankingExecutor,
};

pub use graph_rag::{
    Community as GraphCommunity, Entity as GraphEntity, GraphStore, Relation as GraphRelation,
};
pub use graph_rag::{GraphRAG, GraphRAGConfig, GraphRAGError, GraphRAGResult, QueryMode};

// Re-export key types from dependency crates for convenience
pub use lc_embeddings::{
    cosine_similarity, EmbeddingError, Embeddings, MockEmbeddings, OpenAIEmbeddings,
};
pub use lc_vector_stores::{
    ChunkDocument, ChunkedDocumentStore, ChunkedDocumentStoreTrait, ChunkedVectorStore,
    DocumentStore, InMemoryDocumentStore,
};
pub use lc_vector_stores::{
    Document, InMemoryVectorStore, SearchResult, VectorStore, VectorStoreError,
};
