#![warn(missing_docs)]
// lc-rag/src/lib.rs
//! RAG (Retrieval-Augmented Generation) module for LangChainRust.
//!
//! Provides document loaders, text splitters, retrievers, BM25 search,
//! hybrid retrieval, GraphRAG, HyDE, multi-query, reranking,
//! multimodal (image+text) retrieval, and a full RAG pipeline builder.

pub mod adapter;
pub mod bm25;
pub mod contextual;
pub mod graph_rag;
pub mod hybrid;
pub mod hyde;
pub mod late_chunking;
pub mod loaders;
pub mod mmr;
pub mod multi_query;
pub mod multimodal;
pub mod parent_document;
pub mod pipeline;
pub mod reranking;
pub mod retriever;
pub mod retriever_runnable;
pub mod self_query;
pub mod semantic_cache;
pub mod semantic_splitter;
pub mod splitter;
mod structured;
pub mod unified_hybrid;

pub use adapter::RagRunnable;
pub use contextual::{
    ContextualConfig, ContextualEnhancer, ContextualError, CONTEXTUAL_METADATA_KEY,
};
pub use parent_document::ParentDocumentRetriever;
pub use pipeline::{RAGPipeline, RAGPipelineBuilder, RAGQueryResult};
pub use retriever_runnable::RetrieverRunnable;

pub use loaders::{
    CSVLoader, DocumentLoader, DocxLoader, HTMLLoader, JSONLoader, LoaderError, MarkdownLoader,
    PDFLoader, SitemapLoader, TextLoader, WebScraperLoader,
};
pub use retriever::{Retriever, RetrieverError, RetrieverTrait, SimilarityRetriever};
pub use self_query::{SelfQueryArgs, SelfQueryRetriever};
pub use semantic_cache::{CacheHitKind, CachedRetriever, SemanticCacheConfig, SemanticCacheCore};
pub use semantic_splitter::SemanticSplitter;
pub use splitter::{RecursiveCharacterSplitter, TextSplitter};

pub use bm25::{
    AutoMergingConfig, BM25Params, BM25Retriever, ChunkedBM25Retriever, ChunkedSearchResult,
    Tokenizer,
};

pub use hybrid::{filter_by_score, reciprocal_rank_fusion, RetrievalSource, RetrievedDocument};
// T11 (0.23): pure MMR selection (id, relevance, embedding) used as the vector-side
// diversity post-processor behind `UnifiedHybridIndex::retrieve_mmr`.
pub use mmr::mmr;
pub use unified_hybrid::{FusionMode, HybridIndexConfig, HybridSearchResult, UnifiedHybridIndex};

pub use multi_query::{
    MultiQueryConfig, MultiQueryError, MultiQueryRetriever, StaticQueryGenerator,
};

// B7 (v0.22.4): image+text chunking and cross-modal retrieval over one
// shared vision embedding space.
pub use multimodal::{
    ImageAsset, MediaBlock, ModalityFilter, MultimodalChunkConfig, MultimodalChunker,
    MultimodalRetriever, MM_CAPTION_KEY, MM_KIND_IMAGE, MM_KIND_KEY, MM_KIND_TEXT, MM_MIME_KEY,
    MM_URL_KEY,
};

pub use hyde::{HyDEConfig, HyDEError, HyDERetriever};
pub use late_chunking::{late_chunk, late_index_in, pool_tokens, LateChunk, LateChunkConfig};

pub use reranking::{
    BM25Reranker, KeywordReranker, Reranker, RerankingConfig, RerankingError, RerankingExecutor,
};

pub use graph_rag::{
    Community as GraphCommunity, Entity as GraphEntity, GraphStore, Relation as GraphRelation,
};
pub use graph_rag::{
    GlobalLevel, GraphRAG, GraphRAGConfig, GraphRAGError, GraphRAGResult, QueryMode,
};

// Re-export key types from dependency crates for convenience
pub use lc_embeddings::{
    cosine_similarity, EmbeddingError, Embeddings, MockEmbeddings, OpenAIEmbeddings,
};
pub use lc_embeddings::{ImageInput, VisionEmbeddings};
pub use lc_vector_stores::{
    ChunkDocument, ChunkedDocumentStore, ChunkedDocumentStoreTrait, ChunkedVectorStore,
    DocumentStore, InMemoryDocumentStore,
};
pub use lc_vector_stores::{
    Document, InMemoryVectorStore, SearchResult, VectorStore, VectorStoreError,
};
