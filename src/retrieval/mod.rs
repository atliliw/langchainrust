// src/retrieval/mod.rs
pub mod bm25;
pub mod chunked_hybrid;
pub mod graph_rag;
pub mod hybrid;
pub mod hyde;
mod loaders;
pub mod multi_query;
pub mod reranking;
mod retriever;
mod semantic_splitter;
mod splitter;
pub mod unified_hybrid;

pub use loaders::{
    CSVLoader, DocumentLoader, DocxLoader, HTMLLoader, JSONLoader, LoaderError, MarkdownLoader,
    PDFLoader, SitemapLoader, TextLoader, WebScraperLoader,
};
pub use retriever::{Retriever, RetrieverError, RetrieverTrait, SimilarityRetriever};
pub use semantic_splitter::SemanticSplitter;
pub use splitter::{RecursiveCharacterSplitter, TextSplitter};

pub use bm25::{
    AutoMergingConfig, BM25Index, BM25Params, BM25Retriever, ChunkedBM25Retriever,
    ChunkedSearchResult, Tokenizer,
};

pub use chunked_hybrid::ChunkedHybridRetriever;
pub use hybrid::{reciprocal_rank_fusion, HybridRetriever, RetrievalSource, RetrievedDocument};
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

pub use crate::embeddings::{cosine_similarity, Embeddings, MockEmbeddings, OpenAIEmbeddings};
pub use crate::vector_stores::{
    ChunkDocument, ChunkedDocumentStore, ChunkedVectorStore, DocumentStore, InMemoryDocumentStore,
};
pub use crate::vector_stores::{Document, InMemoryVectorStore, SearchResult, VectorStore};
