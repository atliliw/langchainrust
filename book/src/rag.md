# RAG & Retrieval

LangChainRust provides a full Retrieval-Augmented Generation stack: document loaders, splitters, BM25, hybrid search, HyDE, multi-query, reranking, and GraphRAG.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `RAGPipeline` | Pipeline | End-to-end chunk + embed + store + retrieve + generate |
| `BM25Retriever` | Sparse | Classic BM25 keyword search with English + Chinese tokenization |
| `ChunkedBM25Retriever` | Sparse | BM25 with parent-child auto-merging (LlamaIndex-style) |
| `HybridRetriever` | Hybrid | BM25 + vector via Reciprocal Rank Fusion |
| `UnifiedHybridIndex` | Hybrid | Dual-index with auto-splitting and detailed scores |
| `HyDERetriever` | Expansion | Generate hypothetical document, then retrieve with it |
| `MultiQueryRetriever` | Expansion | Generate multiple query variants, merge results |
| `RerankingExecutor` | Reranking | Post-retrieval scoring with `KeywordReranker` or `BM25Reranker` |
| `GraphRAG` | Graph | Entity/relation extraction, community detection, multi-mode query |
| `SemanticSplitter` | Splitter | Embedding-based semantic chunking |
| `RecursiveCharacterSplitter` | Splitter | Character-level recursive splitting |

## RAGPipeline

```rust
use langchainrust::{
    RAGPipelineBuilder, OpenAIChat, OpenAIConfig,
    OpenAIEmbeddings, OpenAIEmbeddingsConfig, InMemoryVectorStore,
};

let rag = RAGPipelineBuilder::new()
    .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
    .embeddings(OpenAIEmbeddings::new(OpenAIEmbeddingsConfig::new("sk-...")))
    .vector_store(InMemoryVectorStore::new())
    .retrieve_k(5)
    .system("Answer based on the provided context.")
    .build()?;

// Or inject any `RetrieverTrait` implementation directly (BM25, hybrid, ...):
// let rag = RAGPipelineBuilder::new()
//     .llm(OpenAIChat::new(OpenAIConfig::new("sk-...")))
//     .retriever(BM25Retriever::new())
//     .build()?;

rag.index_documents(docs).await?;
let answer = rag.query("What is Rust?").await?;
let result = rag.query_with_sources("What is Rust?").await?;
// result.answer, result.sources
```

## Hybrid Retrieval with Reranking

```rust
use langchainrust::{
    BM25Retriever, HybridRetriever, RerankingExecutor, KeywordReranker,
    reciprocal_rank_fusion,
};

// BM25 sparse retrieval
let bm25 = BM25Retriever::new();
bm25.add_documents_sync(docs.clone());

// Vector retrieval (via SimilarityRetriever)
let vector_results: Vec<Document> = retriever.retrieve("query", 10).await?;

// Fuse with Reciprocal Rank Fusion
let hybrid = HybridRetriever::new().with_top_k(10, 10).with_rrf_k(60);
let fused = hybrid.retrieve(bm25_results, vector_results);

// Rerank
let executor = RerankingExecutor::new(Box::new(KeywordReranker::new()))
    .with_top_n(5)
    .with_min_score(0.1);
let reranked = executor.rerank_documents("query", fused_docs)?;
```

## GraphRAG

```rust
use langchainrust::{GraphRAG, GraphRAGConfig, QueryMode as GraphQueryMode};

let graph_rag = GraphRAG::new(llm).with_config(
    GraphRAGConfig::new()
        .with_max_entities_per_doc(10)
        .with_max_relations_per_doc(10),
);

graph_rag.add_documents(&docs).await?;
graph_rag.build_communities().await?;

let result = graph_rag.query("How does X relate to Y?", GraphQueryMode::Local).await?;
// result.answer, result.sources, result.mode
```

## Document Loaders

| Loader | Format |
|--------|--------|
| `TextLoader` | Plain text |
| `CSVLoader` | CSV |
| `MarkdownLoader` | Markdown |
| `HTMLLoader` | HTML |
| `JSONLoader` | JSON |
| `PDFLoader` | PDF |
| `DocxLoader` | Word DOCX |
| `WebScraperLoader` | Web pages |
| `SitemapLoader` | Sitemap XML |

All loaders implement `DocumentLoader` with `async fn load() -> Result<Vec<Document>, LoaderError>`.
