# RAG & Retrieval

LangChainRust provides a full Retrieval-Augmented Generation stack: document loaders, splitters, BM25, hybrid search (RRF / weighted fusion), HyDE, multi-query, lexical *and hosted neural* reranking, MMR diversity selection, small-to-big retrieval (sentence-window / parent-document), token-level late chunking, and GraphRAG.

## Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| `RAGPipeline` | Pipeline | End-to-end chunk + embed + store + retrieve + generate |
| `BM25Retriever` | Sparse | Classic BM25 keyword search with English + Chinese tokenization |
| `ChunkedBM25Retriever` | Sparse | BM25 with parent-child auto-merging (LlamaIndex-style) |
| `UnifiedHybridIndex` | Hybrid | Dual-index with auto-splitting and detailed scores |
| `reciprocal_rank_fusion` | Fusion | Combine BM25 + vector results via Reciprocal Rank Fusion |
| `HyDERetriever` | Expansion | Generate hypothetical document, then retrieve with it |
| `MultiQueryRetriever` | Expansion | Generate multiple query variants, merge results |
| `RerankingExecutor` | Reranking | Post-retrieval scoring with `KeywordReranker` or `BM25Reranker` |
| `AsyncReranker` / `CohereRerank` / `JinaRerank` | Neural rerank | Hosted cross-encoder reranking behind one async trait; `rerank_async` helper |
| `mmr` / `retrieve_mmr` | Diversity | Maximal Marginal Relevance: trade relevance against result redundancy |
| `FusionMode::Weighted` | Fusion | Linear `w_bm25·bm25 + w_vec·vector` alternative to RRF |
| `SentenceWindowRetriever` | Small-to-big | Retrieve against single sentences, return a ±N-sentence window |
| `ParentDocumentRetriever` | Small-to-big | Small leaf chunks for recall; any hit returns the whole parent document |
| `late_chunk` / `add_late_chunked_document` | Late chunking | One token-level embedding pass, mean-pool per chunk, inject both legs |
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
    BM25Retriever, RerankingExecutor, KeywordReranker,
    reciprocal_rank_fusion,
};

// BM25 sparse retrieval
let bm25 = BM25Retriever::new();
bm25.add_documents_sync(docs.clone());

// Vector retrieval (via SimilarityRetriever)
let vector_results: Vec<Document> = retriever.retrieve("query", 10).await?;

// Fuse with Reciprocal Rank Fusion
let fused = reciprocal_rank_fusion(bm25_results, vector_results, 60);

// Rerank
let executor = RerankingExecutor::new(Box::new(KeywordReranker::new()))
    .with_top_n(5)
    .with_min_score(0.1);
let reranked = executor.rerank_documents("query", fused_docs)?;
```

## Neural Cross-Encoder Reranking

Lexical rerankers are cheap and offline; when recall quality matters more, a hosted
cross-encoder scores each candidate against the query jointly. Both providers sit
behind one async trait, so swapping providers does not touch call-site code.

```rust
use langchainrust::{CohereRerank, JinaRerank, AsyncReranker, rerank_async};

// Empty key falls back to COHERE_API_KEY; defaults to rerank-multilingual-v3.0.
// JinaRerank::new("") reads JINA_API_KEY (jina-reranker-v2-base-multilingual).
let cohere = CohereRerank::new("")
    .with_model("rerank-v3.5");             // optional
    // .with_base_url("https://api.cohere.com") (also how tests point at a mock)

// Overshoot on recall, then let the cross-encoder pick the final top_n.
let pool = retriever.retrieve_with_scores("query", 20).await?;
let top: Vec<SearchResult> = rerank_async(&cohere, "query", pool, 5).await?;
```

Contract details worth knowing: the helper maps the API's `results[].index` back to
your input positions (provider order is not guaranteed), a malformed response is an
error rather than silently unsorted input, and both clients bypass ambient proxy
settings (`.no_proxy()`) because rerank endpoints are usually called server-to-server.

## MMR Diversity & Weighted Fusion

RRF sorts purely by rank, so one narrow topic can fill the whole top-k. MMR greedily
picks each next item for *relevance minus similarity to what is already selected*,
with a λ dial: `λ=1` is plain relevance, `λ=0` maximizes diversity (0.5–0.7 is the
usual range).

```rust
use langchainrust::{UnifiedHybridIndex, HybridIndexConfig, FusionMode, mmr};

// Weighted linear fusion instead of the default RRF.
let config = HybridIndexConfig::new()
    .with_fusion(FusionMode::Weighted { bm25_weight: 0.3, vector_weight: 0.7 });
let index = UnifiedHybridIndex::with_config(embeddings, store, 1536, config);

// MMR as a retriever method: pool 20 fused candidates, re-embed just those once,
// return a diversified 5.
let picks = index.retrieve_mmr("query", 20, 5, 0.6).await?;

// Or use the pure algorithm over your own (id, relevance, embedding) triples.
let ids: Vec<String> = mmr(&candidates, 0.6, 5);
```

Fusion scores are min-max normalized inside the candidate pool before MMR, so the λ
trade-off means the same thing under RRF and weighted fusion.

## Small-to-Big Retrieval

Embedding one sentence gives a precise hit; answering usually needs the surrounding
context. Two retrievers implement the small-to-big pattern.

```rust
use langchainrust::{SentenceWindowRetriever, ParentDocumentRetriever,
                    InMemoryChunkedDocumentStore};
use std::sync::Arc;

// Sentence window: index single sentences, return ±N sentences around each hit.
let sw = SentenceWindowRetriever::from_documents(docs)
    .with_window(2)      // two sentences on each side (default)
    .with_top_k(3);
let windows = sw.retrieve("query", 3).await?;   // one deduplicated window per source

// Parent document: tiny leaf chunks for recall, the ENTIRE parent doc for the LLM.
let store = Arc::new(InMemoryChunkedDocumentStore::new());
store.add_parent_with_chunks(parent_doc, vec!["leaf 1".into(), "leaf 2".into()]).await?;
let pdr = ParentDocumentRetriever::new(store);
let parents = pdr.retrieve("query", 4).await?;
```

Unlike `ChunkedBM25Retriever`'s hit-ratio-gated auto-merging, `ParentDocumentRetriever`
*always* returns the parent — leaves exist only for matching.

## Late Chunking

Normal chunking embeds each chunk in isolation, so cross-chunk references lose their
antecedent. Late chunking runs one **token-level** embedding pass over the full
document, then mean-pools token vectors inside each chunk window — every chunk vector
carries document-wide context.

```rust
use langchainrust::{late_chunk, LateChunkConfig, UnifiedHybridIndex};

let config = LateChunkConfig::new().with_chunk_size(256).with_chunk_overlap(32);
let chunks = late_chunk(&token_embedder, &document.content, &config).await?;

// Inject one document into BOTH legs at once: pooled vectors into the vector
// index, chunk texts into the BM25/parent store. Deterministic ids make
// re-registering the same parent idempotent.
index.add_late_chunked_document(document, &chunks).await?;
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
