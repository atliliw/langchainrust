// lc-rag/examples/rag_pipeline_main.rs
//! T6-2 (v0.26) `RAGPipeline` flagship demo — one pipeline that stitches
//! vector local retrieval + rerank, with GraphRAG global search as a second leg.
//!
//! Two retrieval legs over the SAME small corpus:
//!
//!   1. **vector local leg**: `InMemoryVectorStore` + a mock `Embeddings`
//!      (`MockEmbeddings`, hash-based, offline) are wrapped in a
//!      [`SimilarityRetriever`] and handed to
//!      [`RAGPipelineBuilder`](crate::RAGPipelineBuilder) as the explicit
//!      `.retriever(...)`. The builder prefers the injected retriever and never
//!      needs its own embeddings/store fallback. `query_with_sources` returns an
//!      answer (from a mock `BaseChatModel`) plus the retrieved `sources`.
//!   2. **rerank leg**: the retrieved `sources` are re-scored by a mock
//!      [`Reranker`] through [`RerankingExecutor::rerank_documents`], proving the
//!      reorder is real (a query word in one doc makes it jump ahead).
//!
//! Everything is mocked so the demo runs fully offline:
//!
//!   ```text
//!   cargo run -p lc-rag --example rag_pipeline_main
//!   ```
//!
//! # GraphRAG global leg (not executed here)
//!
//! The same corpus can be indexed into a knowledge graph for *whole-store*
//! overviews that vector local search cannot give — a broad map question like
//! "what topics does this corpus cover" instead of a specific-hit question:
//!
//! ```text
//! let graph = GraphRAG::new(real_llm)          // any BaseChatModel
//!     .with_config(GraphRAGConfig::new());     // optional
//! graph.add_documents(&docs).await?;           // entity/relation extraction
//! graph.build_communities().await?;            // Leiden community detection
//! let res = graph.query(q, QueryMode::Global).await?;
//! // res.answer / res.sources are a corpus-wide synthesis
//! ```
//!
//! This leg needs a **real LLM** for entity extraction + community summarization
//! (and is network-bound), so the offline example deliberately does not run it —
//! swap in any `BaseChatModel` (e.g. `OpenAIChat`) to turn it on.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::Stream;
use lc_core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult, StreamChunk};
use lc_core::runnables::{Runnable, RunnableConfig};
use lc_providers::ProviderError;
use lc_rag::reranking::{Reranker, RerankingError, RerankingExecutor};
use lc_rag::{
    Document, Embeddings, InMemoryVectorStore, MockEmbeddings, RAGPipelineBuilder,
    SimilarityRetriever, VectorStore,
};
use lc_schema::Message;

/// Mock chat model: echoes whether the query keyword reached it via the context,
/// proving the retrieval path delivered sources to the generator.
#[derive(Debug)]
struct MockChat;

#[derive(Debug)]
struct MockChatError(String);

impl std::fmt::Display for MockChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mock chat error: {}", self.0)
    }
}

impl std::error::Error for MockChatError {}

impl From<MockChatError> for ProviderError {
    fn from(e: MockChatError) -> Self {
        ProviderError::Config(e.to_string())
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for MockChat {
    type Error = MockChatError;

    async fn invoke(
        &self,
        input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        self.chat(input, _config).await
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChat {
    fn model_name(&self) -> &str {
        "mock-chat"
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.len() / 4
    }

    fn with_temperature(self, _temp: f32) -> Self
    where
        Self: Sized,
    {
        self
    }

    fn with_max_tokens(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }
}

#[async_trait]
impl BaseChatModel for MockChat {
    async fn chat(
        &self,
        messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        // The pipeline assembles [system, human]; the human message carries the
        // retrieved context.
        let last = messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok(LLMResult {
            content: format!(
                "Found {} sources; context mentions 'revenue': {}.",
                count_sources(&last),
                last.contains("revenue")
            ),
            model: "mock-chat".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Self::Error>> + Send>>, Self::Error>
    {
        Err(MockChatError("no stream".to_string()))
    }
}

/// Counts the bracketed sources of the assembled context (`[1] ... [2] ...`).
fn count_sources(context: &str) -> usize {
    context.matches("[").count()
}

/// Mock reranker: scores each document by how many distinct query words it
/// shares. Pure, deterministic, offline — exists only to make the reorder real.
struct MockReranker;

impl Reranker for MockReranker {
    fn score(
        &self,
        query: &str,
        documents: &[lc_rag::Document],
    ) -> Result<Vec<f32>, RerankingError> {
        let terms: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 1)
            .map(str::to_lowercase)
            .collect();
        Ok(documents
            .iter()
            .map(|d| {
                let lower = d.content.to_lowercase();
                terms.iter().filter(|t| lower.contains(t.as_str())).count() as f32
            })
            .collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- Corpus: a small set of short texts with distinguishable keywords. ----
    let docs = vec![
        Document::new("Acme revenue doubled after the new industrial adhesive launch."),
        Document::new("The company's CFO presented the quarterly earnings to investors."),
        Document::new("Rust is a systems programming language loved by backend engineers."),
    ];

    // ---- 1. Vector local leg: retrieve via a SimilarityRetriever. ----
    // The store + mock embeddings are wrapped in a retriever, which is injected
    // explicitly; the builder needs no embeddings/store fallback of its own.
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    let embeddings: Arc<dyn Embeddings> = Arc::new(MockEmbeddings::new(8));
    let retriever = SimilarityRetriever::new(store, embeddings);

    let rag = RAGPipelineBuilder::new()
        .llm(MockChat)
        .retriever(retriever)
        .retrieve_k(3)
        .system("You are a concise research assistant.")
        .build()?;

    rag.index_documents(docs.clone()).await?;

    let question = "How did Acme revenue perform?";
    let result = rag.query_with_sources(question).await?;

    println!("== Vector local leg ==");
    println!("question : {question}");
    println!("answer   : {}", result.answer);
    println!("sources  :");
    for (i, s) in result.sources.iter().enumerate() {
        println!("  [{}] {}", i + 1, s.content);
    }

    // ---- 2. Rerank leg: reorder the retrieved sources by a mock reranker. ----
    // Give the query a word that lives in source 3 ("revenue" also appears there
    // via MockEmbeddings score noise) and in sources 1+2 — the mock reranker will
    // lift the docs sharing the most query words to the top.
    let query = "revenue and earnings";
    let reranked = RerankingExecutor::new(Box::new(MockReranker))
        .with_top_n(2)
        .rerank_documents(query, result.sources.clone())?;

    println!("\n== Rerank leg ==");
    println!("query : {query}");
    for (i, r) in reranked.iter().enumerate() {
        println!("  [{}] score={:.3}: {}", i + 1, r.score, r.document.content);
    }
    assert!(
        reranked.len() <= 2,
        "rerank must keep only the top 2 sources"
    );

    println!("\n== GraphRAG global leg (skeleton only, see doc header) ==");
    println!(
        "GraphRAG::new(llm) -> add_documents -> build_communities -> query(q, QueryMode::Global)"
    );
    println!("skipped offline: needs a real LLM. Pipeline demo done.");

    Ok(())
}
