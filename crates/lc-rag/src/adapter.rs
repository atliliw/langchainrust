// lc-rag/src/adapter.rs
//! RagRunnable adapter - bridges RAGPipeline to the Runnable trait.
//!
//! This allows RAG pipelines to participate in LCEL pipelines via `pipe()`.

use async_trait::async_trait;
use lc_core::runnables::{LcelError, Runnable, RunnableConfig};
use std::sync::Arc;

use crate::pipeline::RAGPipeline;

/// Adapter that wraps a `RAGPipeline` as a `Runnable<String, String>`.
///
/// This enables RAG pipelines to participate in LCEL pipelines:
///
/// ```rust,ignore
/// let rag_runnable = RagRunnable::new(Arc::new(pipeline));
/// let pipeline = rag_runnable.pipe(parser);
/// ```
pub struct RagRunnable {
    pipeline: Arc<RAGPipeline>,
}

impl RagRunnable {
    /// Create a new adapter wrapping the given RAG pipeline.
    pub fn new(pipeline: Arc<RAGPipeline>) -> Self {
        Self { pipeline }
    }
}

#[async_trait]
impl Runnable<String, String> for RagRunnable {
    type Error = LcelError;

    async fn invoke(
        &self,
        input: String,
        _config: Option<RunnableConfig>,
    ) -> Result<String, LcelError> {
        self.pipeline
            .query(&input)
            .await
            .map_err(|e| LcelError::Other(format!("RAG query error: {}", e)))
    }

    // stream, batch, transform use default implementations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Retriever;
    use crate::pipeline::RAGPipelineBuilder;
    use lc_providers::{OpenAIChat, OpenAIConfig};

    /// Verifies the adapter can wrap a real RAGPipeline and maps query-time
    /// failures into `LcelError`, completely offline: BM25 needs no network/LLM,
    /// and the `OpenAIChat` is only *constructed* (no call runs until `invoke`,
    /// which fails fast against the unreachable 127.0.0.1 base URL).
    #[tokio::test]
    async fn rag_runnable_wraps_pipeline_and_maps_query_errors() {
        let config = OpenAIConfig::new("test_key").with_base_url("http://127.0.0.1:1/v1");
        let pipeline = RAGPipelineBuilder::new()
            .llm(OpenAIChat::new(config))
            .retriever(BM25Retriever::new())
            .retrieve_k(1)
            .build()
            .expect("a retriever-only pipeline builds with no LLM call");

        let rag_runnable = RagRunnable::new(Arc::new(pipeline));

        // Retrieval returns no sources; the pipeline then calls the LLM, which
        // fails against the unreachable base URL => invoke must surface a mapped
        // `LcelError` (prefixed), not panic inside the adapter.
        let err = rag_runnable
            .invoke("what is rust".to_string(), None)
            .await
            .expect_err("unreachable LLM base URL must fail the query");
        assert!(
            err.to_string().contains("RAG query error"),
            "adapter must wrap pipeline errors: {err}"
        );
    }
}
