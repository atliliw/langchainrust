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

    #[test]
    fn rag_runnable_creation() {
        // Just verify the type exists and compiles
        // Actual integration tests would require a live LLM
    }
}
