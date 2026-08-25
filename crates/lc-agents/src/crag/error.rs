// lc-agents/src/crag/error.rs
//! CRAG error types.

use super::grader::GraderError;
use super::rewriter::RewriterError;

/// CRAG error types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CRAGError {
    /// No documents were retrieved from the retriever.
    #[error("No documents retrieved for the query")]
    NoDocumentsRetrieved,

    /// Document retrieval failed.
    #[error("Retrieval error: {0}")]
    RetrievalError(lc_rag::RetrieverError),

    /// Document grading failed.
    #[error("Grading error: {0}")]
    GradingError(GraderError),

    /// Query rewriting failed.
    #[error("Query rewriting error: {0}")]
    RewritingError(RewriterError),

    /// Web search fallback failed.
    #[error("Web search error: {0}")]
    WebSearchError(lc_core::tools::ToolError),

    /// Answer generation failed.
    #[error("Answer generation error: {0}")]
    GenerationError(String),

    /// Hallucination check failed.
    #[error("Hallucination check error: {0}")]
    HallucinationCheckError(String),
}
