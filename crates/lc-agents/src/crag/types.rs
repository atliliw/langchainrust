// lc-agents/src/crag/types.rs
//! Result types for [`crate::crag::CorrectiveRAGAgent`].

use lc_vector_stores::Document;

/// Result of a CRAG invocation.
#[derive(Debug, Clone)]
pub struct CRAGResult {
    /// The generated answer.
    pub answer: String,
    /// Whether the answer is grounded in the source documents.
    pub grounded: bool,
    /// Source documents used to generate the answer.
    pub sources: Vec<Document>,
    /// Relevance grade scores for each source document.
    pub grade_scores: Vec<f64>,
    /// Relevance reasoning from the grader for each source document.
    pub grade_reasoning: Vec<Option<String>>,
}
