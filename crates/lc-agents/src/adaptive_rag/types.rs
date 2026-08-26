// lc-agents/src/adaptive_rag/types.rs
//! Decision / result / error types for [`crate::adaptive_rag::AdaptiveRAG`].

use lc_rag::RetrieverError;
use lc_vector_stores::Document;

/// Decision made by the adaptive router.
#[derive(Debug, Clone, PartialEq)]
pub enum RagDecision {
    /// No retrieval needed - LLM can answer directly.
    NoRetrieval,
    /// Single search query sufficient.
    SingleSearch,
    /// Complex query - use multi-query expansion.
    MultiQuery,
}

impl std::fmt::Display for RagDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RagDecision::NoRetrieval => write!(f, "no_retrieval"),
            RagDecision::SingleSearch => write!(f, "single_search"),
            RagDecision::MultiQuery => write!(f, "multi_query"),
        }
    }
}

/// Result returned by [`AdaptiveRAG::invoke`](crate::adaptive_rag::AdaptiveRAG::invoke).
#[derive(Debug, Clone)]
pub struct AdaptiveRAGResult {
    /// The generated answer.
    pub answer: String,
    /// The routing decision that was made.
    pub decision: RagDecision,
    /// Source documents used (empty when `decision` is `NoRetrieval`).
    pub sources: Vec<Document>,
}

/// Errors produced by [`AdaptiveRAG`](crate::adaptive_rag::AdaptiveRAG).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AdaptiveRAGError {
    /// LLM invocation failed.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Retrieval failed.
    #[error("retrieval error: {0}")]
    Retrieval(#[from] RetrieverError),

    /// Failed to parse the routing decision from the LLM response.
    #[error("decision parse error: {0}")]
    DecisionParse(String),
}
