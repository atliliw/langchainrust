// lc-agents/src/deep_research/error.rs
//! Error types for deep research operations.

/// Error types for deep research operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ResearchError {
    /// LLM invocation error.
    #[error("LLM error: {0}")]
    Llm(String),

    /// Search tool execution error.
    #[error("search error: {0}")]
    Search(String),

    /// No search results were found for any query.
    #[error("no results found")]
    NoResults,
}
