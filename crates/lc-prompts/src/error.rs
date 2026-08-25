// crates/lc-prompts/src/error.rs
//! Error types for prompt templates.

/// Error type for prompt template operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PromptsError {
    /// A template referenced a variable that was not provided.
    #[error("Missing variable: {0}")]
    MissingVariable(String),
}
