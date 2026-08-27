// lc-core/src/runnables/error.rs
//! Unified error type for LCEL (LangChain Expression Language) pipelines.
//!
//! `LcelError` bridges all sub-crate error types into a single enum,
//! enabling `RunnableSequence` and other LCEL combinators to work with
//! a uniform error type regardless of which components are in the pipeline.
//!
//! # Design Decision
//!
//! `LcelError` stores error descriptions as `String` rather than wrapping
//! concrete sub-error types (e.g. `Chain(ChainError)`). This avoids:
//! 1. Circular dependencies between `lc-core` and downstream crates
//! 2. Type-erased pipeline steps where the concrete error type is lost anyway
//! 3. Bloating the enum with every provider-specific error variant
//!
//! The `Display` representation preserves enough information for debugging.

use crate::output_parsers::OutputParserError;
use std::fmt;

/// Unified error type for LCEL pipelines.
///
/// All `Runnable` components that participate in LCEL composition
/// must have an `Error` type that implements `Into<LcelError>`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LcelError {
    /// Error from an LLM provider (OpenAI, Anthropic, Gemini, Ollama, etc.).
    Provider(String),

    /// Error from a chain execution.
    Chain(String),

    /// Error from an agent execution.
    Agent(String),

    /// Error from a graph execution.
    Graph(String),

    /// Error from a tool execution.
    Tool(String),

    /// Error from an output parser.
    OutputParser(String),

    /// Error during streaming.
    Stream(String),

    /// Error in pipeline composition or execution.
    Pipeline(String),

    /// Type-erasure downcast failure.
    TypeMismatch(String),

    /// Catch-all for errors that don't fit other variants.
    Other(String),
}

impl fmt::Display for LcelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LcelError::Provider(msg) => write!(f, "Provider error: {msg}"),
            LcelError::Chain(msg) => write!(f, "Chain error: {msg}"),
            LcelError::Agent(msg) => write!(f, "Agent error: {msg}"),
            LcelError::Graph(msg) => write!(f, "Graph error: {msg}"),
            LcelError::Tool(msg) => write!(f, "Tool error: {msg}"),
            LcelError::OutputParser(msg) => write!(f, "Output parser error: {msg}"),
            LcelError::Stream(msg) => write!(f, "Stream error: {msg}"),
            LcelError::Pipeline(msg) => write!(f, "Pipeline error: {msg}"),
            LcelError::TypeMismatch(msg) => write!(f, "Type mismatch: {msg}"),
            LcelError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LcelError {}

// Allow `Infallible` to convert into `LcelError` (never actually happens).
impl From<std::convert::Infallible> for LcelError {
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!()
    }
}

// Allow output parser errors into `LcelError` so parsers can be the
// second (or later) step of a `pipe()` chain — `R2::Error: Into<LcelError>`
// is required by `RunnableExt::pipe`.
impl From<OutputParserError> for LcelError {
    fn from(e: OutputParserError) -> Self {
        LcelError::OutputParser(e.to_string())
    }
}

// Allow tool errors into `LcelError` so tools can be a step of a `pipe()` chain
// (e.g. `tool.pipe(...)`), mapping into the existing `Tool` variant.
impl From<crate::tools::ToolError> for LcelError {
    fn from(e: crate::tools::ToolError) -> Self {
        LcelError::Tool(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats_correctly() {
        assert_eq!(
            LcelError::Provider("openai timeout".to_string()).to_string(),
            "Provider error: openai timeout"
        );
        assert_eq!(
            LcelError::Chain("missing input".to_string()).to_string(),
            "Chain error: missing input"
        );
        assert_eq!(
            LcelError::TypeMismatch("expected String got i32".to_string()).to_string(),
            "Type mismatch: expected String got i32"
        );
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<LcelError>();
    }

    #[test]
    fn from_output_parser_error() {
        // parser errors must convert cleanly into LcelError (no panic) so the parser compiles as the second pipe step
        let e = OutputParserError::JsonError("bad json".to_string());
        let lcel: LcelError = e.into();
        assert!(matches!(
            lcel,
            LcelError::OutputParser(ref msg) if msg.contains("bad json")
        ));
        assert_eq!(
            lcel.to_string(),
            "Output parser error: JSON error: bad json"
        );
    }

    #[test]
    fn from_tool_error() {
        // tool errors must convert cleanly into LcelError (no panic) so `tool.pipe(...)` compiles
        use crate::tools::ToolError;
        let e = ToolError::InvalidInput("bad input".to_string());
        let lcel: LcelError = e.into();
        assert!(matches!(
            lcel,
            LcelError::Tool(ref msg) if msg.contains("bad input")
        ));
        assert_eq!(lcel.to_string(), "Tool error: Invalid input: bad input");
    }
}
