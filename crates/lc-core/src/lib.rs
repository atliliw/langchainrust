// lc-core/src/lib.rs
//! Core abstractions for LangChainRust.
//!
//! This crate provides the foundational traits and types:
//! - `Runnable`: Base execution interface
//! - `BaseLanguageModel`: LLM abstraction
//! - `BaseChatModel`: Chat model interface
//! - `BaseTool`, `Tool`: Tool abstraction
//! - `RunnableConfig`: Execution configuration with callbacks
//! - Output parsers, structured output, caching, token counting, batch API

pub mod batch;
pub mod cache;
pub mod json_parse;
pub mod language_models;
pub mod math;
pub mod output_parsers;
pub mod router_llm;
pub mod runnables;
pub mod structured_output;
pub mod token_counter;
pub mod tools;

// Re-export key types at crate root for convenience
pub use json_parse::{parse_llm_json, parse_llm_json_with_retry, LlmJsonParseError};

pub use language_models::{BaseChatModel, BaseLanguageModel};
pub use output_parsers::{
    BaseOutputParser, CommaSeparatedListOutputParser, JsonOutputParser, OutputParserError,
    OutputParserResult, StrOutputParser, StructuredOutputParser, TypedOutputParser,
};
pub use runnables::{Runnable, RunnableConfig};
pub use structured_output::{
    stream_structured_output, with_structured_output, PartialJsonError, PartialJsonParser,
    StreamingStructuredOutputExt, StructuredOutputError, StructuredOutputExt,
};
pub use tools::{
    BaseTool, FunctionCall, FunctionDefinition, StructuredOutput, Tool, ToolCall, ToolCallResult,
    ToolDefinition, ToolError, ToolRegistry,
};

/// Unified error type for the lc-core crate.
///
/// Aggregates all core-specific error types so the `?` operator works
/// seamlessly across core module boundaries.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Tool execution error.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// JSON parse error from LLM output.
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] LlmJsonParseError),

    /// Batch processing error.
    #[error("Batch error: {0}")]
    Batch(#[from] batch::BatchError),

    /// Router error.
    #[error("Router error: {0}")]
    Router(#[from] router_llm::RouterError),

    /// Structured output extraction error.
    #[error("Structured output error: {0}")]
    StructuredOutput(#[from] StructuredOutputError),

    /// Partial JSON parsing error.
    #[error("Partial JSON error: {0}")]
    PartialJson(#[from] PartialJsonError),

    /// Output parser error.
    #[error("Output parser error: {0}")]
    OutputParser(#[from] OutputParserError),

    /// Math operation error.
    #[error("Math error: {0}")]
    Math(#[from] math::MathError),

    /// Any other error (e.g., from providers that haven't been extracted yet).
    #[error("{0}")]
    Other(String),
}

// Allow external error types to convert into CoreError via string wrapping
impl From<std::convert::Infallible> for CoreError {
    fn from(_: std::convert::Infallible) -> Self {
        unreachable!()
    }
}

/// Helper to convert any error into `CoreError::Other`.
/// Use this instead of `?` when the error type is not a known CoreError variant.
pub fn other_error<E: std::fmt::Display>(err: E) -> CoreError {
    CoreError::Other(err.to_string())
}
