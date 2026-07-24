// src/core/mod.rs
//! Core abstractions for LangChainRust.
//!
//! This module provides the foundational traits and types:
//! - `Runnable`: Base execution interface
//! - `BaseLanguageModel`: LLM abstraction
//! - `BaseChatModel`: Chat model interface
//! - `BaseTool`, `Tool`: Tool abstraction

pub mod batch;
pub mod cache;
pub mod language_models;
pub mod math;
pub mod output_parsers;
pub mod router_llm;
pub mod runnables;
pub mod structured_output;
pub mod token_counter;
pub mod tools;

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
