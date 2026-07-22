// src/core/mod.rs
//! Core abstractions for LangChainRust.
//!
//! This module provides the foundational traits and types:
//! - `Runnable`: Base execution interface
//! - `BaseLanguageModel`: LLM abstraction
//! - `BaseChatModel`: Chat model interface
//! - `BaseTool`, `Tool`: Tool abstraction

pub mod runnables;
pub mod language_models;
pub mod tools;
pub mod output_parsers;
pub mod cache;
pub mod token_counter;
pub mod structured_output;
pub mod math;

pub use runnables::{Runnable, RunnableConfig};
pub use language_models::{BaseLanguageModel, BaseChatModel};
pub use tools::{
    BaseTool, Tool, ToolError, ToolRegistry,
    ToolDefinition, ToolCall, ToolCallResult, FunctionDefinition, FunctionCall,
    StructuredOutput,
};
pub use output_parsers::{
    BaseOutputParser, OutputParserError, OutputParserResult,
    StrOutputParser, CommaSeparatedListOutputParser,
    JsonOutputParser, StructuredOutputParser, TypedOutputParser,
};
pub use structured_output::{StructuredOutputExt, StructuredOutputError, with_structured_output};
