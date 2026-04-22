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

pub use runnables::{Runnable, RunnableConfig};
pub use language_models::{BaseLanguageModel, BaseChatModel};
pub use tools::{
    BaseTool, Tool, ToolError, ToolRegistry,
    ToolDefinition, ToolCall, ToolCallResult, FunctionDefinition, FunctionCall,
    StructuredOutput,
};
