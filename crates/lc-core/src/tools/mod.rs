// src/core/tools/mod.rs
//! Tool abstractions for agent function calling.
//!
//! Provides base traits and types for tool integration:
//! - `BaseTool`: String-based tool interface (object-safe)
//! - `Tool`: Type-safe generic tool interface
//! - `ToolDefinition`: LLM function calling definition
//! - `ToolRegistry`: Tool collection and lookup

mod base;
mod registry;
mod structured;
mod structured_output;
mod tool_definition;

pub use base::{to_tool_definition, BaseTool, Tool, ToolError};
pub use registry::ToolRegistry;
pub use structured::StructuredTool;
pub use structured_output::StructuredOutput;
pub use tool_definition::{
    FunctionCall, FunctionDefinition, ToolCall, ToolCallBuilder, ToolCallResult, ToolDefinition,
};
