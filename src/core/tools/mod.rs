// src/core/tools/mod.rs
//! Tool abstractions for agent function calling.
//!
//! Provides base traits and types for tool integration:
//! - `BaseTool`: String-based tool interface (object-safe)
//! - `Tool`: Type-safe generic tool interface
//! - `ToolDefinition`: LLM function calling definition
//! - `ToolRegistry`: Tool collection and lookup

mod base;
mod structured;
mod registry;
mod tool_definition;
mod structured_output;

pub use base::{BaseTool, Tool, ToolError, to_tool_definition};
pub use structured::StructuredTool;
pub use registry::ToolRegistry;
pub use tool_definition::{ToolDefinition, FunctionDefinition, ToolCall, FunctionCall, ToolCallResult};
pub use structured_output::StructuredOutput;