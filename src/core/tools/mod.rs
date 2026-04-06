// src/core/tools/mod.rs
//! 工具基础模块
//!
//! 参考 Python 版本: langchain/libs/core/langchain_core/tools/base.py

mod base;
mod structured;
mod registry;

pub use base::{BaseTool, Tool, ToolError};
pub use structured::StructuredTool;
pub use registry::ToolRegistry;