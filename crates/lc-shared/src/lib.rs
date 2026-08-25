#![warn(missing_docs)]
// lc-shared/src/lib.rs
//! Shared types for the langchainrust workspace.
//!
//! This crate contains types that are needed by multiple sub-crates,
//! breaking circular dependencies:
//!
//! - **Tool types** (`ToolCall`, `FunctionCall`, `ToolCallResult`): needed by
//!   both `lc-schema` (Message uses ToolCall) and `lc-core` (tool definitions).
//!
//! - **Document types** (`Document`, `VectorDocument`, `SearchResult`, `ChunkDocument`):
//!   needed by both `lc-vector-stores` and `lc-rag`.
//!
//! - **Splitter types** (`TextSplitter`, `RecursiveCharacterSplitter`):
//!   needed by both `lc-vector-stores` and `lc-rag`.

/// Document types: `Document`, `VectorDocument`, `SearchResult`, `ChunkDocument`.
pub mod document {
    pub use crate::document_types::{ChunkDocument, Document, SearchResult, VectorDocument};
}

/// Splitter types: `TextSplitter`, `RecursiveCharacterSplitter`.
pub mod splitter {
    pub use crate::splitter_types::{RecursiveCharacterSplitter, TextSplitter};
}

/// Tool types: `ToolCall`, `FunctionCall`, `ToolCallResult`.
pub mod tools {
    pub use crate::tool_types::{FunctionCall, ToolCall, ToolCallBuilder, ToolCallResult};
}

// Tolerant JSON repair — sinks the LLM-JSON repair pipeline from lc-core so
// that `ToolCall::parse_arguments` (in this crate) shares a single tolerant
// parser with lc-core's `json_parse` instead of duplicating strict parsing.
pub mod json_repair;

// Flat re-exports for convenience
pub use document_types::{ChunkDocument, Document, SearchResult, VectorDocument};
pub use json_repair::{parse_tolerant_json, repair_json, JsonRepairError};
pub use splitter_types::{RecursiveCharacterSplitter, TextSplitter};
pub use tool_types::{FunctionCall, ToolCall, ToolCallBuilder, ToolCallResult};

mod document_types;
mod splitter_types;
mod tool_types;
