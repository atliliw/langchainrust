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

pub mod document {
    pub use crate::document_types::{ChunkDocument, Document, SearchResult, VectorDocument};
}

pub mod splitter {
    pub use crate::splitter_types::{
        CharacterTextSplitter, RecursiveCharacterSplitter, TextSplitter,
    };
}

pub mod tools {
    pub use crate::tool_types::{FunctionCall, ToolCall, ToolCallResult};
}

// Flat re-exports for convenience
pub use document_types::{ChunkDocument, Document, SearchResult, VectorDocument};
pub use splitter_types::{CharacterTextSplitter, RecursiveCharacterSplitter, TextSplitter};
pub use tool_types::{FunctionCall, ToolCall, ToolCallResult};

mod document_types;
mod splitter_types;
mod tool_types;
