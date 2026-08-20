// lc-memory/src/lib.rs
//! Memory system for conversation history management.
//!
//! Provides conversation memory management functionality.
//!
//! # Core Concepts
//!
//! - **BaseMemory**: Base trait for memory.
//! - **ConversationBufferMemory**: Simple conversation buffer.
//! - **ConversationBufferWindowMemory**: Conversation buffer with window.
//! - **ConversationSummaryMemory**: LLM-based summary compression.
//! - **ConversationSummaryBufferMemory**: Hybrid compression (summary + recent messages).
//! - **PersistentMemory**: Trait for persistent memory storage.
//! - **ContextWindow**: Fits messages within a token budget.
//!
//! # Feature Flags
//!
//! - `mongodb-persistence` — Enables `MongoPersistentMemory` (MongoDB-backed persistent memory).
//! - `vectorstore-memory` — Enables `VectorStoreRetrieverMemory` (vector-store-backed memory).

pub mod base;
pub mod buffer;
pub mod context_window;
pub mod persistent;
pub mod summary;
pub mod summary_buffer;
pub mod window;
pub mod with_history;

#[cfg(feature = "vectorstore-memory")]
pub mod vectorstore_memory;

#[cfg(feature = "mongodb-persistence")]
pub mod mongo_memory;

#[cfg(test)]
mod test_support;

pub use base::{
    memory_variables_to_messages, BaseChatMemory, BaseMemory, ChatMessageHistory, MemoryError,
};
pub use buffer::ConversationBufferMemory;
pub use context_window::{ContextWindow, Strategy};
pub use persistent::{MemoryData, PersistenceConfig, PersistentMemory};
pub use summary::ConversationSummaryMemory;
pub use summary_buffer::ConversationSummaryBufferMemory;
pub use window::ConversationBufferWindowMemory;
pub use with_history::RunnableWithMessageHistory;

#[cfg(feature = "vectorstore-memory")]
pub use vectorstore_memory::VectorStoreRetrieverMemory;

#[cfg(feature = "mongodb-persistence")]
pub use mongo_memory::MongoPersistentMemory;
