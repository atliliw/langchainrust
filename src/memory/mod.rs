// src/memory/mod.rs
//! Memory system for conversation history management.
//!
//! Provides conversation memory management functionality.
//!
//! # Core Concepts
//!
//! - **BaseMemory**: Base trait for memory.
//! - **ConversationBufferMemory**: Simple conversation buffer.
//! - **ConversationBufferWindowMemory**: Conversation buffer with window.
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::{ConversationBufferMemory, BaseMemory};
//! use std::collections::HashMap;
//!
//! let mut memory = ConversationBufferMemory::new();
//!
//! // Save conversation
//! let inputs = HashMap::from([("input".to_string(), "Hello".to_string())]);
//! let outputs = HashMap::from([("output".to_string(), "Hi!".to_string())]);
//! memory.save_context(&inputs, &outputs).await?;
//!
//! // Load memory
//! let vars = memory.load_memory_variables(&HashMap::new()).await?;
//! println!("{:?}", vars.get("history"));
//! ```

pub mod base;
pub mod buffer;
pub mod window;
pub mod summary;
pub mod summary_buffer;

pub use base::{BaseMemory, MemoryError, ChatMessageHistory};
pub use buffer::ConversationBufferMemory;
pub use window::ConversationBufferWindowMemory;
pub use summary::ConversationSummaryMemory;
pub use summary_buffer::ConversationSummaryBufferMemory;