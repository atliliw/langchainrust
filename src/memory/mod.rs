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
//! - **ConversationSummaryMemory**: LLM-based summary compression.
//! - **ConversationSummaryBufferMemory**: Hybrid compression (summary + recent messages).
//! - **PersistentMemory**: Trait for persistent memory storage.
//! - **MongoPersistentMemory**: MongoDB-backed persistent memory.
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
//!
//! # Persistent Memory Example
//!
//! ```ignore
//! use langchainrust::{MongoPersistentMemory, PersistentMemory, OpenAIChat};
//!
//! let llm = OpenAIChat::new(config);
//! let mut memory = MongoPersistentMemory::new(
//!     "mongodb://localhost:27017",
//!     "my_db",
//!     "memory_sessions",
//!     llm
//! ).await?;
//!
//! memory.set_session_id("session_123");
//! memory.save_context(&inputs, &outputs).await?;  // Auto-saves to MongoDB
//! ```

pub mod base;
pub mod buffer;
pub mod window;
pub mod summary;
pub mod summary_buffer;
pub mod persistent;

#[cfg(feature = "mongodb-persistence")]
pub mod mongo_memory;

pub use base::{BaseMemory, MemoryError, ChatMessageHistory};
pub use buffer::ConversationBufferMemory;
pub use window::ConversationBufferWindowMemory;
pub use summary::ConversationSummaryMemory;
pub use summary_buffer::ConversationSummaryBufferMemory;
pub use persistent::{PersistentMemory, PersistenceConfig, MemoryData};

#[cfg(feature = "mongodb-persistence")]
pub use mongo_memory::MongoPersistentMemory;