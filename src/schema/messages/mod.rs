// src/schema/messages/mod.rs
//! Message types for LangChain
//!
//! Based on Python: langchain/libs/core/langchain_core/messages/
//!
//! Messages are the inputs and outputs of chat models.

mod message;

pub use message::{Message, MessageType};
