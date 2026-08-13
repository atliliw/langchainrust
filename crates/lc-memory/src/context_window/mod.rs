// lc-memory/src/context_window/mod.rs
//! Context Window for long context management.
//!
//! Manages conversation context by fitting messages within a token limit,
//! using either truncation or LLM-based summarization strategies.
//!
//! # Core Concepts
//!
//! - **ContextWindow**: Fits messages within a max token budget.
//! - **Strategy::Truncate**: Drops oldest messages, preserving system messages.
//! - **Strategy::Summarize**: Uses an LLM to compress old messages into a summary.
//!
//! # Example
//!
//! ```ignore
//! use lc_memory::{ContextWindow, Strategy};
//! use lc_core::token_counter::TiktokenCounter;
//!
//! // Truncation strategy
//! let cw = ContextWindow::new(4096)?;
//! let fitted = cw.fit(messages).await?;
//!
//! // Summarization strategy
//! let cw = ContextWindow::with_strategy(4096, Strategy::summarize(llm))?;
//! let fitted = cw.fit(messages).await?;
//! ```

pub mod manager;
pub mod trimmer;

pub use manager::ContextWindow;
pub use trimmer::Strategy;

#[cfg(test)]
mod tests;
