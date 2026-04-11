// src/core/runnables/mod.rs
//! Runnable module - LangChain Expression Language (LCEL) core
//!
//! The Runnable trait is the foundation of LangChain's composability.
//! Every component (LLM, Prompt, Tool, etc.) implements Runnable,
//! enabling them to be chained together seamlessly.

mod config;
mod runnable_trait;

pub use config::RunnableConfig;
pub use runnable_trait::Runnable;
