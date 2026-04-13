// src/callbacks/mod.rs
//! Callback and tracing system
//!
//! This module provides callbacks for observability, tracing, and monitoring.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use langchainrust::callbacks::{CallbackManager, StdOutHandler, LangSmithHandler};
//! use std::sync::Arc;
//!
//! // Create callback manager with handlers
//! let manager = CallbackManager::new()
//!     .add_handler(Arc::new(StdOutHandler::new()))
//!     .add_handler(Arc::new(LangSmithHandler::from_env()?));
//!
//! // Use with LLM or Agent
//! let llm = OpenAIChat::new(config);
//! // llm.with_callbacks(Arc::new(manager));
//! ```
//!
//! # Environment Variables for LangSmith
//!
//! - `LANGSMITH_API_KEY`: API key (required, starts with "ls_")
//! - `LANGSMITH_TRACING`: Enable tracing (default: "true")
//! - `LANGSMITH_PROJECT`: Project name (default: "default")
//! - `LANGSMITH_ENDPOINT`: API endpoint (default: LangSmith official)
//! - `LANGSMITH_WORKSPACE_ID`: Workspace ID (required for org accounts)

mod run_type;
mod run_tree;
mod base;
mod langsmith_client;
pub mod handlers;

pub use run_type::RunType;
pub use run_tree::{RunCreate, RunTree, RunUpdate};
pub use base::{CallbackHandler, CallbackManager};
pub use langsmith_client::{LangSmithClient, LangSmithConfig, LangSmithError};
pub use handlers::{LangSmithHandler, StdOutHandler, FileCallbackHandler, LogFormat};