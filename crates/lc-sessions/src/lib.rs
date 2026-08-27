#![warn(missing_docs)]
//! Sessions management
//!
//! Manages the session lifecycle of multi-turn conversations: creating/getting/archiving
//! sessions, chatting within a session (history auto-maintained), with pluggable storage.
//!
//! # Example
//! ```no_run
//! use lc_sessions::{SessionManager, MemorySessionStore};
//! use lc_core::BaseChatModel;
//! use std::sync::Arc;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let manager = SessionManager::new(Arc::new(MemorySessionStore::new()));
//! let id = manager.create_session().await?;
//! // let llm = ...; // any type implementing BaseChatModel
//! // let reply = manager.chat(&id, &llm, "你好".to_string()).await?;
//! # Ok(())
//! # }
//! ```

pub mod manager;
pub mod memory_store;
pub mod session;
pub mod session_runnable;
pub mod store;

pub use manager::SessionManager;
pub use memory_store::MemorySessionStore;
pub use session::{Session, SessionStatus};
pub use session_runnable::SessionManagerRunnable;
pub use store::{SessionError, SessionStore};
