#![warn(missing_docs)]
//! Sessions management
//!
//! Two APIs coexist in 0.22.0:
//!
//! - **`events` (recommended)**: an append-only event log with deterministic
//!   replay, forks, turn-granular compaction, and a checkpoint placeholder.
//!   Start with [`EventSessionManager`] + [`MemoryEventStore`].
//! - **Legacy mutable sessions** (below): kept functional, marked deprecated,
//!   removal in 0.23.0. See `docs/internal/v0.22.0/MIGRATION.md`.
//!
//! Manages the session lifecycle of multi-turn conversations: creating/getting/archiving
//! sessions, chatting within a session (history auto-maintained), with pluggable storage.
//!
//! # Example (events API)
//! ```no_run
//! use lc_sessions::{EventSessionManager, MemoryEventStore};
//! use lc_core::BaseChatModel;
//! use std::sync::Arc;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let manager = EventSessionManager::new(Arc::new(MemoryEventStore::new()));
//! let id = manager.create_session().await?;
//! // let llm = ...; // any type implementing BaseChatModel
//! // let reply = manager.chat(&id, &llm, "你好".to_string()).await?;
//! # Ok(())
//! # }
//! ```

pub mod events;
pub mod manager;
pub mod memory_store;
pub mod session;
pub mod session_runnable;
pub mod store;

pub use events::{
    AutoCompaction, EventPayload, EventSessionManager, EventStore, MemoryEventStore,
    NoopCheckpoint, ProjectedSession, SessionCheckpoint, SessionEvent, Turn,
};
#[allow(deprecated)]
pub use manager::SessionManager;
pub use memory_store::MemorySessionStore;
pub use session::{Session, SessionStatus};
pub use session_runnable::SessionManagerRunnable;
pub use store::{SessionError, SessionStore};
