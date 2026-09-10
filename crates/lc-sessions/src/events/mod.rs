//! Event-sourced sessions (0.22.0): append-only event log, deterministic
//! replay, forks, turn-granular compaction, and a checkpoint interface
//! placeholder for durable execution (0.23.0).
//!
//! # Layout
//!
//! - [`model`]: `SessionEvent` / `EventPayload` / `Turn`
//! - [`store`]: `EventStore` trait + `MemoryEventStore` (idempotent appends)
//! - [`replay`]: `project()` / `to_session()` / `compact_to_snapshot()` /
//!   `assert_no_orphan_tool_results()`
//! - [`checkpoint`]: `SessionCheckpoint` trait (placeholder) + `NoopCheckpoint`
//! - [`manager`]: `EventSessionManager` — events-API chat/fork/compaction

pub mod checkpoint;
pub mod manager;
pub mod model;
pub mod replay;
pub mod store;

pub use checkpoint::{NoopCheckpoint, SessionCheckpoint};
pub use manager::{AutoCompaction, EventSessionManager};
pub use model::{EventPayload, SessionEvent, Turn};
pub use replay::{project, to_session, ProjectedSession};
pub use store::{EventStore, MemoryEventStore};
