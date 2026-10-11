//! Event-sourced sessions (0.22.0): append-only event log, deterministic
//! replay, forks, turn-granular compaction, and durable checkpoints.
//!
//! # Layout
//!
//! - [`model`]: `SessionEvent` / `EventPayload` / `Turn`
//! - [`store`]: `EventStore` trait + `MemoryEventStore` (idempotent appends)
//! - [`replay`]: `project()` / `to_session()` / `compact_to_snapshot()` /
//!   `age_bounded_turns()` / `merge_projections()` /
//!   `assert_no_orphan_tool_results()`
//! - [`checkpoint`]: `SessionCheckpoint` trait + `NoopCheckpoint` / `FileCheckpoint`
//! - [`manager`]: `EventSessionManager` — events-API chat/fork/compaction/gc/recovery

pub mod checkpoint;
pub mod manager;
pub mod model;
pub mod replay;
pub mod store;

pub use checkpoint::{FileCheckpoint, NoopCheckpoint, SessionCheckpoint};
pub use manager::{AutoCompaction, EventSessionManager};
pub use model::{EventPayload, SessionEvent, Turn};
pub use replay::{age_bounded_turns, merge_projections, project, to_session, ProjectedSession};
pub use store::{EventStore, MemoryEventStore};
