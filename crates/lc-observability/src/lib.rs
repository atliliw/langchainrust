// lc-observability/src/lib.rs
//! Pluggable observability sinks for LangChainRust.
//!
//! The framework exposes the [`MetricsSink`] interface and an [`ObsEvent`] payload
//! from `lc-core` (written by `TokenTrackingLLM` and `AgentExecutor`). This crate
//! ships concrete sinks that implement it:
//!
//! - [`JsonLinesSink`] (default `jsonl` feature): append each event as one JSON
//!   line to a file.
//! - MongoSink (`mongodb` feature): insert each event as a document.
//!
//! Export failures are returned as [`ObsError`]; the framework logs a `warn` and
//! the main flow continues — sinks never interrupt the agent.

mod jsonl;
#[cfg(feature = "mongodb")]
mod mongo;

pub use jsonl::JsonLinesSink;
#[cfg(feature = "mongodb")]
pub use mongo::MongoSink;

pub use lc_core::observability::{MetricsSink, ObsError, ObsEvent};
