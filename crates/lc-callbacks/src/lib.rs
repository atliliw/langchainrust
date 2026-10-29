#![warn(missing_docs)]
// lc-callbacks/src/lib.rs
//! Callback and tracing system
//!
//! This module provides callbacks for observability, tracing, and monitoring.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use lc_callbacks::{CallbackManager, StdOutHandler, LangSmithHandler};
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

mod base;
pub mod circuit_breaker;
// Batched export + local Edge buffer (lc-callbacks T3): sink-agnostic batch
// delivery (LangSmith / Langfuse / …) with bounded retry and an on-disk park
// for undeliverable batches. Fed by the per-trace flush callback on
// [`InMemoryTracingBackend`].
pub mod export;
pub mod handlers;
mod langsmith_client;
#[cfg(feature = "otlp")]
pub mod otlp;
pub mod pricing;
mod run_tree;
mod run_type;
// Pure attribute-name vocabulary (no OTel SDK types): compiled even without
// the `opentelemetry` feature so instrumentation (e.g. lc-agents stamping run
// metadata) can reference the names without pulling in the SDK.
pub mod semconv;
pub mod tracing;

pub use base::{CallbackHandler, CallbackManager};
pub use circuit_breaker::{BreakerStatus, CircuitBreaker};
pub use export::{
    hmac_sha256_hex, sha256_hex, sign_payload, trace_event, BatchExporter, BatchSink, EdgeBuffer,
    ExportError, ExportEvent, FlushOutcome, LangfuseConfig, LangfuseError, LangfuseSignature,
    LangfuseSink, RetryPolicy, DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_ENTRIES, TRACE_EVENT_KIND,
};
#[cfg(feature = "opentelemetry")]
pub use handlers::OtelHandler;
pub use handlers::{
    FileCallbackHandler, GenericHandler, LangSmithHandler, LogFormat, StdOutHandler,
};
pub use langsmith_client::{LangSmithClient, LangSmithConfig, LangSmithError};
pub use pricing::{estimate_cost_usd, price_for, ModelPrice};
pub use run_tree::{RunCreate, RunTree, RunUpdate};
pub use run_type::RunType;

// Tracing exports
#[cfg(feature = "opentelemetry")]
pub use tracing::OtelTracingBackend;
pub use tracing::{
    aggregate_cost, clear_span_stack, init_task_span_stack, ConsoleTracingBackend,
    InMemoryTracingBackend, SpanGuard, SpanId, SpanKind, SpanStatus, SpanTokenUsage, TraceNode,
    TraceSpan, Tracer, TracingBackend,
};
