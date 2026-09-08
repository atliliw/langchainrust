// lc-agents/src/metrics.rs
//! Agent execution metrics (P1-5)
//!
//! Re-exported from `lc-core` (v0.20.2): the aggregate type moved there so the
//! unified observability payload [`ObsEvent`](lc_core::observability::ObsEvent)
//! can reference it from the core layer. This module keeps the
//! `lc_agents::metrics::AgentMetrics` path for compatibility.
//!
//! Metrics are written to `AgentExecutor::last_metrics()` at the end of every
//! execution, emitted as an audit log via `log_summary()`, and (with a sink
//! attached via `with_metrics_sink`) exported as an `ObsEvent::AgentMetrics`.

pub use lc_core::observability::AgentMetrics;
