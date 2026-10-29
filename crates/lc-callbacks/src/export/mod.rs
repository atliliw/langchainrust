//! Batched export plus a local on-disk Edge buffer (lc-callbacks T3: "批导出 +
//! 本地 Edge 缓冲").
//!
//! The pipeline is sink-agnostic: [`BatchSink`] implementations carry the
//! LangSmith / Langfuse / OTLP payload shape, while one retry/backoff/disk-park
//! implementation serves them all. A trace completes → its root span ends on the
//! [`InMemoryTracingBackend`](crate::InMemoryTracingBackend) → the traced tree is
//! handed to a [`TraceFlushCallback`](crate::tracing::backend::TraceFlushCallback)
//! → [`BatchExporter::trace_flush_callback`] turns it into an [`ExportEvent`] and
//! accumulates it. A full batch is delivered; an unreachable sink parks the batch
//! on the [`EdgeBuffer`] for later [`replay`](BatchExporter::replay).

pub mod batch;
pub mod edge_buffer;
// Langfuse Ingestion `BatchSink` (HMAC-SHA256-signed): closes the other half of
// "LangSmith / Langfuse 导出". Pure signing primitives + a `#[tokio::test]` that
// round-trips a signed request through a TCP mock.
pub mod langfuse_sink;

pub use batch::{
    trace_event, BatchExporter, BatchSink, ExportError, ExportEvent, FlushOutcome, RetryPolicy,
    DEFAULT_BATCH_SIZE, TRACE_EVENT_KIND,
};
pub use edge_buffer::{EdgeBuffer, DEFAULT_BUFFER_ENTRIES};
pub use langfuse_sink::{
    hmac_sha256_hex, sha256_hex, sign_payload, LangfuseConfig, LangfuseError, LangfuseSignature,
    LangfuseSink,
};
