//! Batched export pipeline: accumulate events, deliver in batches, park what
//! cannot be delivered on a local disk queue.
//!
//! The pipeline is deliberately sink-agnostic ([`BatchSink`]) so the Langfuse
//! and LangSmith exporters share one retry/backoff/buffer implementation
//! instead of each growing its own.
//!
//! # Delivery contract
//!
//! * [`BatchExporter::record`] only appends to the in-memory pending batch — it
//!   never touches the network, so it is safe to call from a synchronous
//!   context (see [`BatchExporter::trace_flush_callback`]).
//! * [`BatchExporter::flush`] delivers the pending batch through the sink with
//!   bounded retry/backoff. A delivery that still fails is **not** an error the
//!   caller has to handle: the batch is parked on the [`EdgeBuffer`] and the
//!   outcome reports [`FlushOutcome::Buffered`]. Observability must never fail
//!   the run it observes.
//! * [`BatchExporter::replay`] re-sends buffered batches oldest-first and stops
//!   at the first failure, leaving the remaining batches queued so FIFO order
//!   survives a partial recovery.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::tracing::backend::TraceFlushCallback;
use crate::tracing::span::TraceNode;

use super::edge_buffer::EdgeBuffer;

/// Default number of events accumulated before a batch is delivered.
pub const DEFAULT_BATCH_SIZE: usize = 64;

/// Kind tag used by [`trace_event`] for a completed trace tree.
pub const TRACE_EVENT_KIND: &str = "trace";

/// Failure from a [`BatchSink`] or from the local edge buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportError {
    /// Transport failure, timeout, 429 or 5xx — worth another attempt.
    Retryable(String),
    /// Contract error (4xx): retrying repeats the same failure verbatim, so the
    /// batch is dropped rather than parked.
    Fatal(String),
    /// Local edge-buffer failure (disk full, unreadable directory, …).
    Io(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Retryable(msg) => write!(f, "retryable export error: {msg}"),
            Self::Fatal(msg) => write!(f, "fatal export error: {msg}"),
            Self::Io(msg) => write!(f, "export buffer error: {msg}"),
        }
    }
}

impl std::error::Error for ExportError {}

/// One exportable event handed to a [`BatchSink`].
///
/// The envelope (kind / id / timestamp) is shared by every sink; `body` is the
/// sink-specific payload. [`EdgeBuffer`] serializes events verbatim, so a
/// buffered batch replays into exactly the request that failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportEvent {
    /// Sink-specific event type, e.g. `"trace-create"` (Langfuse ingestion) or
    /// `"run-create"` (LangSmith batch endpoint).
    pub kind: String,
    /// Stable event id; sinks use it as their dedup/upsert key.
    pub id: String,
    /// RFC 3339 timestamp the event was produced at.
    pub timestamp: String,
    /// Sink-specific JSON body.
    pub body: serde_json::Value,
}

impl ExportEvent {
    /// Build an event stamped with the current time.
    pub fn now(kind: impl Into<String>, id: impl Into<String>, body: serde_json::Value) -> Self {
        Self::at(kind, id, chrono::Utc::now().to_rfc3339(), body)
    }

    /// Build an event with an explicit timestamp.
    pub fn at(
        kind: impl Into<String>,
        id: impl Into<String>,
        timestamp: impl Into<String>,
        body: serde_json::Value,
    ) -> Self {
        Self {
            kind: kind.into(),
            id: id.into(),
            timestamp: timestamp.into(),
            body,
        }
    }
}

/// Destination that accepts a batch of events in one call.
#[async_trait]
pub trait BatchSink: Send + Sync {
    /// Deliver one batch.
    ///
    /// Returning [`ExportError::Retryable`] lets [`BatchExporter`] retry with
    /// backoff; [`ExportError::Fatal`] stops the attempt loop immediately.
    async fn send_batch(&self, batch: &[ExportEvent]) -> Result<(), ExportError>;

    /// Short sink name used in log lines.
    fn name(&self) -> &str;
}

/// Bounded retry/backoff applied to one batch before it is parked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts, first try included. Clamped to at least 1.
    pub max_attempts: usize,
    /// Base backoff in milliseconds, doubled each attempt (`0` disables the
    /// sleep, which is what the offline tests use).
    pub backoff_base_ms: u64,
}

impl RetryPolicy {
    /// Build a policy.
    pub const fn new(max_attempts: usize, backoff_base_ms: u64) -> Self {
        Self {
            max_attempts,
            backoff_base_ms,
        }
    }

    /// Exponential backoff for `attempt` (0-based), saturating instead of
    /// overflowing on a large attempt count.
    fn delay_for(&self, attempt: usize) -> Duration {
        Duration::from_millis(self.backoff_base_ms.saturating_mul(1u64 << attempt.min(16)))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3, 150)
    }
}

/// What a [`BatchExporter::flush`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlushOutcome {
    /// Nothing was pending, so no request was made.
    Empty,
    /// The pending batch was still below `batch_size`; nothing was sent.
    Pending {
        /// Events waiting in the in-memory batch.
        queued: usize,
    },
    /// The batch was delivered.
    Sent(usize),
    /// The sink stayed unreachable; the events were parked on the edge buffer.
    Buffered {
        /// Number of parked events.
        events: usize,
        /// Last delivery error.
        error: String,
    },
    /// The sink rejected the batch permanently (4xx); the events were dropped.
    Dropped {
        /// Number of dropped events.
        events: usize,
        /// Delivery error.
        error: String,
    },
}

/// Accumulates events and delivers them in batches through a [`BatchSink`].
///
/// See the module docs for the delivery contract.
pub struct BatchExporter {
    sink: Arc<dyn BatchSink>,
    buffer: Arc<EdgeBuffer>,
    policy: RetryPolicy,
    batch_size: usize,
    pending: Mutex<Vec<ExportEvent>>,
}

impl BatchExporter {
    /// Build an exporter with the given batch size and retry policy.
    pub fn new(
        sink: Arc<dyn BatchSink>,
        buffer: Arc<EdgeBuffer>,
        batch_size: usize,
        policy: RetryPolicy,
    ) -> Self {
        Self {
            sink,
            buffer,
            policy,
            batch_size: batch_size.max(1),
            pending: Mutex::new(Vec::new()),
        }
    }

    /// Build an exporter with [`DEFAULT_BATCH_SIZE`] and [`RetryPolicy::default`].
    pub fn with_defaults(sink: Arc<dyn BatchSink>, buffer: Arc<EdgeBuffer>) -> Self {
        Self::new(sink, buffer, DEFAULT_BATCH_SIZE, RetryPolicy::default())
    }

    /// Name of the underlying sink.
    pub fn sink_name(&self) -> &str {
        self.sink.name()
    }

    /// Number of events needed to trigger a delivery.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Events currently waiting in the in-memory batch.
    pub fn pending_len(&self) -> usize {
        self.pending.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Batches currently parked on the on-disk edge buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Append one event without doing any I/O.
    ///
    /// Returns `true` when the batch has reached `batch_size`, i.e. when a
    /// delivery is now due.
    pub fn record(&self, event: ExportEvent) -> bool {
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        pending.push(event);
        pending.len() >= self.batch_size
    }

    /// Append an event and deliver the batch when it is full.
    ///
    /// On a successful delivery the on-disk backlog is drained too, so a remote
    /// that just came back is caught up without an explicit [`replay`] call.
    ///
    /// [`replay`]: BatchExporter::replay
    pub async fn enqueue(&self, event: ExportEvent) -> Result<FlushOutcome, ExportError> {
        if !self.record(event) {
            return Ok(FlushOutcome::Pending {
                queued: self.pending_len(),
            });
        }
        let outcome = self.flush().await?;
        if matches!(outcome, FlushOutcome::Sent(_)) {
            self.replay().await?;
        }
        Ok(outcome)
    }

    /// Deliver the pending batch once it reaches `batch_size`.
    ///
    /// A below-threshold batch is **kept pending** (reported [`FlushOutcome::Pending`])
    /// rather than pushed out partial, so a batching sink is not hammered with a
    /// request per event. An [`enqueue`](BatchExporter::enqueue) only calls this on
    /// a full batch; a direct `flush` call defers a partial batch until it fills.
    ///
    /// A still-failing full-batch delivery is parked on the edge buffer and
    /// reported as [`FlushOutcome::Buffered`]; only a local buffer failure is an
    /// `Err`.
    pub async fn flush(&self) -> Result<FlushOutcome, ExportError> {
        let queued = self.pending_len();
        if queued == 0 {
            return Ok(FlushOutcome::Empty);
        }
        if queued < self.batch_size {
            return Ok(FlushOutcome::Pending { queued });
        }
        let batch = self.take_pending();
        let events = batch.len();
        match self.deliver(&batch).await {
            Ok(()) => Ok(FlushOutcome::Sent(events)),
            Err(err @ ExportError::Fatal(_)) => {
                // A contract error (bad key, bad payload) will never succeed on
                // replay either, so parking it would wedge the queue head.
                log::error!(
                    "export to {} rejected {events} event(s) permanently: {err}; dropping the batch",
                    self.sink.name()
                );
                Ok(FlushOutcome::Dropped {
                    events,
                    error: err.to_string(),
                })
            }
            Err(err) => {
                self.buffer.push(&batch)?;
                log::warn!(
                    "export to {} failed ({err}); parked {events} event(s) in {}",
                    self.sink.name(),
                    self.buffer.dir().display()
                );
                Ok(FlushOutcome::Buffered {
                    events,
                    error: err.to_string(),
                })
            }
        }
    }

    /// Re-send parked batches oldest-first.
    ///
    /// Stops at the first batch that still fails (the rest stay queued, keeping
    /// FIFO order intact). Returns how many batches were delivered.
    pub async fn replay(&self) -> Result<usize, ExportError> {
        let mut replayed = 0usize;
        for (path, batch) in self.buffer.batches()? {
            match self.deliver(&batch).await {
                Ok(()) => {
                    self.buffer.remove(&path)?;
                    replayed += 1;
                }
                Err(ExportError::Fatal(msg)) => {
                    log::error!(
                        "dropping permanently-rejected buffered batch {}: {msg}",
                        path.display()
                    );
                    self.buffer.remove(&path)?;
                }
                Err(err) => {
                    log::warn!(
                        "export replay stopped at {} after {replayed} batch(es): {err}",
                        path.display()
                    );
                    break;
                }
            }
        }
        Ok(replayed)
    }

    /// Deliver one batch with bounded retry/backoff.
    async fn deliver(&self, batch: &[ExportEvent]) -> Result<(), ExportError> {
        let attempts = self.policy.max_attempts.max(1);
        let mut last = ExportError::Retryable("no attempt was made".to_string());
        for attempt in 0..attempts {
            match self.sink.send_batch(batch).await {
                Ok(()) => return Ok(()),
                Err(err @ ExportError::Fatal(_)) => return Err(err),
                Err(err) => {
                    last = err;
                    if attempt + 1 < attempts {
                        let delay = self.policy.delay_for(attempt);
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                    }
                }
            }
        }
        Err(last)
    }

    /// Take the pending events, leaving the in-memory batch empty.
    fn take_pending(&self) -> Vec<ExportEvent> {
        std::mem::take(&mut *self.pending.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Build a synchronous trace-flush callback that feeds this exporter.
    ///
    /// The callback is handed to
    /// [`InMemoryTracingBackend::add_trace_flush_callback`] and is invoked once
    /// per completed trace. It only *records* the serialized trace (no I/O, so
    /// span teardown never blocks); when that record completes a full batch the
    /// delivery is handed to the current tokio runtime. With no runtime active
    /// the batch simply stays pending until something calls
    /// [`flush`](BatchExporter::flush).
    ///
    /// [`InMemoryTracingBackend::add_trace_flush_callback`]:
    /// crate::tracing::InMemoryTracingBackend::add_trace_flush_callback
    pub fn trace_flush_callback(self: &Arc<Self>) -> TraceFlushCallback {
        let exporter = Arc::clone(self);
        Arc::new(move |node: &TraceNode| {
            if !exporter.record(trace_event(node)) {
                return Ok(());
            }
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    let exporter = Arc::clone(&exporter);
                    handle.spawn(async move {
                        match exporter.flush().await {
                            Ok(FlushOutcome::Sent(_)) => {
                                if let Err(err) = exporter.replay().await {
                                    log::warn!("trace batch replay failed: {err}");
                                }
                            }
                            Ok(_) => {}
                            Err(err) => log::warn!("trace batch flush failed: {err}"),
                        }
                    });
                }
                Err(_) => log::warn!(
                    "trace batch for {} is full but no tokio runtime is active; it stays pending",
                    node.span.id
                ),
            }
            Ok(())
        })
    }
}

/// Serialize a completed trace tree into one export event.
///
/// The body carries the root span plus the flattened span set, so a sink can
/// either replay the whole trace or index the individual spans.
pub fn trace_event(node: &TraceNode) -> ExportEvent {
    let spans = node.flatten();
    let timestamp = node
        .span
        .end_time
        .clone()
        .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
    ExportEvent::at(
        TRACE_EVENT_KIND,
        node.span.id.clone(),
        timestamp,
        serde_json::json!({
            "root": &node.span,
            "span_count": spans.len(),
            "spans": spans,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::span::{make_span, SpanKind};

    /// Sink that records every batch it is asked to deliver and fails on
    /// demand.
    struct RecordingSink {
        batches: Mutex<Vec<Vec<ExportEvent>>>,
        attempts: Mutex<usize>,
        /// Per-attempt script: `Err` entries are returned in order; once the
        /// script is exhausted the sink succeeds.
        failures: Mutex<Vec<ExportError>>,
    }

    impl RecordingSink {
        fn new(failures: Vec<ExportError>) -> Arc<Self> {
            Arc::new(Self {
                batches: Mutex::new(Vec::new()),
                attempts: Mutex::new(0),
                failures: Mutex::new(failures),
            })
        }

        fn batches(&self) -> Vec<Vec<ExportEvent>> {
            self.batches
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
        }

        fn attempts(&self) -> usize {
            *self.attempts.lock().unwrap_or_else(|p| p.into_inner())
        }
    }

    #[async_trait]
    impl BatchSink for RecordingSink {
        async fn send_batch(&self, batch: &[ExportEvent]) -> Result<(), ExportError> {
            *self.attempts.lock().unwrap_or_else(|p| p.into_inner()) += 1;
            let mut failures = self.failures.lock().unwrap_or_else(|p| p.into_inner());
            if !failures.is_empty() {
                let err = failures.remove(0);
                // A Fatal script entry is sticky: every later attempt fails too.
                if matches!(err, ExportError::Fatal(_)) {
                    failures.push(err.clone());
                }
                return Err(err);
            }
            drop(failures);
            self.batches
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(batch.to_vec());
            Ok(())
        }

        fn name(&self) -> &str {
            "recording"
        }
    }

    fn exporter(
        sink: Arc<RecordingSink>,
        dir: &std::path::Path,
        size: usize,
    ) -> Arc<BatchExporter> {
        let buffer = Arc::new(EdgeBuffer::open(dir, 16).unwrap());
        Arc::new(BatchExporter::new(
            sink,
            buffer,
            size,
            // Zero backoff keeps the retry tests instant.
            RetryPolicy::new(3, 0),
        ))
    }

    fn event(id: &str) -> ExportEvent {
        ExportEvent::at("test", id, "2026-01-01T00:00:00Z", serde_json::json!({}))
    }

    #[tokio::test]
    async fn sends_only_once_the_batch_is_full() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 3);

        exporter.enqueue(event("a")).await.unwrap();
        exporter.enqueue(event("b")).await.unwrap();
        assert!(
            sink.batches().is_empty(),
            "a partial batch must not hit the network"
        );
        assert_eq!(exporter.pending_len(), 2);

        let outcome = exporter.enqueue(event("c")).await.unwrap();
        assert_eq!(outcome, FlushOutcome::Sent(3));
        assert_eq!(exporter.pending_len(), 0, "a delivered batch is cleared");

        let batches = sink.batches();
        assert_eq!(batches.len(), 1);
        let ids: Vec<&str> = batches[0].iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c"], "batch keeps enqueue order");
    }

    #[tokio::test]
    async fn retries_a_transient_failure_then_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![
            ExportError::Retryable("connect refused".into()),
            ExportError::Retryable("503".into()),
        ]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);

        let outcome = exporter.enqueue(event("a")).await.unwrap();

        assert_eq!(outcome, FlushOutcome::Sent(1));
        assert_eq!(sink.attempts(), 3, "first try plus two retries");
        assert_eq!(exporter.buffer_len(), 0, "nothing parked after recovery");
    }

    #[tokio::test]
    async fn caps_attempts_on_a_persistent_transient_failure() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
        ]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);

        exporter.enqueue(event("a")).await.unwrap();

        assert_eq!(sink.attempts(), 3, "the retry cap bounds the attempts");
    }

    #[tokio::test]
    async fn a_fatal_failure_is_not_retried_and_not_parked() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![ExportError::Fatal("HTTP 401".into())]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);

        let outcome = exporter.enqueue(event("a")).await.unwrap();

        assert!(
            matches!(outcome, FlushOutcome::Dropped { events: 1, .. }),
            "got {outcome:?}"
        );
        assert_eq!(sink.attempts(), 1, "4xx is a contract error, not a blip");
        assert_eq!(
            exporter.buffer_len(),
            0,
            "a permanently-rejected batch must not wedge the queue"
        );
    }

    #[tokio::test]
    async fn an_unreachable_sink_parks_the_batch_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
        ]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);

        let outcome = exporter.enqueue(event("a")).await.unwrap();

        assert!(
            matches!(outcome, FlushOutcome::Buffered { events: 1, .. }),
            "got {outcome:?}"
        );
        assert_eq!(
            exporter.buffer_len(),
            1,
            "the batch landed on the disk queue"
        );
        assert!(sink.batches().is_empty());
    }

    #[tokio::test]
    async fn replay_redelivers_parked_batches_in_fifo_order() {
        let dir = tempfile::tempdir().unwrap();
        // Every attempt fails for the first two batches, then the sink recovers.
        let sink = RecordingSink::new(vec![
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
            ExportError::Retryable("down".into()),
        ]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);

        exporter.enqueue(event("first")).await.unwrap();
        exporter.enqueue(event("second")).await.unwrap();
        assert_eq!(exporter.buffer_len(), 2, "both batches parked");
        assert!(sink.batches().is_empty(), "nothing delivered while down");

        // Remote recovers.
        let replayed = exporter.replay().await.unwrap();

        assert_eq!(replayed, 2);
        assert_eq!(exporter.buffer_len(), 0, "the queue drained");
        let ids: Vec<String> = sink
            .batches()
            .iter()
            .map(|batch| batch[0].id.clone())
            .collect();
        assert_eq!(ids, vec!["first", "second"], "FIFO order is preserved");
    }

    #[tokio::test]
    async fn replay_stops_at_the_first_still_failing_batch() {
        let dir = tempfile::tempdir().unwrap();
        // Six failures park the two batches; three more make the first replay
        // attempt fail as well.
        let mut failures = Vec::new();
        for _ in 0..9 {
            failures.push(ExportError::Retryable("down".into()));
        }
        let sink = RecordingSink::new(failures);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 1);
        exporter.enqueue(event("first")).await.unwrap();
        exporter.enqueue(event("second")).await.unwrap();

        // The remote is still down: replay must not drop the backlog.
        let replayed = exporter.replay().await.unwrap();

        assert_eq!(replayed, 0);
        assert_eq!(exporter.buffer_len(), 2, "the queue is untouched");
    }

    #[tokio::test]
    async fn flush_keeps_a_below_size_batch_pending_until_it_fills() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![]);
        let exporter = exporter(Arc::clone(&sink), dir.path(), 4);

        // A below-threshold batch is kept pending, not pushed out partial.
        exporter.record(event("a"));
        let outcome = exporter.flush().await.unwrap();
        assert_eq!(outcome, FlushOutcome::Pending { queued: 1 });
        assert!(sink.batches().is_empty());

        // It keeps accumulating in-memory across records and flushes.
        exporter.record(event("b"));
        let outcome = exporter.enqueue(event("c")).await.unwrap();
        assert_eq!(outcome, FlushOutcome::Pending { queued: 3 });
        assert!(sink.batches().is_empty());

        // Filling the batch triggers delivery of the whole 4 in order.
        let outcome = exporter.enqueue(event("d")).await.unwrap();
        assert_eq!(outcome, FlushOutcome::Sent(4));
        let batches = sink.batches();
        let ids: Vec<&str> = batches[0].iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn trace_event_carries_the_root_and_the_flattened_span_set() {
        let root = make_span("root".into(), None, "agent", SpanKind::Agent);
        let mut child = make_span("child".into(), Some("root".into()), "llm", SpanKind::Llm);
        child.end_time = Some("2026-01-01T00:00:01Z".into());
        let mut root_ended = root.clone();
        root_ended.end_time = Some("2026-01-01T00:00:02Z".into());

        let node = TraceNode {
            span: root_ended,
            children: vec![TraceNode {
                span: child,
                children: vec![],
            }],
        };

        let event = trace_event(&node);

        assert_eq!(event.kind, TRACE_EVENT_KIND);
        assert_eq!(event.id, "root");
        assert_eq!(event.timestamp, "2026-01-01T00:00:02Z");
        assert_eq!(event.body["span_count"], 2);
        assert_eq!(event.body["root"]["id"], "root");
        assert_eq!(event.body["spans"][0]["id"], "root");
        assert_eq!(event.body["spans"][1]["id"], "child");
    }

    #[tokio::test]
    async fn trace_flush_callback_records_a_trace_without_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let sink = RecordingSink::new(vec![]);
        // A batch of 8 never fills on a single trace, so the callback only records.
        let exporter = exporter(Arc::clone(&sink), dir.path(), 8);
        let callback = exporter.trace_flush_callback();

        let root = make_span("root".into(), None, "agent", SpanKind::Agent);
        let node = TraceNode {
            span: root,
            children: vec![],
        };
        callback(&node).expect("the bridge callback never fails");

        assert_eq!(exporter.pending_len(), 1);
        assert!(sink.batches().is_empty());
    }
}
