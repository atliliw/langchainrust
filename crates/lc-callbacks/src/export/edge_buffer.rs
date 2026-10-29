//! Bounded on-disk FIFO queue for export batches that could not be delivered.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::batch::{ExportError, ExportEvent};

/// Default number of batches the on-disk queue keeps before it starts dropping
/// the oldest one.
pub const DEFAULT_BUFFER_ENTRIES: usize = 1024;

/// File-name prefix identifying a buffered batch.
const FILE_PREFIX: &str = "batch-";
/// File-name suffix for buffered batches.
const FILE_SUFFIX: &str = ".json";

/// Queue bookkeeping guarded by the buffer's mutex.
struct EdgeState {
    /// Sequence number handed to the next buffered batch.
    next_seq: u64,
    /// Number of batches discarded because the queue was at capacity.
    dropped: u64,
}

/// A bounded, on-disk FIFO queue of undelivered export batches.
///
/// Each batch is one JSON file named `batch-<seq>.json`; ordering is by the
/// zero-padded sequence number, so replaying the queue reproduces the order the
/// events were produced in. The queue survives a process restart: [`open`]
/// resumes numbering after the highest sequence already on disk.
///
/// # Overflow policy — drop oldest
///
/// When the queue already holds `max_entries` batches, the **oldest** batch is
/// deleted to make room for the new one. The alternative (rejecting the new
/// batch) would either stall the caller or lose the newest data, and the newest
/// trace is the one an operator is most likely to be looking at. Each drop is
/// counted in [`dropped`](EdgeBuffer::dropped) and logged, so the loss is
/// observable rather than silent.
///
/// # Blocking I/O
///
/// The queue uses `std::fs` deliberately: its writes are small and bounded, and
/// the trace-flush callback that feeds it is synchronous
/// ([`crate::tracing::backend::TraceFlushCallback`]), so the buffer has to work
/// without a tokio runtime in scope.
///
/// [`open`]: EdgeBuffer::open
pub struct EdgeBuffer {
    dir: PathBuf,
    max_entries: usize,
    state: Mutex<EdgeState>,
}

impl EdgeBuffer {
    /// Open (creating if needed) the queue rooted at `dir`.
    ///
    /// `max_entries` is clamped to at least 1. Existing batch files are left in
    /// place and numbering resumes after the highest sequence found, so a
    /// restarted process appends behind the backlog instead of overwriting it.
    pub fn open(dir: impl Into<PathBuf>, max_entries: usize) -> Result<Self, ExportError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| ExportError::Io(format!("create {}: {e}", dir.display())))?;
        let next_seq = Self::scan(&dir)?
            .last()
            .map(|(seq, _)| seq + 1)
            .unwrap_or(1);
        Ok(Self {
            dir,
            max_entries: max_entries.max(1),
            state: Mutex::new(EdgeState {
                next_seq,
                dropped: 0,
            }),
        })
    }

    /// Directory the queue writes into.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Configured batch capacity.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Number of batches discarded so far because the queue was full.
    pub fn dropped(&self) -> u64 {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).dropped
    }

    /// Append one batch, dropping the oldest batch when the queue is full.
    pub fn push(&self, batch: &[ExportEvent]) -> Result<(), ExportError> {
        let bytes = serde_json::to_vec(batch)
            .map_err(|e| ExportError::Io(format!("serialize buffered batch: {e}")))?;
        let seq = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let seq = state.next_seq;
            state.next_seq += 1;
            seq
        };
        let path = self
            .dir
            .join(format!("{FILE_PREFIX}{seq:020}{FILE_SUFFIX}"));
        std::fs::write(&path, &bytes)
            .map_err(|e| ExportError::Io(format!("write {}: {e}", path.display())))?;
        self.enforce_capacity()
    }

    /// Buffered batches in FIFO order, oldest first.
    ///
    /// A file that no longer parses as a batch can never be replayed, so it is
    /// dropped with a warning instead of wedging the queue forever.
    pub fn batches(&self) -> Result<Vec<(PathBuf, Vec<ExportEvent>)>, ExportError> {
        let mut out = Vec::new();
        for (_, path) in Self::scan(&self.dir)? {
            let bytes = std::fs::read(&path)
                .map_err(|e| ExportError::Io(format!("read {}: {e}", path.display())))?;
            match serde_json::from_slice::<Vec<ExportEvent>>(&bytes) {
                Ok(batch) => out.push((path, batch)),
                Err(err) => {
                    log::warn!(
                        "dropping unreadable export buffer file {}: {err}",
                        path.display()
                    );
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Ok(out)
    }

    /// Remove a batch file (after it was delivered). A missing file is not an
    /// error: another replay may already have consumed it.
    pub fn remove(&self, path: &Path) -> Result<(), ExportError> {
        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(ExportError::Io(format!("remove {}: {err}", path.display()))),
        }
    }

    /// Number of batches currently queued.
    pub fn len(&self) -> usize {
        match Self::scan(&self.dir) {
            Ok(files) => files.len(),
            Err(err) => {
                log::warn!("export buffer scan failed: {err}");
                0
            }
        }
    }

    /// Whether the queue holds no batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Delete oldest batches until the queue is within capacity.
    fn enforce_capacity(&self) -> Result<(), ExportError> {
        loop {
            let files = Self::scan(&self.dir)?;
            if files.len() <= self.max_entries {
                return Ok(());
            }
            let (_, oldest) = &files[0];
            std::fs::remove_file(oldest)
                .map_err(|e| ExportError::Io(format!("drop {}: {e}", oldest.display())))?;
            let dropped = {
                let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                state.dropped += 1;
                state.dropped
            };
            log::warn!(
                "export edge buffer at capacity ({}); dropped oldest batch {} (total dropped: {})",
                self.max_entries,
                oldest.display(),
                dropped
            );
        }
    }

    /// Batch files in ascending sequence order.
    fn scan(dir: &Path) -> Result<Vec<(u64, PathBuf)>, ExportError> {
        let mut out = Vec::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| ExportError::Io(format!("read {}: {e}", dir.display())))?;
        for entry in entries {
            let entry = entry.map_err(|e| ExportError::Io(e.to_string()))?;
            let name = entry.file_name();
            if let Some(seq) = parse_seq(&name.to_string_lossy()) {
                out.push((seq, entry.path()));
            }
        }
        out.sort_by_key(|(seq, _)| *seq);
        Ok(out)
    }
}

/// Extract the sequence number from a `batch-<seq>.json` file name.
fn parse_seq(name: &str) -> Option<u64> {
    name.strip_prefix(FILE_PREFIX)?
        .strip_suffix(FILE_SUFFIX)?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: &str) -> ExportEvent {
        ExportEvent::at("test", id, "2026-01-01T00:00:00Z", serde_json::json!({}))
    }

    fn batch(ids: &[&str]) -> Vec<ExportEvent> {
        ids.iter().map(|id| event(id)).collect()
    }

    #[test]
    fn push_then_batches_round_trips_in_fifo_order() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = EdgeBuffer::open(dir.path(), 8).unwrap();

        buffer.push(&batch(&["a1", "a2"])).unwrap();
        buffer.push(&batch(&["b1"])).unwrap();

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
        let queued = buffer.batches().unwrap();
        assert_eq!(queued.len(), 2);
        assert_eq!(queued[0].1[0].id, "a1", "oldest batch first");
        assert_eq!(queued[0].1[1].id, "a2");
        assert_eq!(queued[1].1[0].id, "b1");
    }

    #[test]
    fn remove_deletes_only_the_named_batch() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = EdgeBuffer::open(dir.path(), 8).unwrap();
        buffer.push(&batch(&["a"])).unwrap();
        buffer.push(&batch(&["b"])).unwrap();

        let queued = buffer.batches().unwrap();
        buffer.remove(&queued[0].0).unwrap();

        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.batches().unwrap()[0].1[0].id, "b");
        // Removing an already-gone file is a no-op, not an error.
        buffer.remove(&queued[0].0).unwrap();
    }

    #[test]
    fn drops_oldest_batch_when_at_capacity() {
        // Overflow policy: the *oldest* batch is sacrificed, never the newest.
        let dir = tempfile::tempdir().unwrap();
        let buffer = EdgeBuffer::open(dir.path(), 2).unwrap();

        buffer.push(&batch(&["oldest"])).unwrap();
        buffer.push(&batch(&["middle"])).unwrap();
        buffer.push(&batch(&["newest"])).unwrap();

        let queued = buffer.batches().unwrap();
        assert_eq!(queued.len(), 2, "capacity is enforced");
        assert_eq!(queued[0].1[0].id, "middle");
        assert_eq!(queued[1].1[0].id, "newest");
        assert_eq!(buffer.dropped(), 1);
    }

    #[test]
    fn reopen_resumes_numbering_after_the_existing_backlog() {
        let dir = tempfile::tempdir().unwrap();
        {
            let buffer = EdgeBuffer::open(dir.path(), 8).unwrap();
            buffer.push(&batch(&["first"])).unwrap();
            buffer.push(&batch(&["second"])).unwrap();
        }
        // A restarted process must append *behind* the backlog, not overwrite it.
        let reopened = EdgeBuffer::open(dir.path(), 8).unwrap();
        reopened.push(&batch(&["third"])).unwrap();

        let ids: Vec<String> = reopened
            .batches()
            .unwrap()
            .into_iter()
            .map(|(_, b)| b[0].id.clone())
            .collect();
        assert_eq!(ids, vec!["first", "second", "third"]);
    }

    #[test]
    fn unreadable_files_are_dropped_instead_of_wedging_the_queue() {
        let dir = tempfile::tempdir().unwrap();
        let buffer = EdgeBuffer::open(dir.path(), 8).unwrap();
        buffer.push(&batch(&["good"])).unwrap();
        std::fs::write(
            dir.path().join("batch-00000000000000000099.json"),
            b"{not json",
        )
        .unwrap();

        let queued = buffer.batches().unwrap();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].1[0].id, "good");
        assert!(!dir.path().join("batch-00000000000000000099.json").exists());
    }
}
