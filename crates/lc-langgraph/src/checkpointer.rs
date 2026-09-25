// crates/lc-langgraph/src/checkpointer.rs
//! Checkpointing for state persistence

use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Name of the implicit thread every checkpoint saved through the legacy (non
/// threaded) API lives under. Backends that do not model threads treat every
/// checkpoint as belonging to this thread.
pub const DEFAULT_THREAD: &str = "default";

/// Checkpointer trait for state persistence.
///
/// The trait grew a "threaded" family of methods in 0.25.0: each carries a
/// `thread` (conversation / workflow lineage) so a single checkpointer can host
/// many interleaved runs without one thread's checkpoints leaking into another.
/// Every threaded method has a default implementation that delegates to the
/// legacy single-threaded method of the same name, so existing implementors are
/// unaffected unless they opt into threading.
#[async_trait]
pub trait Checkpointer<S: StateSchema>: Send + Sync {
    /// Insert a new checkpoint, recording how much of the recursion budget had
    /// been consumed by the run at that point (M6).
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String>;
    /// Load the state saved under the given checkpoint id.
    async fn load(&self, checkpoint_id: &str) -> GraphResult<S>;
    /// List checkpoint ids, ordered from oldest to most recent (H5).
    async fn list(&self) -> GraphResult<Vec<String>>;
    /// Delete the checkpoint with the given id.
    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()>;
    /// State and recursion budget of the most recently saved checkpoint.
    async fn last(&self) -> GraphResult<Option<(S, usize)>>;

    /// List every checkpoint as a full snapshot (state + ordering keys +
    /// recursion budget), oldest first.
    ///
    /// Used by [`CompiledGraph::get_state_history`](crate::compiled::CompiledGraph::get_state_history)
    /// and time-travel forks. The default reconstructs a state-only history from
    /// [`list`](Self::list) + [`load`](Self::load) and reports an unknown
    /// `timestamp`/`seq`/`recursion_count` of `0`; backends that store the full
    /// [`CheckpointData`] (or its columns) override this to report the real
    /// ordering keys and budget.
    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let mut snaps = Vec::new();
        for id in self.list().await? {
            let state = self.load(&id).await?;
            snaps.push(CheckpointInfo {
                id,
                timestamp: 0,
                seq: 0,
                recursion_count: 0,
                state,
                parent: None,
            });
        }
        Ok(snaps)
    }

    /// Replace the state stored in an existing checkpoint (the LangGraph
    /// `updateState` analogue), using optimistic concurrency control.
    ///
    /// `expected_version` is the version of the state the edit is based on
    /// (every fresh checkpoint starts at version `1`, and each successful
    /// `update_state` bumps it). When the stored version has moved on because
    /// another writer edited the checkpoint first, the call fails with
    /// [`GraphError::CheckpointVersionConflict`] instead of overwriting that
    /// edit. Returns the new version.
    ///
    /// Backends without edit support keep the default implementation, which
    /// fails with [`GraphError::CheckpointError`].
    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        let _ = (state, expected_version);
        Err(GraphError::CheckpointError(format!(
            "update_state is not supported on checkpoint '{checkpoint_id}' by this checkpointer",
        )))
    }

    // ------------------------------------------------------------------
    // Threaded API (B4): defaults delegate to the legacy global semantics.
    // ------------------------------------------------------------------

    /// Save a checkpoint under `thread`. Default: delegate to [`save`](Self::save)
    /// (which, for thread-bound backends, already routes to the bound thread).
    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        let _ = thread;
        self.save(state, recursion_count).await
    }

    /// Save a checkpoint under `thread`, stamped as forked from `parent_id`
    /// (the checkpoint this lineage branches from). Default: delegate to
    /// [`save_threaded`](Self::save_threaded), discarding the parent link.
    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        let _ = parent_id;
        self.save_threaded(thread, state, recursion_count).await
    }

    /// Load a checkpoint saved under `thread`. Default: delegate to
    /// [`load`](Self::load).
    async fn load_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<S> {
        let _ = thread;
        self.load(checkpoint_id).await
    }

    /// List checkpoints under `thread`, oldest first. Default: delegate to
    /// [`list`](Self::list).
    async fn list_threaded(&self, thread: &str) -> GraphResult<Vec<String>> {
        let _ = thread;
        self.list().await
    }

    /// Delete a checkpoint saved under `thread`. Default: delegate to
    /// [`delete`](Self::delete).
    async fn delete_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<()> {
        let _ = thread;
        self.delete(checkpoint_id).await
    }

    /// Most recent checkpoint under `thread`. Default: delegate to
    /// [`last`](Self::last).
    async fn last_threaded(&self, thread: &str) -> GraphResult<Option<(S, usize)>> {
        let _ = thread;
        self.last().await
    }

    /// Full snapshots under `thread`, oldest first. Default: delegate to
    /// [`snapshots`](Self::snapshots).
    async fn snapshots_threaded(&self, thread: &str) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let _ = thread;
        self.snapshots().await
    }

    /// Return the (thread, parent) lineage of the checkpoint with the given id,
    /// when the backend can report it. The default reports no lineage info.
    async fn checkpoint_lineage(
        &self,
        checkpoint_id: &str,
    ) -> GraphResult<Option<(String, Option<String>)>> {
        let _ = checkpoint_id;
        Ok(None)
    }
}

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: StateSchema")]
pub struct CheckpointData<S: StateSchema> {
    /// Unique identifier of the checkpoint.
    pub id: String,
    /// The state snapshot stored in the checkpoint.
    pub state: S,
    /// Unix timestamp (seconds) when the checkpoint was created.
    pub timestamp: i64,
    /// Arbitrary metadata associated with the checkpoint.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Monotonic sequence number assigned by the checkpointer on save. Breaks
    /// ties between checkpoints saved within the same `timestamp` second, so
    /// "most recent" is well-defined even for fast back-to-back saves (H5).
    #[serde(default)]
    pub seq: u64,
    /// Recursion budget consumed when this checkpoint was taken, so a resume
    /// continues counting against the same `recursion_limit` instead of
    /// restarting from zero (M6).
    #[serde(default)]
    pub recursion_count: usize,
    /// Optimistic-concurrency version: `1` for a fresh checkpoint, bumped on
    /// every [`Checkpointer::update_state`]. Backed by an atomic compare-and
    /// swap in the durable checkpointers.
    #[serde(default = "initial_version")]
    pub version: u64,
    /// Thread (conversation / workflow lineage) this checkpoint belongs to.
    /// `None` means it lives under [`DEFAULT_THREAD`].
    #[serde(default)]
    pub thread_id: Option<String>,
    /// Id of the checkpoint this one was forked from, or `None` for a root run.
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Fresh checkpoints start at version 1 (0 only occurs in pre-0.22.4 files).
fn initial_version() -> u64 {
    1
}

/// A single snapshot in a graph's execution history: the state captured at one
/// checkpoint, together with the ordering keys and the recursion budget consumed
/// up to that point (used to fork a fresh timeline from an old snapshot).
#[derive(Debug, Clone)]
pub struct CheckpointInfo<S: StateSchema> {
    /// Unique id of the checkpoint this snapshot came from.
    pub id: String,
    /// Unix timestamp (seconds) when the checkpoint was created.
    pub timestamp: i64,
    /// Sequence assigned by the checkpointer; breaks same-second ties.
    pub seq: u64,
    /// Recursion budget consumed when the snapshot was captured. A fork seeded
    /// from this snapshot continues counting against this budget.
    pub recursion_count: usize,
    /// The state snapshot.
    pub state: S,
    /// Id of the checkpoint this snapshot was forked from, when the backend can
    /// report it (fork lineage).
    pub parent: Option<String>,
}

impl<S: StateSchema> CheckpointData<S> {
    /// Create a new checkpoint for the given state.
    pub fn new(state: S) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            state,
            timestamp: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
            seq: 0,
            recursion_count: 0,
            version: 1,
            thread_id: None,
            parent_id: None,
        }
    }

    /// Construct with the checkpointer-assigned sequence and the run's current
    /// recursion budget.
    pub fn with_progress(state: S, seq: u64, recursion_count: usize) -> Self {
        let mut data = Self::new(state);
        data.seq = seq;
        data.recursion_count = recursion_count;
        data
    }
}

/// Per-thread checkpoint storage for the in-memory checkpointers. Each thread
/// keeps its own buffered history and a monotonic sequence counter, so two
/// threads interleaving their saves never observe each other's checkpoints.
struct ThreadStore<S: StateSchema> {
    items: VecDeque<CheckpointData<S>>,
    seq: AtomicU64,
}

impl<S: StateSchema> ThreadStore<S> {
    fn new() -> Self {
        Self {
            items: VecDeque::new(),
            seq: AtomicU64::new(0),
        }
    }

    fn find(&self, id: &str) -> Option<&CheckpointData<S>> {
        self.items.iter().find(|d| d.id == id)
    }

    fn find_mut(&mut self, id: &str) -> Option<&mut CheckpointData<S>> {
        self.items.iter_mut().find(|d| d.id == id)
    }

    /// Sorted references (by timestamp, then seq) — the canonical order for
    /// `list` / `snapshots`.
    fn sorted(&self) -> Vec<&CheckpointData<S>> {
        let mut v: Vec<&CheckpointData<S>> = self.items.iter().collect();
        v.sort_by_key(|d| (d.timestamp, d.seq));
        v
    }

    fn last(&self) -> Option<&CheckpointData<S>> {
        self.items.iter().max_by_key(|d| (d.timestamp, d.seq))
    }
}

type ThreadMap<S> = HashMap<String, ThreadStore<S>>;

async fn mem_save<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
    state: &S,
    recursion_count: usize,
) -> GraphResult<String> {
    let mut guard = threads.lock().await;
    let store = guard
        .entry(thread.to_string())
        .or_insert_with(ThreadStore::new);
    let seq = store.seq.fetch_add(1, Ordering::SeqCst);
    let mut data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
    data.thread_id = Some(thread.to_string());
    let id = data.id.clone();
    store.items.push_back(data);
    Ok(id)
}

async fn mem_save_fork<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    parent_id: Option<&str>,
    thread: &str,
    state: &S,
    recursion_count: usize,
) -> GraphResult<String> {
    let mut guard = threads.lock().await;
    let store = guard
        .entry(thread.to_string())
        .or_insert_with(ThreadStore::new);
    let seq = store.seq.fetch_add(1, Ordering::SeqCst);
    let mut data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
    data.thread_id = Some(thread.to_string());
    data.parent_id = parent_id.map(ToOwned::to_owned);
    let id = data.id.clone();
    store.items.push_back(data);
    Ok(id)
}

async fn mem_load<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
    checkpoint_id: &str,
) -> GraphResult<S> {
    let guard = threads.lock().await;
    let store = guard.get(thread).ok_or_else(|| {
        GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
    })?;
    store
        .find(checkpoint_id)
        .map(|d| d.state.clone())
        .ok_or_else(|| {
            GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
        })
}

async fn mem_list<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
) -> GraphResult<Vec<String>> {
    let guard = threads.lock().await;
    Ok(guard
        .get(thread)
        .map(|s| s.sorted().iter().map(|d| d.id.clone()).collect())
        .unwrap_or_default())
}

async fn mem_snapshots<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
) -> GraphResult<Vec<CheckpointInfo<S>>> {
    let guard = threads.lock().await;
    Ok(guard
        .get(thread)
        .map(|s| {
            s.sorted()
                .iter()
                .map(|d| checkpoint_info_from_data(*d))
                .collect()
        })
        .unwrap_or_default())
}

async fn mem_last<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
) -> GraphResult<Option<(S, usize)>> {
    let guard = threads.lock().await;
    Ok(guard
        .get(thread)
        .and_then(|s| s.last())
        .map(|d| (d.state.clone(), d.recursion_count)))
}

async fn mem_delete<S: StateSchema>(
    threads: &Mutex<ThreadMap<S>>,
    thread: &str,
    checkpoint_id: &str,
) -> GraphResult<()> {
    let mut guard = threads.lock().await;
    if let Some(store) = guard.get_mut(thread) {
        store.items.retain(|d| d.id != checkpoint_id);
    }
    Ok(())
}

/// Build a [`CheckpointInfo`] from stored [`CheckpointData`] (used by the
/// in-memory checkpointers' `snapshots`).
fn checkpoint_info_from_data<S: StateSchema>(d: &CheckpointData<S>) -> CheckpointInfo<S> {
    CheckpointInfo {
        id: d.id.clone(),
        timestamp: d.timestamp,
        seq: d.seq,
        recursion_count: d.recursion_count,
        state: d.state.clone(),
        parent: d.parent_id.clone(),
    }
}

/// In-memory OCC edit shared by all the in-memory checkpointers: bump the
/// version only while holding the map lock, so two concurrent edits cannot both
/// succeed against the same base version. The checkpoint is searched across
/// every thread (ids are globally unique UUIDs).
async fn update_locked<S: StateSchema>(
    checkpoints: &Mutex<ThreadMap<S>>,
    checkpoint_id: &str,
    state: &S,
    expected_version: u64,
) -> GraphResult<u64> {
    let mut guard = checkpoints.lock().await;
    for store in guard.values_mut() {
        if let Some(data) = store.find_mut(checkpoint_id) {
            if data.version != expected_version {
                return Err(GraphError::CheckpointVersionConflict {
                    checkpoint_id: checkpoint_id.to_string(),
                    expected: expected_version,
                    actual: data.version,
                });
            }
            data.state = state.clone();
            data.version += 1;
            return Ok(data.version);
        }
    }
    Err(GraphError::CheckpointError(format!(
        "Checkpoint '{checkpoint_id}' not found"
    )))
}

/// In-memory checkpointer for development
pub struct MemoryCheckpointer<S: StateSchema> {
    threads: Mutex<ThreadMap<S>>,
}

impl<S: StateSchema> MemoryCheckpointer<S> {
    /// Create a new empty in-memory checkpointer.
    pub fn new() -> Self {
        Self {
            threads: Mutex::new(HashMap::new()),
        }
    }
}

impl<S: StateSchema> Default for MemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for MemoryCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        self.save_threaded(DEFAULT_THREAD, state, recursion_count)
            .await
    }

    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        mem_save(&self.threads, thread, state, recursion_count).await
    }

    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        mem_save_fork(&self.threads, parent_id, thread, state, recursion_count).await
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        self.load_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn load_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<S> {
        mem_load(&self.threads, thread, checkpoint_id).await
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        self.list_threaded(DEFAULT_THREAD).await
    }

    async fn list_threaded(&self, thread: &str) -> GraphResult<Vec<String>> {
        mem_list(&self.threads, thread).await
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        self.last_threaded(DEFAULT_THREAD).await
    }

    async fn last_threaded(&self, thread: &str) -> GraphResult<Option<(S, usize)>> {
        mem_last(&self.threads, thread).await
    }

    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        self.snapshots_threaded(DEFAULT_THREAD).await
    }

    async fn snapshots_threaded(&self, thread: &str) -> GraphResult<Vec<CheckpointInfo<S>>> {
        mem_snapshots(&self.threads, thread).await
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        update_locked(&self.threads, checkpoint_id, state, expected_version).await
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.delete_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn delete_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<()> {
        mem_delete(&self.threads, thread, checkpoint_id).await
    }

    async fn checkpoint_lineage(
        &self,
        checkpoint_id: &str,
    ) -> GraphResult<Option<(String, Option<String>)>> {
        let guard = self.threads.lock().await;
        for (thread, store) in guard.iter() {
            if let Some(d) = store.find(checkpoint_id) {
                return Ok(Some((thread.clone(), d.parent_id.clone())));
            }
        }
        Ok(None)
    }
}

/// Thread-safe memory checkpointer
pub struct ThreadSafeMemoryCheckpointer<S: StateSchema> {
    threads: Mutex<ThreadMap<S>>,
}

impl<S: StateSchema> ThreadSafeMemoryCheckpointer<S> {
    /// Create a new empty thread-safe memory checkpointer.
    pub fn new() -> Self {
        Self {
            threads: Mutex::new(HashMap::new()),
        }
    }
}

impl<S: StateSchema> Default for ThreadSafeMemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for ThreadSafeMemoryCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        self.save_threaded(DEFAULT_THREAD, state, recursion_count)
            .await
    }

    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        mem_save(&self.threads, thread, state, recursion_count).await
    }

    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        mem_save_fork(&self.threads, parent_id, thread, state, recursion_count).await
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        self.load_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn load_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<S> {
        mem_load(&self.threads, thread, checkpoint_id).await
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        self.list_threaded(DEFAULT_THREAD).await
    }

    async fn list_threaded(&self, thread: &str) -> GraphResult<Vec<String>> {
        mem_list(&self.threads, thread).await
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        self.last_threaded(DEFAULT_THREAD).await
    }

    async fn last_threaded(&self, thread: &str) -> GraphResult<Option<(S, usize)>> {
        mem_last(&self.threads, thread).await
    }

    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        self.snapshots_threaded(DEFAULT_THREAD).await
    }

    async fn snapshots_threaded(&self, thread: &str) -> GraphResult<Vec<CheckpointInfo<S>>> {
        mem_snapshots(&self.threads, thread).await
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        update_locked(&self.threads, checkpoint_id, state, expected_version).await
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.delete_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn delete_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<()> {
        mem_delete(&self.threads, thread, checkpoint_id).await
    }

    async fn checkpoint_lineage(
        &self,
        checkpoint_id: &str,
    ) -> GraphResult<Option<(String, Option<String>)>> {
        let guard = self.threads.lock().await;
        for (thread, store) in guard.iter() {
            if let Some(d) = store.find(checkpoint_id) {
                return Ok(Some((thread.clone(), d.parent_id.clone())));
            }
        }
        Ok(None)
    }
}

/// Validate a single path segment (a checkpoint id or thread) — rejects empty
/// segments and anything that could traverse out of the checkpoint directory.
fn validate_segment(seg: &str) -> GraphResult<()> {
    if seg.is_empty() {
        return Err(GraphError::CheckpointError(
            "checkpoint path segment must not be empty".to_string(),
        ));
    }
    if seg.contains("..") || seg.contains('/') || seg.contains('\\') {
        return Err(GraphError::CheckpointError(format!(
            "Invalid checkpoint identifier '{seg}': path traversal detected"
        )));
    }
    if std::path::Path::new(seg).is_absolute() {
        return Err(GraphError::CheckpointError(format!(
            "Invalid checkpoint identifier '{seg}': absolute path not allowed"
        )));
    }
    Ok(())
}

/// H6: seed the `next_seq` counter from the highest `<seq>-` filename prefix found
/// under the checkpointer's base directory (best-effort, sync scan at construction)
/// so a resumed process does not reuse seq numbers already on disk. The scan covers
/// the base directory itself and one level of thread subdirectories
/// (`<base>/<thread>/<seq>-<id>.json`), since every thread shares the single
/// `next_seq` counter. The parse tolerates a legacy `<id>.json` name with no numeric
/// prefix (treated as seq 0) and an absent / unreadable directory (stays at 0).
fn seed_seq_from_dir(dir: &std::path::Path) -> u64 {
    let mut max_seq: u64 = 0;
    let scan = |d: &std::path::Path, max_seq: &mut u64| {
        let Ok(entries) = std::fs::read_dir(d) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            // `{seq}-{id}.json` → parse the leading numeric run.
            let Some(dash) = name.find('-') else { continue };
            if let Ok(seq) = name[..dash].parse::<u64>() {
                *max_seq = (*max_seq).max(seq);
            }
        }
    };
    scan(dir, &mut max_seq);
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                scan(&entry.path(), &mut max_seq);
            }
        }
    }
    max_seq
}

/// File-based checkpointer for persistent storage.
///
/// New checkpoints are written under `<base>/<thread>/<seq>-<id>.json`. Checkpoints
/// written by older versions, which live flat at `<base>/<id>.json`, are folded
/// lazily into the default thread (read-only: they are listed and loadable but
/// never rewritten in place).
pub struct FileCheckpointer<S: StateSchema> {
    directory: std::path::PathBuf,
    next_seq: AtomicU64,
    /// Serializes the read-check-write critical section of `update_state`
    /// within this process (cross-process OCC is provided by the SQLite /
    /// Postgres / Redis backends, not by plain JSON files).
    update_lock: Mutex<()>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> FileCheckpointer<S> {
    /// Create a file-based checkpointer that persists checkpoints under the given directory.
    pub fn new(directory: impl Into<std::path::PathBuf>) -> GraphResult<Self> {
        let dir = directory.into();
        if !dir.exists() {
            std::fs::create_dir_all(&dir).map_err(|e| {
                GraphError::CheckpointError(format!(
                    "Failed to create directory '{}': {}",
                    dir.display(),
                    e
                ))
            })?;
        }
        // H6: seed the sequence counter from the max seq already on disk instead of
        // always restarting at 0. A checkpointer that persist/restores across
        // processes must not reuse seq numbers (each new checkpoint's `{seq}-{id}.json`
        // filename and its `data.seq` would collide with prior ones, corrupting
        // `sorted_ids`' ordering and the snapshot lineage). Best-effort: a scan error
        // (or an empty dir) leaves the counter at 0.
        let next_seq = seed_seq_from_dir(&dir);
        Ok(Self {
            directory: dir,
            next_seq: AtomicU64::new(next_seq),
            update_lock: Mutex::new(()),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Legacy flat layout `<base>/<id>.json` (written by versions before the
    /// thread-aware layout).
    fn legacy_flat_path(&self, id: &str) -> GraphResult<PathBuf> {
        validate_segment(id)?;
        Ok(self.directory.join(format!("{id}.json")))
    }

    /// Sanitize and resolve a thread's subdirectory `<base>/<thread>`.
    fn thread_dir(&self, thread: &str) -> GraphResult<PathBuf> {
        validate_segment(thread)?;
        Ok(self.directory.join(thread))
    }

    /// Find the checkpoint file for `id` inside `dir`, matching the
    /// `<seq>-<id>.json` layout.
    async fn find_file(&self, dir: &std::path::Path, id: &str) -> GraphResult<Option<PathBuf>> {
        if !dir.exists() {
            return Ok(None);
        }
        let suffix = format!("-{id}");
        let mut entries = tokio::fs::read_dir(dir)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir error: {}", e)))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir entry error: {}", e)))?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.ends_with(&suffix))
            {
                return Ok(Some(path));
            }
        }
        Ok(None)
    }

    /// Resolve a checkpoint file + its parsed data, checking the legacy flat
    /// layout first (default thread only) then the thread subdirectory.
    async fn resolve_data(
        &self,
        thread: &str,
        id: &str,
    ) -> GraphResult<Option<(PathBuf, CheckpointData<S>)>> {
        validate_segment(id)?;
        validate_segment(thread)?;
        if thread == DEFAULT_THREAD {
            let legacy = self.legacy_flat_path(id)?;
            if legacy.exists() {
                let json = tokio::fs::read_to_string(&legacy)
                    .await
                    .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
                let data: CheckpointData<S> = serde_json::from_str(&json).map_err(|e| {
                    GraphError::CheckpointError(format!("Deserialize error: {}", e))
                })?;
                return Ok(Some((legacy, data)));
            }
        }
        let dir = self.thread_dir(thread)?;
        if let Some(path) = self.find_file(&dir, id).await? {
            let json = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
            let data: CheckpointData<S> = serde_json::from_str(&json)
                .map_err(|e| GraphError::CheckpointError(format!("Deserialize error: {}", e)))?;
            return Ok(Some((path, data)));
        }
        Ok(None)
    }

    /// Atomic write: write `*.json.tmp`, fsync it, rename over the real file, then
    /// fsync the parent directory (H5). The `.tmp` extension is never picked up by any
    /// `.json` scanner.
    async fn atomic_write(&self, path: &std::path::Path, json: &str) -> GraphResult<()> {
        let tmp_path = path.with_extension("json.tmp");
        // H5: write + `sync_all` before the rename, so a crash / power loss after the
        // rename cannot leave the target file as a torn page of the tmp write. Without
        // the fsync, the rename may persist to the directory journal ahead of the file's
        // data blocks, and the renamed checkpoint reads back corrupted.
        {
            let mut f = tokio::fs::File::create(&tmp_path)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Create error: {}", e)))?;
            f.write_all(json.as_bytes())
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Write error: {}", e)))?;
            f.sync_all()
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Fsync error: {}", e)))?;
        }
        tokio::fs::rename(&tmp_path, path)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Atomic rename error: {}", e)))?;
        // H5: fsync the parent directory so the rename itself is durable — otherwise a
        // crash can still roll the directory (new filename) back. On platforms where
        // directory fsync is unsupported (some filesystems), treat failure as recoverable.
        if let Some(parent) = path.parent() {
            if let Ok(dir) = tokio::fs::File::open(parent).await {
                let _ = dir.sync_all().await;
            }
        }
        Ok(())
    }

    /// Collect `(timestamp, seq, id)` sort keys for every checkpoint reachable
    /// under `thread`: for the default thread this includes legacy flat files at
    /// the base directory plus the `default/` subdirectory; any other thread
    /// only looks in its own subdirectory.
    async fn sorted_ids(&self, thread: &str) -> GraphResult<Vec<(i64, u64, String)>> {
        validate_segment(thread)?;
        let mut dirs: Vec<PathBuf> = Vec::new();
        if thread == DEFAULT_THREAD {
            dirs.push(self.directory.clone());
        }
        dirs.push(self.thread_dir(thread)?);

        let mut items: Vec<(i64, u64, String)> = Vec::new();
        for dir in dirs {
            if !dir.exists() {
                continue;
            }
            let mut entries = tokio::fs::read_dir(&dir)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Read dir error: {}", e)))?;
            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Read dir entry error: {}", e)))?
            {
                let path = entry.path();
                if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let json = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
                let data: CheckpointData<S> = serde_json::from_str(&json).map_err(|e| {
                    GraphError::CheckpointError(format!("Deserialize error: {}", e))
                })?;
                items.push((data.timestamp, data.seq, data.id));
            }
        }
        // H5: sort by (timestamp, seq) ascending; seq breaks ties within the same second.
        items.sort();
        Ok(items)
    }
}

// NOTE: `Default` is intentionally NOT implemented for `FileCheckpointer` (Q1).
// The default constructor would have to create the `.checkpoints` directory, which
// is I/O that can fail (read-only cwd, disk full, permissions) — `Default` cannot
// report that failure, so it would have to panic. Use `FileCheckpointer::new(...)`
// which returns a `GraphResult` and surfaces the error instead.

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for FileCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        self.save_threaded(DEFAULT_THREAD, state, recursion_count)
            .await
    }

    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        validate_segment(thread)?;
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let mut data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
        data.thread_id = Some(thread.to_string());
        let id = data.id.clone();
        let dir = self.thread_dir(thread)?;
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Create dir error: {}", e)))?;
        let path = dir.join(format!("{seq}-{id}.json"));
        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;
        self.atomic_write(&path, &json).await?;
        Ok(id)
    }

    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        validate_segment(thread)?;
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let mut data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
        data.thread_id = Some(thread.to_string());
        data.parent_id = parent_id.map(ToOwned::to_owned);
        let id = data.id.clone();
        let dir = self.thread_dir(thread)?;
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Create dir error: {}", e)))?;
        let path = dir.join(format!("{seq}-{id}.json"));
        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;
        self.atomic_write(&path, &json).await?;
        Ok(id)
    }

    async fn load_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<S> {
        let (_path, data) = self
            .resolve_data(thread, checkpoint_id)
            .await?
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{}' not found", checkpoint_id))
            })?;
        Ok(data.state)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        self.load_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn list_threaded(&self, thread: &str) -> GraphResult<Vec<String>> {
        Ok(self
            .sorted_ids(thread)
            .await?
            .into_iter()
            .map(|(_, _, id)| id)
            .collect())
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        self.list_threaded(DEFAULT_THREAD).await
    }

    async fn last_threaded(&self, thread: &str) -> GraphResult<Option<(S, usize)>> {
        let items = self.sorted_ids(thread).await?;
        let Some((_, _, last_id)) = items.into_iter().last() else {
            return Ok(None);
        };
        let (_path, data) = self.resolve_data(thread, &last_id).await?.ok_or_else(|| {
            GraphError::CheckpointError(format!("Checkpoint '{}' not found", last_id))
        })?;
        Ok(Some((data.state, data.recursion_count)))
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        self.last_threaded(DEFAULT_THREAD).await
    }

    async fn snapshots_threaded(&self, thread: &str) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let ids = self.sorted_ids(thread).await?;
        let mut snaps = Vec::with_capacity(ids.len());
        for (_, _, id) in ids {
            let (_path, data) = self.resolve_data(thread, &id).await?.ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{id}' not found"))
            })?;
            snaps.push(checkpoint_info_from_data(&data));
        }
        Ok(snaps)
    }

    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        self.snapshots_threaded(DEFAULT_THREAD).await
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        let _guard = self.update_lock.lock().await;
        let (resolved_path, mut data) = self
            .resolve_data(DEFAULT_THREAD, checkpoint_id)
            .await?
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
            })?;
        if data.version != expected_version {
            return Err(GraphError::CheckpointVersionConflict {
                checkpoint_id: checkpoint_id.to_string(),
                expected: expected_version,
                actual: data.version,
            });
        }
        data.state = state.clone();
        data.version += 1;

        let dir = self.thread_dir(DEFAULT_THREAD)?;
        let path = dir.join(format!("{}-{}.json", data.seq, checkpoint_id));
        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;
        self.atomic_write(&path, &json).await?;

        // Fold a migrated legacy flat checkpoint out now that the authoritative
        // copy lives under the default-thread directory. Leaving the flat file
        // behind made the id appear twice in `sorted_ids` and — because
        // `resolve_data` prefers the legacy path — made this edit invisible to
        // `load`/`last`/`snapshots` (silent stale reads).
        if resolved_path == self.legacy_flat_path(checkpoint_id)? {
            tokio::fs::remove_file(&resolved_path)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Delete legacy file: {}", e)))?;
        }

        Ok(data.version)
    }

    async fn delete_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<()> {
        validate_segment(thread)?;
        validate_segment(checkpoint_id)?;
        if thread == DEFAULT_THREAD {
            let legacy = self.legacy_flat_path(checkpoint_id)?;
            if legacy.exists() {
                tokio::fs::remove_file(&legacy)
                    .await
                    .map_err(|e| GraphError::CheckpointError(format!("Delete error: {}", e)))?;
            }
        }
        let dir = self.thread_dir(thread)?;
        if let Some(path) = self.find_file(&dir, checkpoint_id).await? {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Delete error: {}", e)))?;
        }
        Ok(())
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.delete_threaded(DEFAULT_THREAD, checkpoint_id).await
    }

    async fn checkpoint_lineage(
        &self,
        checkpoint_id: &str,
    ) -> GraphResult<Option<(String, Option<String>)>> {
        // Probe the default thread first, then any thread that has a matching file.
        let mut candidates: Vec<String> = vec![DEFAULT_THREAD.to_string()];
        if let Ok(dirs) = tokio::fs::read_dir(&self.directory).await {
            let mut entries = dirs;
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry.path().is_dir() {
                    candidates.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        for thread in candidates {
            if let Some((_path, data)) = self.resolve_data(&thread, checkpoint_id).await? {
                return Ok(Some((thread, data.parent_id.clone())));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AgentState;

    #[tokio::test]
    async fn test_thread_safe_checkpointer() {
        let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();

        let state = AgentState::new("test".to_string());
        let id = checkpointer.save(&state, 0).await.unwrap();

        let loaded = checkpointer.load(&id).await.unwrap();
        assert_eq!(loaded.input, "test");

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 1);

        checkpointer.delete(&id).await.unwrap();
        let list = checkpointer.list().await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_memory_threads_are_isolated() {
        let cp = ThreadSafeMemoryCheckpointer::<AgentState>::new();
        let a1 = cp
            .save_threaded("thread-a", &AgentState::new("a1"), 1)
            .await
            .unwrap();
        let b1 = cp
            .save_threaded("thread-b", &AgentState::new("b1"), 1)
            .await
            .unwrap();
        let a2 = cp
            .save_threaded("thread-a", &AgentState::new("a2"), 2)
            .await
            .unwrap();

        // Each thread only sees its own checkpoints.
        assert_eq!(
            cp.list_threaded("thread-a").await.unwrap(),
            vec![a1.clone(), a2.clone()]
        );
        assert_eq!(
            cp.list_threaded("thread-b").await.unwrap(),
            vec![b1.clone()]
        );

        // Cross-thread reads are invisible (b does not load a's a2, and vice versa).
        assert!(cp.load_threaded("thread-b", &a2).await.is_err());
        assert!(cp.load_threaded("thread-a", &b1).await.is_err());

        // last() is per-thread.
        let (last_a, count_a) = cp.last_threaded("thread-a").await.unwrap().unwrap();
        assert_eq!((last_a.input.as_str(), count_a), ("a2", 2));
        let (last_b, count_b) = cp.last_threaded("thread-b").await.unwrap().unwrap();
        assert_eq!((last_b.input.as_str(), count_b), ("b1", 1));

        // Deleting from one thread does not touch the other.
        cp.delete_threaded("thread-a", &a1).await.unwrap();
        assert_eq!(cp.list_threaded("thread-a").await.unwrap(), vec![a2]);
        assert_eq!(cp.list_threaded("thread-b").await.unwrap(), vec![b1]);
    }

    #[tokio::test]
    async fn test_memory_save_fork_records_parent() {
        let cp = ThreadSafeMemoryCheckpointer::<AgentState>::new();
        let root = cp.save(&AgentState::new("root"), 0).await.unwrap();
        let fork = cp
            .save_fork_threaded(Some(&root), DEFAULT_THREAD, &AgentState::new("fork"), 3)
            .await
            .unwrap();
        let (thread, parent) = cp.checkpoint_lineage(&fork).await.unwrap().unwrap();
        assert_eq!(thread, DEFAULT_THREAD);
        assert_eq!(parent.as_deref(), Some(root.as_str()));
    }

    #[tokio::test]
    async fn test_file_checkpointer() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();

        let state = AgentState::new("file_test".to_string());
        let id = checkpointer.save(&state, 0).await.unwrap();

        let loaded = checkpointer.load(&id).await.unwrap();
        assert_eq!(loaded.input, "file_test");

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 1);

        checkpointer.delete(&id).await.unwrap();
        let list = checkpointer.list().await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_file_checkpointer_atomic_write() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();

        let id = checkpointer
            .save(&AgentState::new("atomic".to_string()), 0)
            .await
            .unwrap();

        // The checkpoint lives in the default-thread subdirectory under the
        // `<seq>-<id>.json` layout, complete and parseable, with no `.tmp` leftover.
        let default_dir = temp_dir.path().join(DEFAULT_THREAD);
        let mut main: Option<PathBuf> = None;
        let mut entries = tokio::fs::read_dir(&default_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            let path = entry.path();
            assert!(
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| !s.ends_with(".json")),
                "no tmp file must linger ({})",
                path.display()
            );
            assert!(
                path.extension().and_then(|e| e.to_str()) == Some("json"),
                "only the real json lives in the thread dir"
            );
            let json = tokio::fs::read_to_string(&path).await.unwrap();
            let data: CheckpointData<AgentState> = serde_json::from_str(&json).unwrap();
            assert_eq!(data.id, id);
            main = Some(path);
        }
        assert!(main.is_some(), "checkpoint must be written under default/");

        // A stale root `.tmp` file must not be read by list() (extension filter).
        std::fs::write(temp_dir.path().join("stale.json.tmp"), b"{}").unwrap();
        let list = checkpointer.list().await.unwrap();
        assert_eq!(list, vec![id]);
    }

    #[tokio::test]
    async fn test_file_checkpointer_multiple() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();

        let id1 = checkpointer
            .save(&AgentState::new("state1".to_string()), 0)
            .await
            .unwrap();
        let id2 = checkpointer
            .save(&AgentState::new("state2".to_string()), 0)
            .await
            .unwrap();
        let _id3 = checkpointer
            .save(&AgentState::new("state3".to_string()), 0)
            .await
            .unwrap();

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 3);

        let loaded = checkpointer.load(&id2).await.unwrap();
        assert_eq!(loaded.input, "state2");

        checkpointer.delete(&id1).await.unwrap();
        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_file_checkpointer_path_traversal() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();

        // Path traversal should be rejected
        let result = checkpointer.load("..").await;
        assert!(result.is_err());

        let result = checkpointer.load("../etc/passwd").await;
        assert!(result.is_err());

        // Thread names are validated too.
        assert!(checkpointer
            .save_threaded("../x", &AgentState::new("n"), 0)
            .await
            .is_err());
    }

    #[test]
    fn test_file_legacy_flat_migrates_to_default() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Simulate a pre-0.25.x flat-layout checkpoint at `<dir>/<id>.json`.
        let id = Uuid::new_v4().to_string();
        let mut data = CheckpointData::new(AgentState::new("legacy".to_string()));
        data.id = id.clone();
        std::fs::write(
            temp_dir.path().join(format!("{id}.json")),
            serde_json::to_string(&data).unwrap(),
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();

            // The legacy flat file is folded into the default thread (read-only).
            let list = checkpointer.list().await.unwrap();
            assert_eq!(list, vec![id.clone()]);
            let loaded = checkpointer.load(&id).await.unwrap();
            assert_eq!(loaded.input, "legacy");

            // Read-only: no new `default/` dir is created and the original file is untouched.
            assert!(!temp_dir.path().join(DEFAULT_THREAD).exists());
            assert!(temp_dir.path().join(format!("{id}.json")).exists());
        });
    }

    #[tokio::test]
    async fn test_list_orders_oldest_to_newest() {
        let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
        checkpointer
            .save(&AgentState::new("first".to_string()), 0)
            .await
            .unwrap();
        checkpointer
            .save(&AgentState::new("second".to_string()), 1)
            .await
            .unwrap();
        checkpointer
            .save(&AgentState::new("third".to_string()), 2)
            .await
            .unwrap();

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 3);
        // H5: the last id must be the most recent save (not HashMap arbitrary order).
        let (state, _) = checkpointer.last().await.unwrap().unwrap();
        assert_eq!(state.input, "third");
    }

    #[tokio::test]
    async fn test_last_returns_recursion_count() {
        let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
        checkpointer
            .save(&AgentState::new("a".to_string()), 7)
            .await
            .unwrap();
        checkpointer
            .save(&AgentState::new("b".to_string()), 12)
            .await
            .unwrap();

        // M6: last() returns the recursion_count of the most recent save
        let (state, recursion_count) = checkpointer.last().await.unwrap().unwrap();
        assert_eq!(state.input, "b");
        assert_eq!(recursion_count, 12);
    }

    #[tokio::test]
    async fn test_update_state_occ_memory() {
        let checkpointer = ThreadSafeMemoryCheckpointer::<AgentState>::new();
        let id = checkpointer
            .save(&AgentState::new("v1".to_string()), 0)
            .await
            .unwrap();

        // First edit based on version 1 succeeds and bumps to 2.
        let version = checkpointer
            .update_state(&id, &AgentState::new("v2".to_string()), 1)
            .await
            .unwrap();
        assert_eq!(version, 2);
        assert_eq!(checkpointer.load(&id).await.unwrap().input, "v2");

        // A stale edit still based on version 1 must be rejected, not overwrite.
        let conflict = checkpointer
            .update_state(&id, &AgentState::new("v3-stale".to_string()), 1)
            .await
            .unwrap_err();
        match conflict {
            GraphError::CheckpointVersionConflict {
                expected, actual, ..
            } => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            other => panic!("expected CheckpointVersionConflict, got {other:?}"),
        }
        assert_eq!(checkpointer.load(&id).await.unwrap().input, "v2");

        // An edit based on the current version 2 succeeds.
        checkpointer
            .update_state(&id, &AgentState::new("v3".to_string()), 2)
            .await
            .unwrap();
        assert_eq!(checkpointer.load(&id).await.unwrap().input, "v3");
    }

    #[tokio::test]
    async fn test_update_state_missing_and_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();
        let id = checkpointer
            .save(&AgentState::new("disk-v1".to_string()), 0)
            .await
            .unwrap();
        checkpointer
            .update_state(&id, &AgentState::new("disk-v2".to_string()), 1)
            .await
            .unwrap();
        assert_eq!(checkpointer.load(&id).await.unwrap().input, "disk-v2");
        // Stale version conflicts on disk too.
        assert!(checkpointer
            .update_state(&id, &AgentState::new("stale".to_string()), 1)
            .await
            .is_err());
        // Unknown checkpoint id errors rather than inserting.
        assert!(checkpointer
            .update_state("missing", &AgentState::new("x".to_string()), 1)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_file_checkpointer_last_orders_by_save() {
        let temp_dir = tempfile::tempdir().unwrap();
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path()).unwrap();
        checkpointer
            .save(&AgentState::new("one".to_string()), 1)
            .await
            .unwrap();
        checkpointer
            .save(&AgentState::new("two".to_string()), 2)
            .await
            .unwrap();

        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 2);
        let (state, recursion_count) = checkpointer.last().await.unwrap().unwrap();
        assert_eq!(state.input, "two");
        assert_eq!(recursion_count, 2);
    }
}
