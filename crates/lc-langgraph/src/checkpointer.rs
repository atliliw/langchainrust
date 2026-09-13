// crates/lc-langgraph/src/checkpointer.rs
//! Checkpointing for state persistence

use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Checkpointer trait for state persistence
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
}

/// Fresh checkpoints start at version 1 (0 only occurs in pre-0.22.4 files).
fn initial_version() -> u64 {
    1
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

/// In-memory checkpointer for development
pub struct MemoryCheckpointer<S: StateSchema> {
    checkpoints: Mutex<HashMap<String, CheckpointData<S>>>,
    next_seq: AtomicU64,
}

impl<S: StateSchema> MemoryCheckpointer<S> {
    /// Create a new empty in-memory checkpointer.
    pub fn new() -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
            next_seq: AtomicU64::new(0),
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
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
        let id = data.id.clone();
        self.checkpoints.lock().await.insert(id.clone(), data);
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        self.checkpoints
            .lock()
            .await
            .get(checkpoint_id)
            .map(|d| d.state.clone())
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{}' not found", checkpoint_id))
            })
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        let guard = self.checkpoints.lock().await;
        let mut items: Vec<(i64, u64, String)> = guard
            .values()
            .map(|d| (d.timestamp, d.seq, d.id.clone()))
            .collect();
        // H5: the old HashMap.keys() order was nondeterministic; sort by (timestamp, seq)
        // ascending instead, so callers taking `.last()` get the most recent checkpoint.
        items.sort();
        Ok(items.into_iter().map(|(_, _, id)| id).collect())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let guard = self.checkpoints.lock().await;
        Ok(guard
            .values()
            .max_by_key(|d| (d.timestamp, d.seq))
            .map(|d| (d.state.clone(), d.recursion_count)))
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        update_locked(&self.checkpoints, checkpoint_id, state, expected_version).await
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
}

/// In-memory OCC edit shared by [`MemoryCheckpointer`] and
/// [`ThreadSafeMemoryCheckpointer`]: bump the version only while holding the
/// map lock, so two concurrent edits cannot both succeed against the same
/// base version.
async fn update_locked<S: StateSchema>(
    checkpoints: &Mutex<HashMap<String, CheckpointData<S>>>,
    checkpoint_id: &str,
    state: &S,
    expected_version: u64,
) -> GraphResult<u64> {
    let mut guard = checkpoints.lock().await;
    let data = guard.get_mut(checkpoint_id).ok_or_else(|| {
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
    Ok(data.version)
}

/// Thread-safe memory checkpointer
pub struct ThreadSafeMemoryCheckpointer<S: StateSchema> {
    checkpoints: Mutex<HashMap<String, CheckpointData<S>>>,
    next_seq: AtomicU64,
}

impl<S: StateSchema> ThreadSafeMemoryCheckpointer<S> {
    /// Create a new empty thread-safe memory checkpointer.
    pub fn new() -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
            next_seq: AtomicU64::new(0),
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
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
        let id = data.id.clone();
        self.checkpoints.lock().await.insert(id.clone(), data);
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let checkpoints = self.checkpoints.lock().await;
        checkpoints
            .get(checkpoint_id)
            .map(|d| d.state.clone())
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{}' not found", checkpoint_id))
            })
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        let guard = self.checkpoints.lock().await;
        let mut items: Vec<(i64, u64, String)> = guard
            .values()
            .map(|d| (d.timestamp, d.seq, d.id.clone()))
            .collect();
        // H5: sort by (timestamp, seq) ascending; `.last()` is the most recent checkpoint.
        items.sort();
        Ok(items.into_iter().map(|(_, _, id)| id).collect())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let guard = self.checkpoints.lock().await;
        Ok(guard
            .values()
            .max_by_key(|d| (d.timestamp, d.seq))
            .map(|d| (d.state.clone(), d.recursion_count)))
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        update_locked(&self.checkpoints, checkpoint_id, state, expected_version).await
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
}

/// File-based checkpointer for persistent storage
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
        Ok(Self {
            directory: dir,
            next_seq: AtomicU64::new(0),
            update_lock: Mutex::new(()),
            _phantom: std::marker::PhantomData,
        })
    }

    fn checkpoint_path(&self, id: &str) -> GraphResult<std::path::PathBuf> {
        // Sanitize id to prevent path traversal: reject ".." and absolute paths
        if id.contains("..") || id.contains('/') || id.contains('\\') {
            return Err(GraphError::CheckpointError(format!(
                "Invalid checkpoint id '{}': path traversal detected",
                id
            )));
        }
        if std::path::Path::new(id).is_absolute() {
            return Err(GraphError::CheckpointError(format!(
                "Invalid checkpoint id '{}': absolute path not allowed",
                id
            )));
        }
        Ok(self.directory.join(format!("{}.json", id)))
    }

    /// Read every checkpoint file's `(timestamp, seq, id)` sort keys.
    async fn sorted_ids(&self) -> GraphResult<Vec<(i64, u64, String)>> {
        let mut items: Vec<(i64, u64, String)> = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.directory)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir error: {}", e)))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir entry error: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let Some(id) = path.file_stem().and_then(|s| s.to_str()).map(String::from) else {
                    continue;
                };
                let json = tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
                let data: CheckpointData<S> = serde_json::from_str(&json).map_err(|e| {
                    GraphError::CheckpointError(format!("Deserialize error: {}", e))
                })?;
                items.push((data.timestamp, data.seq, id));
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
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let data = CheckpointData::with_progress(state.clone(), seq, recursion_count);
        let id = data.id.clone();
        let path = self.checkpoint_path(&id)?;

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;

        // Atomic write: write `{id}.json.tmp` first, then rename over the real file, so a
        // crash or interrupt mid-JSON cannot corrupt the checkpoint (same pattern as
        // FileResumeStore). The `.tmp` extension is never picked up by sorted_ids' `.json` filter.
        let tmp_path = self.directory.join(format!("{id}.json.tmp"));
        tokio::fs::write(&tmp_path, &json)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Write error: {}", e)))?;
        tokio::fs::rename(&tmp_path, &path)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Atomic rename error: {}", e)))?;

        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let path = self.checkpoint_path(checkpoint_id)?;

        if !path.exists() {
            return Err(GraphError::CheckpointError(format!(
                "Checkpoint '{}' not found",
                checkpoint_id
            )));
        }

        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;

        let data: CheckpointData<S> = serde_json::from_str(&json)
            .map_err(|e| GraphError::CheckpointError(format!("Deserialize error: {}", e)))?;

        Ok(data.state)
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        Ok(self
            .sorted_ids()
            .await?
            .into_iter()
            .map(|(_, _, id)| id)
            .collect())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let Some((_, _, last_id)) = self.sorted_ids().await?.into_iter().last() else {
            return Ok(None);
        };
        let json = tokio::fs::read_to_string(&self.checkpoint_path(&last_id)?)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
        let data: CheckpointData<S> = serde_json::from_str(&json)
            .map_err(|e| GraphError::CheckpointError(format!("Deserialize error: {}", e)))?;
        Ok(Some((data.state, data.recursion_count)))
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        let _guard = self.update_lock.lock().await;
        let path = self.checkpoint_path(checkpoint_id)?;
        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
        let mut data: CheckpointData<S> = serde_json::from_str(&json)
            .map_err(|e| GraphError::CheckpointError(format!("Deserialize error: {}", e)))?;
        if data.version != expected_version {
            return Err(GraphError::CheckpointVersionConflict {
                checkpoint_id: checkpoint_id.to_string(),
                expected: expected_version,
                actual: data.version,
            });
        }
        data.state = state.clone();
        data.version += 1;

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;
        let tmp_path = self.directory.join(format!("{checkpoint_id}.json.tmp"));
        tokio::fs::write(&tmp_path, &json)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Write error: {}", e)))?;
        tokio::fs::rename(&tmp_path, &path)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Atomic rename error: {}", e)))?;
        Ok(data.version)
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        let path = self.checkpoint_path(checkpoint_id)?;

        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| GraphError::CheckpointError(format!("Delete error: {}", e)))?;
        }

        Ok(())
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

        // The main file is complete and parseable; no `.tmp` leftover (rename cleaned it up).
        let main = temp_dir.path().join(format!("{id}.json"));
        assert!(main.exists(), "checkpoint file must exist");
        let json = tokio::fs::read_to_string(&main).await.unwrap();
        assert!(
            serde_json::from_str::<CheckpointData<AgentState>>(&json).is_ok(),
            "checkpoint file must be complete JSON after atomic write"
        );
        assert!(
            !temp_dir.path().join(format!("{id}.json.tmp")).exists(),
            "tmp file must be renamed away, not left behind"
        );

        // A stale `.tmp` file must not be read by list() (extension filter).
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
