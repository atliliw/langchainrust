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
        // H5: 旧的 HashMap.keys() 顺序不定;改为按 (timestamp, seq) 升序,
        // 调用方取 .last() 即得到最近的 checkpoint。
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

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
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
        // H5: 按 (timestamp, seq) 升序,取 .last() 为最近 checkpoint。
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

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
}

/// File-based checkpointer for persistent storage
pub struct FileCheckpointer<S: StateSchema> {
    directory: std::path::PathBuf,
    next_seq: AtomicU64,
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
        // H5: 按 (timestamp, seq) 升序;seq 打破同一秒内的并列。
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

        tokio::fs::write(&path, json)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Write error: {}", e)))?;

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
        // H5: 最后一个 id 必须是最后一次 save 的(而非 HashMap 乱序)。
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

        // M6: last() 返回最近一次 save 的 recursion_count
        let (state, recursion_count) = checkpointer.last().await.unwrap().unwrap();
        assert_eq!(state.input, "b");
        assert_eq!(recursion_count, 12);
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
