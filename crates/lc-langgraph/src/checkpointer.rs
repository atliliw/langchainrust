// crates/lc-langgraph/src/checkpointer.rs
//! Checkpointing for state persistence

use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Checkpointer trait for state persistence
#[async_trait]
pub trait Checkpointer<S: StateSchema>: Send + Sync {
    async fn save(&self, state: &S) -> GraphResult<String>;
    async fn load(&self, checkpoint_id: &str) -> GraphResult<S>;
    async fn list(&self) -> GraphResult<Vec<String>>;
    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()>;
}

/// Checkpoint data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "S: StateSchema")]
pub struct CheckpointData<S: StateSchema> {
    pub id: String,
    pub state: S,
    pub timestamp: i64,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl<S: StateSchema> CheckpointData<S> {
    pub fn new(state: S) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            state,
            timestamp: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
        }
    }
}

/// In-memory checkpointer for development
pub struct MemoryCheckpointer<S: StateSchema> {
    checkpoints: Mutex<HashMap<String, CheckpointData<S>>>,
}

impl<S: StateSchema> MemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
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
    async fn save(&self, state: &S) -> GraphResult<String> {
        let data = CheckpointData::new(state.clone());
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
        Ok(self.checkpoints.lock().await.keys().cloned().collect())
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
}

/// Thread-safe memory checkpointer
pub struct ThreadSafeMemoryCheckpointer<S: StateSchema> {
    checkpoints: Mutex<HashMap<String, CheckpointData<S>>>,
}

impl<S: StateSchema> ThreadSafeMemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
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
    async fn save(&self, state: &S) -> GraphResult<String> {
        let data = CheckpointData::new(state.clone());
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
        Ok(self.checkpoints.lock().await.keys().cloned().collect())
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().await.remove(checkpoint_id);
        Ok(())
    }
}

/// File-based checkpointer for persistent storage
pub struct FileCheckpointer<S: StateSchema> {
    directory: std::path::PathBuf,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> FileCheckpointer<S> {
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
}

// NOTE: `Default` is intentionally NOT implemented for `FileCheckpointer` (Q1).
// The default constructor would have to create the `.checkpoints` directory, which
// is I/O that can fail (read-only cwd, disk full, permissions) — `Default` cannot
// report that failure, so it would have to panic. Use `FileCheckpointer::new(...)`
// which returns a `GraphResult` and surfaces the error instead.

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for FileCheckpointer<S> {
    async fn save(&self, state: &S) -> GraphResult<String> {
        let data = CheckpointData::new(state.clone());
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
        let mut ids = Vec::new();

        let entries = tokio::fs::read_dir(&self.directory)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir error: {}", e)))?;

        let mut entries = entries;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| GraphError::CheckpointError(format!("Read dir entry error: {}", e)))?
        {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(id.to_string());
                }
            }
        }

        Ok(ids)
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
        let id = checkpointer.save(&state).await.unwrap();

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
        let id = checkpointer.save(&state).await.unwrap();

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
            .save(&AgentState::new("state1".to_string()))
            .await
            .unwrap();
        let id2 = checkpointer
            .save(&AgentState::new("state2".to_string()))
            .await
            .unwrap();
        let _id3 = checkpointer
            .save(&AgentState::new("state3".to_string()))
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
}
