// src/langgraph/checkpointer.rs
//! Checkpointing for state persistence

use async_trait::async_trait;
use super::state::StateSchema;
use super::errors::{GraphError, GraphResult};
use std::collections::HashMap;
use tokio::sync::Mutex;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

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
        Self { checkpoints: Mutex::new(HashMap::new()) }
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
        self.checkpoints.lock().await.get(checkpoint_id)
            .map(|d| d.state.clone())
            .ok_or_else(|| GraphError::CheckpointError(
                format!("Checkpoint '{}' not found", checkpoint_id)
            ))
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
    checkpoints: std::sync::Mutex<HashMap<String, CheckpointData<S>>>,
}

impl<S: StateSchema> ThreadSafeMemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self {
            checkpoints: std::sync::Mutex::new(HashMap::new()),
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
        self.checkpoints.lock().unwrap().insert(id.clone(), data);
        Ok(id)
    }
    
    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let checkpoints = self.checkpoints.lock().unwrap();
        checkpoints.get(checkpoint_id)
            .map(|d| d.state.clone())
            .ok_or_else(|| GraphError::CheckpointError(
                format!("Checkpoint '{}' not found", checkpoint_id)
            ))
    }
    
    async fn list(&self) -> GraphResult<Vec<String>> {
        Ok(self.checkpoints.lock().unwrap().keys().cloned().collect())
    }
    
    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        self.checkpoints.lock().unwrap().remove(checkpoint_id);
        Ok(())
    }
}

/// File-based checkpointer for persistent storage
pub struct FileCheckpointer<S: StateSchema> {
    directory: std::path::PathBuf,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> FileCheckpointer<S> {
    pub fn new(directory: impl Into<std::path::PathBuf>) -> Self {
        let dir = directory.into();
        if !dir.exists() {
            std::fs::create_dir_all(&dir).ok();
        }
        Self { 
            directory: dir,
            _phantom: std::marker::PhantomData,
        }
    }
    
    fn checkpoint_path(&self, id: &str) -> std::path::PathBuf {
        self.directory.join(format!("{}.json", id))
    }
}

impl<S: StateSchema> Default for FileCheckpointer<S> {
    fn default() -> Self {
        Self::new(".checkpoints")
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for FileCheckpointer<S> {
    async fn save(&self, state: &S) -> GraphResult<String> {
        let data = CheckpointData::new(state.clone());
        let id = data.id.clone();
        let path = self.checkpoint_path(&id);
        
        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| GraphError::CheckpointError(format!("Serialize error: {}", e)))?;
        
        std::fs::write(&path, json)
            .map_err(|e| GraphError::CheckpointError(format!("Write error: {}", e)))?;
        
        Ok(id)
    }
    
    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let path = self.checkpoint_path(checkpoint_id);
        
        if !path.exists() {
            return Err(GraphError::CheckpointError(
                format!("Checkpoint '{}' not found", checkpoint_id)
            ));
        }
        
        let json = std::fs::read_to_string(&path)
            .map_err(|e| GraphError::CheckpointError(format!("Read error: {}", e)))?;
        
        let data: CheckpointData<S> = serde_json::from_str(&json)
            .map_err(|e| GraphError::CheckpointError(format!("Deserialize error: {}", e)))?;
        
        Ok(data.state)
    }
    
    async fn list(&self) -> GraphResult<Vec<String>> {
        let mut ids = Vec::new();
        
        let entries = std::fs::read_dir(&self.directory)
            .map_err(|e| GraphError::CheckpointError(format!("Read dir error: {}", e)))?;
        
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "json") {
                    if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                        ids.push(id.to_string());
                    }
                }
            }
        }
        
        Ok(ids)
    }
    
    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        let path = self.checkpoint_path(checkpoint_id);
        
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| GraphError::CheckpointError(format!("Delete error: {}", e)))?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::AgentState;
    
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
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path());
        
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
        let checkpointer = FileCheckpointer::<AgentState>::new(temp_dir.path());
        
        let id1 = checkpointer.save(&AgentState::new("state1".to_string())).await.unwrap();
        let id2 = checkpointer.save(&AgentState::new("state2".to_string())).await.unwrap();
        let id3 = checkpointer.save(&AgentState::new("state3".to_string())).await.unwrap();
        
        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 3);
        
        let loaded = checkpointer.load(&id2).await.unwrap();
        assert_eq!(loaded.input, "state2");
        
        checkpointer.delete(&id1).await.unwrap();
        let list = checkpointer.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }
}