// src/langgraph/checkpointer.rs
//! Checkpointing for state persistence

use async_trait::async_trait;
use super::state::StateSchema;
use super::errors::{GraphError, GraphResult};
use std::collections::HashMap;
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
    checkpoints: HashMap<String, CheckpointData<S>>,
}

impl<S: StateSchema> MemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self { checkpoints: HashMap::new() }
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
        Ok(data.id)
    }
    
    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        unimplemented!("MemoryCheckpointer::load requires mutable state")
    }
    
    async fn list(&self) -> GraphResult<Vec<String>> {
        Ok(self.checkpoints.keys().cloned().collect())
    }
    
    async fn delete(&self, _checkpoint_id: &str) -> GraphResult<()> {
        unimplemented!("MemoryCheckpointer::delete requires mutable state")
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
}