// src/memory/persistent.rs
//! Memory Persistence Trait
//!
//! Defines the interface for persistent memory storage.
//! Allows memory to be saved/loaded from external storage (MongoDB, Redis, etc.)

use async_trait::async_trait;
use super::base::{BaseMemory, MemoryError};

/// Persistent Memory Trait
///
/// Extends BaseMemory with persistence capabilities.
/// Implementations can save/load memory state to external storage.
///
/// # Design
/// - Framework provides trait and algorithms
/// - Business layer provides storage implementation
///
/// # Example
/// ```ignore
/// use langchainrust::{MongoPersistentMemory, PersistentMemory};
///
/// let mut memory = MongoPersistentMemory::new(config);
/// memory.load_from_store("session_123").await?;
/// memory.save_context(&inputs, &outputs).await?;
/// memory.save_to_store("session_123").await?;
/// ```
#[async_trait]
pub trait PersistentMemory: BaseMemory {
    /// Load memory state from persistent storage
    ///
    /// # Arguments
    /// * `session_id` - Unique identifier for the conversation session
    ///
    /// # Returns
    /// Ok(()) if successful, MemoryError if failed
    async fn load_from_store(&mut self, session_id: &str) -> Result<(), MemoryError>;
    
    /// Save current memory state to persistent storage
    ///
    /// Called after each conversation turn to persist the updated state.
    ///
    /// # Arguments
    /// * `session_id` - Unique identifier for the conversation session
    async fn save_to_store(&mut self, session_id: &str) -> Result<(), MemoryError>;
    
    /// Delete a session's memory from storage
    ///
    /// # Arguments
    /// * `session_id` - Session to delete
    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError>;
    
    /// Check if a session exists in storage
    ///
    /// # Arguments
    /// * `session_id` - Session to check
    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError>;
    
    /// Get current session ID
    fn current_session_id(&self) -> Option<&str>;
    
    /// Set session ID
    fn set_session_id(&mut self, session_id: String);
}

/// Memory persistence configuration
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Auto-save after each save_context call
    pub auto_save: bool,
    
    /// Auto-load on first access
    pub auto_load: bool,
    
    /// Maximum messages to keep in memory before compression
    pub max_messages: usize,
    
    /// Token limit for summary buffer memory
    pub token_limit: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            auto_save: true,
            auto_load: true,
            max_messages: 100,
            token_limit: 4000,
        }
    }
}

impl PersistenceConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_auto_save(mut self, auto_save: bool) -> Self {
        self.auto_save = auto_save;
        self
    }
    
    pub fn with_auto_load(mut self, auto_load: bool) -> Self {
        self.auto_load = auto_load;
        self
    }
    
    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = max_messages;
        self
    }
    
    pub fn with_token_limit(mut self, token_limit: usize) -> Self {
        self.token_limit = token_limit;
        self
    }
}

/// Memory data structure for serialization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryData {
    /// Session ID
    pub session_id: String,
    
    /// Chat messages (serialized Message objects)
    pub messages: Vec<crate::schema::Message>,
    
    /// Current summary (for summary-based memory)
    pub summary: Option<String>,
    
    /// Memory metadata
    pub metadata: std::collections::HashMap<String, String>,
    
    /// Created timestamp
    pub created_at: String,
    
    /// Last updated timestamp
    pub updated_at: String,
}

impl MemoryData {
    pub fn new(session_id: String) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            session_id,
            messages: Vec::new(),
            summary: None,
            metadata: std::collections::HashMap::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }
    
    pub fn with_messages(mut self, messages: Vec<crate::schema::Message>) -> Self {
        self.messages = messages;
        self.updated_at = chrono::Utc::now().to_rfc3339();
        self
    }
    
    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = Some(summary);
        self.updated_at = chrono::Utc::now().to_rfc3339();
        self
    }
    
    pub fn add_message(&mut self, message: crate::schema::Message) {
        self.messages.push(message);
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
    
    pub fn set_summary(&mut self, summary: String) {
        self.summary = Some(summary);
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_persistence_config_default() {
        let config = PersistenceConfig::default();
        assert!(config.auto_save);
        assert!(config.auto_load);
        assert_eq!(config.max_messages, 100);
        assert_eq!(config.token_limit, 4000);
    }
    
    #[test]
    fn test_persistence_config_custom() {
        let config = PersistenceConfig::new()
            .with_auto_save(false)
            .with_max_messages(50)
            .with_token_limit(2000);
        
        assert!(!config.auto_save);
        assert_eq!(config.max_messages, 50);
        assert_eq!(config.token_limit, 2000);
    }
    
    #[test]
    fn test_memory_data_new() {
        let data = MemoryData::new("session_123".to_string());
        assert_eq!(data.session_id, "session_123");
        assert!(data.messages.is_empty());
        assert!(data.summary.is_none());
    }
    
    #[test]
    fn test_memory_data_with_messages() {
        let messages = vec![
            crate::schema::Message::human("Hello"),
            crate::schema::Message::ai("Hi!"),
        ];
        
        let data = MemoryData::new("session_123".to_string())
            .with_messages(messages);
        
        assert_eq!(data.messages.len(), 2);
    }
}