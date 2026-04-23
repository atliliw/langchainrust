// src/memory/mongo_memory.rs
//! MongoDB Persistent Memory Implementation

use async_trait::async_trait;
use mongodb::{
    Client, Database,
    bson::{doc, oid::ObjectId},
    options::ClientOptions,
    Collection,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;

use super::base::{BaseMemory, MemoryError};
use super::persistent::{PersistentMemory, PersistenceConfig, MemoryData};
use super::summary_buffer::ConversationSummaryBufferMemory;
use crate::language_models::OpenAIChat;
use crate::schema::Message;

/// MongoDB-stored memory document
#[derive(Debug, Serialize, Deserialize)]
struct MongoMemoryDoc {
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    id: Option<ObjectId>,
    session_id: String,
    messages: Vec<Message>,
    summary: Option<String>,
    metadata: HashMap<String, String>,
    created_at: String,
    updated_at: String,
}

impl From<MemoryData> for MongoMemoryDoc {
    fn from(data: MemoryData) -> Self {
        Self {
            id: None,
            session_id: data.session_id,
            messages: data.messages,
            summary: data.summary,
            metadata: data.metadata,
            created_at: data.created_at,
            updated_at: data.updated_at,
        }
    }
}

impl From<MongoMemoryDoc> for MemoryData {
    fn from(doc: MongoMemoryDoc) -> Self {
        Self {
            session_id: doc.session_id,
            messages: doc.messages,
            summary: doc.summary,
            metadata: doc.metadata,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
        }
    }
}

/// MongoDB Persistent Memory
pub struct MongoPersistentMemory {
    inner: RwLock<ConversationSummaryBufferMemory>,
    database: Database,
    collection_name: String,
    session_id: RwLock<Option<String>>,
    config: RwLock<PersistenceConfig>,
}

impl MongoPersistentMemory {
    pub async fn new(
        mongo_uri: &str,
        database_name: &str,
        collection_name: &str,
        llm: OpenAIChat,
        token_limit: usize,
    ) -> Result<Self, MemoryError> {
        let client_options = ClientOptions::parse(mongo_uri)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB connection failed: {}", e)))?;
        
        let client = Client::with_options(client_options)
            .map_err(|e| MemoryError::LoadError(format!("MongoDB client error: {}", e)))?;
        
        let database = client.database(database_name);
        
        let inner = ConversationSummaryBufferMemory::new(llm, token_limit);
        
        Ok(Self {
            inner: RwLock::new(inner),
            database,
            collection_name: collection_name.to_string(),
            session_id: RwLock::new(None),
            config: RwLock::new(PersistenceConfig::default()),
        })
    }
    
    pub async fn with_config(self, config: PersistenceConfig) -> Self {
        *self.config.write().await = config;
        self
    }
    
    fn collection(&self) -> Collection<MongoMemoryDoc> {
        self.database.collection(&self.collection_name)
    }
    
    pub async fn create_indexes(&self) -> Result<(), MemoryError> {
        let collection = self.collection();
        
        collection
            .create_index(
                mongodb::IndexModel::builder()
                    .keys(doc! { "session_id": 1 })
                    .build(),
                None,
            )
            .await
            .map_err(|e| MemoryError::SaveError(format!("Index creation failed: {}", e)))?;
        
        Ok(())
    }
    
    pub async fn get_session_id(&self) -> Option<String> {
        self.session_id.read().await.clone()
    }
    
    pub async fn set_session_id_async(&self, session_id: String) {
        *self.session_id.write().await = Some(session_id);
    }
    
    async fn do_load_from_store(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();
        
        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;
        
        if let Some(doc) = result {
            let data: MemoryData = doc.into();
            
            let mut inner = self.inner.write().await;
            let chat_memory = inner.chat_memory_mut();
            chat_memory.clear();
            
            for msg in &data.messages {
                if matches!(msg.message_type, crate::schema::MessageType::Human) {
                    chat_memory.add_user_message(&msg.content);
                } else if matches!(msg.message_type, crate::schema::MessageType::AI) {
                    chat_memory.add_ai_message(&msg.content);
                } else if matches!(msg.message_type, crate::schema::MessageType::System) {
                    chat_memory.add_system_message(&msg.content);
                }
            }
        }
        
        *self.session_id.write().await = Some(session_id.to_string());
        
        Ok(())
    }
    
    async fn do_save_to_store(&self, session_id: &str) -> Result<(), MemoryError> {
        let inner = self.inner.read().await;
        let messages: Vec<Message> = inner.chat_memory().messages().to_vec();
        let summary = inner.buffer().await;
        
        let now = chrono::Utc::now().to_rfc3339();
        
        let data = MemoryData {
            session_id: session_id.to_string(),
            messages,
            summary: Some(summary),
            metadata: HashMap::new(),
            created_at: now.clone(),
            updated_at: now,
        };
        
        let doc: MongoMemoryDoc = data.into();
        let collection = self.collection();
        let filter = doc! { "session_id": session_id };
        
        collection
            .replace_one(filter, doc, mongodb::options::ReplaceOptions::builder().upsert(true).build())
            .await
            .map_err(|e| MemoryError::SaveError(format!("MongoDB save failed: {}", e)))?;
        
        Ok(())
    }
}

#[async_trait]
impl BaseMemory for MongoPersistentMemory {
    fn memory_variables(&self) -> Vec<&str> {
        vec!["history"]
    }
    
    async fn load_memory_variables(
        &self,
        inputs: &HashMap<String, String>,
    ) -> Result<HashMap<String, serde_json::Value>, MemoryError> {
        let config = self.config.read().await;
        if config.auto_load {
            let session_id = self.session_id.read().await.clone();
            if let Some(sid) = session_id {
                let inner = self.inner.read().await;
                if inner.chat_memory().is_empty() {
                    drop(inner);
                    drop(config);
                    self.do_load_from_store(&sid).await?;
                }
            }
        }
        
        let inner = self.inner.read().await;
        inner.load_memory_variables(inputs).await
    }
    
    async fn save_context(
        &mut self,
        inputs: &HashMap<String, String>,
        outputs: &HashMap<String, String>,
    ) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.save_context(inputs, outputs).await?;
        }
        
        let config = self.config.read().await;
        if config.auto_save {
            let session_id = self.session_id.read().await.clone();
            if let Some(sid) = session_id {
                drop(config);
                self.do_save_to_store(&sid).await?;
            }
        }
        
        Ok(())
    }
    
    async fn clear(&mut self) -> Result<(), MemoryError> {
        {
            let mut inner = self.inner.write().await;
            inner.clear().await?;
        }
        
        let session_id = self.session_id.read().await.clone();
        if let Some(sid) = session_id {
            self.do_delete_session(&sid).await?;
        }
        
        Ok(())
    }
}

#[async_trait]
impl PersistentMemory for MongoPersistentMemory {
    async fn load_from_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_load_from_store(session_id).await
    }
    
    async fn save_to_store(&mut self, session_id: &str) -> Result<(), MemoryError> {
        self.do_save_to_store(session_id).await
    }
    
    async fn delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        self.do_delete_session(session_id).await
    }
    
    async fn session_exists(&self, session_id: &str) -> Result<bool, MemoryError> {
        let collection = self.collection();
        let filter = doc! { "session_id": session_id };
        let result = collection
            .find_one(filter, None)
            .await
            .map_err(|e| MemoryError::LoadError(format!("MongoDB find failed: {}", e)))?;
        
        Ok(result.is_some())
    }
    
    fn current_session_id(&self) -> Option<&str> {
        None
    }
    
    fn set_session_id(&mut self, session_id: String) {
        *self.session_id.blocking_write() = Some(session_id);
    }
}

impl MongoPersistentMemory {
    async fn do_delete_session(&self, session_id: &str) -> Result<(), MemoryError> {
        let collection = self.collection();
        
        collection
            .delete_one(doc! { "session_id": session_id }, None)
            .await
            .map_err(|e| MemoryError::ClearError(format!("MongoDB delete failed: {}", e)))?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mongo_memory_doc_from_memory_data() {
        let data = MemoryData::new("session_123".to_string());
        let doc: MongoMemoryDoc = data.into();
        
        assert_eq!(doc.session_id, "session_123");
        assert!(doc.messages.is_empty());
    }
    
    #[test]
    fn test_memory_data_from_mongo_doc() {
        let doc = MongoMemoryDoc {
            id: None,
            session_id: "session_456".to_string(),
            messages: vec![Message::human("Hello")],
            summary: Some("Test summary".to_string()),
            metadata: HashMap::new(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        };
        
        let data: MemoryData = doc.into();
        assert_eq!(data.session_id, "session_456");
        assert_eq!(data.messages.len(), 1);
    }
}