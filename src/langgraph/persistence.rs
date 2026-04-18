// src/langgraph/persistence.rs
//! Graph persistence for serialization and storage
//!
//! This module provides persistence capabilities for graph definitions,
//! allowing graphs to be saved, loaded, and shared across sessions.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// GraphPersistence trait for storing and loading graph definitions
#[async_trait]
pub trait GraphPersistence: Send + Sync {
    /// Save a graph definition with the given ID
    async fn save(&self, id: &str, definition: &GraphDefinition) -> Result<(), PersistenceError>;
    
    /// Load a graph definition by ID
    async fn load(&self, id: &str) -> Result<GraphDefinition, PersistenceError>;
    
    /// Delete a graph definition by ID
    async fn delete(&self, id: &str) -> Result<(), PersistenceError>;
    
    /// Check if a graph definition exists
    async fn exists(&self, id: &str) -> Result<bool, PersistenceError>;
    
    /// List all stored graph IDs
    async fn list(&self) -> Result<Vec<String>, PersistenceError>;
}

/// Persistence error types
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error("Graph '{0}' not found")]
    NotFound(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Invalid graph definition: {0}")]
    InvalidDefinition(String),
    
    #[error("MongoDB error: {0}")]
    MongoError(String),
    
    #[error("Connection error: {0}")]
    ConnectionError(String),
}

/// GraphDefinition - Serializable graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDefinition {
    /// Unique identifier
    pub id: String,
    
    /// Human-readable name
    pub name: Option<String>,
    
    /// Entry point node name
    pub entry_point: String,
    
    /// Node definitions
    pub nodes: Vec<NodeDefinition>,
    
    /// Edge definitions
    pub edges: Vec<EdgeDefinition>,
    
    /// Router definitions
    pub routers: Vec<RouterDefinition>,
    
    /// Maximum recursion limit
    pub recursion_limit: usize,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphDefinition {
    /// Create a new graph definition with the given entry point
    pub fn new(entry_point: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: None,
            entry_point,
            nodes: Vec::new(),
            edges: Vec::new(),
            routers: Vec::new(),
            recursion_limit: 25,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }
    
    /// Set a custom ID
    pub fn with_id(mut self, id: String) -> Self {
        self.id = id;
        self
    }
    
    /// Set a human-readable name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Set recursion limit
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }
    
    /// Add a node definition
    pub fn add_node(&mut self, node: NodeDefinition) {
        self.nodes.push(node);
        self.updated_at = Utc::now();
    }
    
    /// Add an edge definition
    pub fn add_edge(&mut self, edge: EdgeDefinition) {
        self.edges.push(edge);
        self.updated_at = Utc::now();
    }
    
    /// Add a router definition
    pub fn add_router(&mut self, router: RouterDefinition) {
        self.routers.push(router);
        self.updated_at = Utc::now();
    }
}

/// NodeDefinition - Serializable node structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefinition {
    /// Node name
    pub name: String,
    
    /// Node type
    pub node_type: NodeType,
    
    /// Custom configuration
    pub config: serde_json::Value,
}

/// NodeType - Type of node execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Sync,
    Async,
    Subgraph,
    Custom,
}

/// EdgeDefinition - Serializable edge structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDefinition {
    /// Edge type
    pub edge_type: EdgeType,
    
    /// Source node name
    pub source: String,
    
    /// Target node name (for fixed edges)
    pub target: Option<String>,
    
    /// Multiple targets (for fan-out edges)
    pub targets: Option<Vec<String>>,
    
    /// Router name (for conditional edges)
    pub router_name: Option<String>,
    
    /// Conditional targets mapping (route -> target)
    pub conditional_targets: Option<HashMap<String, String>>,
    
    /// Default target for conditional edges
    pub default_target: Option<String>,
    
    /// Source nodes for fan-in edges
    pub sources: Option<Vec<String>>,
}

impl EdgeDefinition {
    /// Create a fixed edge
    pub fn fixed(source: String, target: String) -> Self {
        Self {
            edge_type: EdgeType::Fixed,
            source,
            target: Some(target),
            targets: None,
            router_name: None,
            conditional_targets: None,
            default_target: None,
            sources: None,
        }
    }
    
    /// Create a conditional edge
    pub fn conditional(
        source: String,
        router_name: String,
        targets: HashMap<String, String>,
        default_target: Option<String>,
    ) -> Self {
        Self {
            edge_type: EdgeType::Conditional,
            source,
            target: None,
            targets: None,
            router_name: Some(router_name),
            conditional_targets: Some(targets),
            default_target,
            sources: None,
        }
    }
    
    /// Create a fan-out edge (parallel execution)
    pub fn fan_out(source: String, targets: Vec<String>) -> Self {
        Self {
            edge_type: EdgeType::FanOut,
            source,
            target: None,
            targets: Some(targets),
            router_name: None,
            conditional_targets: None,
            default_target: None,
            sources: None,
        }
    }
    
    /// Create a fan-in edge (merge from parallel branches)
    pub fn fan_in(sources: Vec<String>, target: String) -> Self {
        Self {
            edge_type: EdgeType::FanIn,
            source: "__fan_in__".to_string(),
            target: Some(target),
            targets: None,
            router_name: None,
            conditional_targets: None,
            default_target: None,
            sources: Some(sources),
        }
    }
}

/// EdgeType - Type of edge connection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    /// Fixed transition
    Fixed,
    
    /// Conditional routing
    Conditional,
    
    /// Parallel fan-out
    FanOut,
    
    /// Merge fan-in
    FanIn,
}

/// RouterDefinition - Serializable router structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterDefinition {
    /// Router name
    pub name: String,
    
    /// Router type (e.g., "function", "state_key")
    pub router_type: String,
    
    /// Possible routes
    pub routes: Vec<String>,
    
    /// Custom configuration
    pub config: serde_json::Value,
}

/// MemoryPersistence - In-memory graph storage
pub struct MemoryPersistence {
    graphs: Mutex<HashMap<String, GraphDefinition>>,
}

impl MemoryPersistence {
    /// Create a new memory persistence instance
    pub fn new() -> Self {
        Self {
            graphs: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for MemoryPersistence {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GraphPersistence for MemoryPersistence {
    async fn save(&self, id: &str, definition: &GraphDefinition) -> Result<(), PersistenceError> {
        let mut graphs = self.graphs.lock().await;
        graphs.insert(id.to_string(), definition.clone());
        Ok(())
    }
    
    async fn load(&self, id: &str) -> Result<GraphDefinition, PersistenceError> {
        let graphs = self.graphs.lock().await;
        graphs.get(id)
            .cloned()
            .ok_or_else(|| PersistenceError::NotFound(id.to_string()))
    }
    
    async fn delete(&self, id: &str) -> Result<(), PersistenceError> {
        let mut graphs = self.graphs.lock().await;
        graphs.remove(id)
            .map(|_| ())
            .ok_or_else(|| PersistenceError::NotFound(id.to_string()))
    }
    
    async fn exists(&self, id: &str) -> Result<bool, PersistenceError> {
        let graphs = self.graphs.lock().await;
        Ok(graphs.contains_key(id))
    }
    
    async fn list(&self) -> Result<Vec<String>, PersistenceError> {
        let graphs = self.graphs.lock().await;
        Ok(graphs.keys().cloned().collect())
    }
}

/// FilePersistence - File-based graph storage
pub struct FilePersistence {
    directory: PathBuf,
}

impl FilePersistence {
    /// Create a new file persistence instance
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        let dir = directory.into();
        if !dir.exists() {
            std::fs::create_dir_all(&dir).ok();
        }
        Self { directory: dir }
    }
    
    fn graph_path(&self, id: &str) -> PathBuf {
        self.directory.join(format!("{}.json", id))
    }
}

impl Default for FilePersistence {
    fn default() -> Self {
        Self::new(".graph_definitions")
    }
}

#[async_trait]
impl GraphPersistence for FilePersistence {
    async fn save(&self, id: &str, definition: &GraphDefinition) -> Result<(), PersistenceError> {
        let path = self.graph_path(id);
        
        let json = serde_json::to_string_pretty(definition)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        
        std::fs::write(&path, json)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        
        Ok(())
    }
    
    async fn load(&self, id: &str) -> Result<GraphDefinition, PersistenceError> {
        let path = self.graph_path(id);
        
        if !path.exists() {
            return Err(PersistenceError::NotFound(id.to_string()));
        }
        
        let json = std::fs::read_to_string(&path)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        
        let definition: GraphDefinition = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;
        
        Ok(definition)
    }
    
    async fn delete(&self, id: &str) -> Result<(), PersistenceError> {
        let path = self.graph_path(id);
        
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        }
        
        Ok(())
    }
    
    async fn exists(&self, id: &str) -> Result<bool, PersistenceError> {
        let path = self.graph_path(id);
        Ok(path.exists())
    }
    
    async fn list(&self) -> Result<Vec<String>, PersistenceError> {
        let mut ids = Vec::new();
        
        let entries = std::fs::read_dir(&self.directory)
            .map_err(|e| PersistenceError::IoError(e.to_string()))?;
        
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
}

// ============================================================================
// MongoDB 持久化实现
// ============================================================================

#[cfg(feature = "mongodb-persistence")]
mod mongo_impl {
    use super::*;
    use mongodb::{
        Client, Collection,
        bson::{doc, Document, from_document, to_document},
        options::{ClientOptions, FindOptions},
    };
    
    /// MongoDB配置
    pub struct MongoConfig {
        /// MongoDB连接URI (例如: mongodb://localhost:27017 或 mongodb+srv://user:pass@cluster.mongodb.net)
        pub uri: String,
        
        /// 数据库名称
        pub database: String,
        
        /// 集合名称
        pub collection: String,
    }
    
    impl MongoConfig {
        /// 创建新的MongoDB配置
        pub fn new(uri: String, database: String, collection: String) -> Self {
            Self { uri, database, collection }
        }
        
        /// 从环境变量创建配置
        /// 
        /// 环境变量:
        /// - MONGO_URI: MongoDB连接URI
        /// - MONGO_DATABASE: 数据库名称 (默认: langgraph)
        /// - MONGO_COLLECTION: 集合名称 (默认: graph_definitions)
        pub fn from_env() -> Self {
            Self {
                uri: std::env::var("MONGO_URI")
                    .expect("MONGO_URI environment variable not set"),
                database: std::env::var("MONGO_DATABASE")
                    .unwrap_or_else(|_| "langgraph".to_string()),
                collection: std::env::var("MONGO_COLLECTION")
                    .unwrap_or_else(|_| "graph_definitions".to_string()),
            }
        }
    }
    
    /// MongoPersistence - MongoDB图存储实现
    pub struct MongoPersistence {
        client: Client,
        collection: Collection<Document>,
        database_name: String,
        collection_name: String,
    }
    
    impl MongoPersistence {
        /// 创建新的MongoDB持久化实例
        /// 
        /// # 参数
        /// - config: MongoDB配置
        /// 
        /// # 示例
        /// ```ignore
        /// let config = MongoConfig::new(
        ///     "mongodb://localhost:27017",
        ///     "langgraph",
        ///     "graph_definitions"
        /// );
        /// let persistence = MongoPersistence::new(config).await?;
        /// ```
        pub async fn new(config: MongoConfig) -> Result<Self, PersistenceError> {
            let client_options = ClientOptions::parse(&config.uri)
                .await
                .map_err(|e| PersistenceError::ConnectionError(e.to_string()))?;
            
            let client = Client::with_options(client_options)
                .map_err(|e| PersistenceError::ConnectionError(e.to_string()))?;
            
            let database = client.database(&config.database);
            let collection = database.collection(&config.collection);
            
            Ok(Self { 
                client, 
                collection,
                database_name: config.database,
                collection_name: config.collection,
            })
        }
        
        /// 从环境变量创建实例
        pub async fn from_env() -> Result<Self, PersistenceError> {
            let config = MongoConfig::from_env();
            Self::new(config).await
        }
        
        /// 获取MongoDB客户端
        pub fn client(&self) -> &Client {
            &self.client
        }
        
        /// 获取集合名称
        pub fn collection_name(&self) -> &str {
            &self.collection_name
        }
        
        /// 获取数据库名称
        pub fn database_name(&self) -> &str {
            &self.database_name
        }
    }
    
    #[async_trait]
    impl GraphPersistence for MongoPersistence {
        async fn save(&self, id: &str, definition: &GraphDefinition) -> Result<(), PersistenceError> {
            let doc = to_document(definition)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            
            // 使用 upsert 操作：如果存在则更新，不存在则插入
            self.collection
                .update_one(
                    doc! { "_id": id },
                    doc! { "$set": doc },
                    mongodb::options::UpdateOptions::builder()
                        .upsert(true)
                        .build(),
                )
                .await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?;
            
            Ok(())
        }
        
        async fn load(&self, id: &str) -> Result<GraphDefinition, PersistenceError> {
            let filter = doc! { "_id": id };
            
            let result = self.collection
                .find_one(filter, None)
                .await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?;
            
            match result {
                Some(doc) => {
                    let definition: GraphDefinition = from_document(doc)
                        .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;
                    Ok(definition)
                }
                None => Err(PersistenceError::NotFound(id.to_string())),
            }
        }
        
        async fn delete(&self, id: &str) -> Result<(), PersistenceError> {
            let filter = doc! { "_id": id };
            
            let result = self.collection
                .delete_one(filter, None)
                .await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?;
            
            if result.deleted_count == 0 {
                Err(PersistenceError::NotFound(id.to_string()))
            } else {
                Ok(())
            }
        }
        
        async fn exists(&self, id: &str) -> Result<bool, PersistenceError> {
            let filter = doc! { "_id": id };
            
            let count = self.collection
                .count_documents(filter, None)
                .await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?;
            
            Ok(count > 0)
        }
        
        async fn list(&self) -> Result<Vec<String>, PersistenceError> {
            let filter = doc! {};
            let options = FindOptions::builder()
                .projection(doc! { "_id": 1 })
                .build();
            
            let mut cursor = self.collection
                .find(filter, options)
                .await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?;
            
            let mut ids = Vec::new();
            while cursor.advance().await
                .map_err(|e| PersistenceError::MongoError(e.to_string()))?
            {
                let doc = cursor.deserialize_current()
                    .map_err(|e| PersistenceError::DeserializationError(e.to_string()))?;
                
                if let Some(id) = doc.get_str("_id").ok() {
                    ids.push(id.to_string());
                }
            }
            
            Ok(ids)
        }
    }
}

#[cfg(feature = "mongodb-persistence")]
pub use mongo_impl::{MongoPersistence, MongoConfig};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_type_serialization() {
        let types = vec![NodeType::Sync, NodeType::Async, NodeType::Subgraph, NodeType::Custom];
        
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let parsed: NodeType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, t);
        }
    }
    
    #[test]
    fn test_edge_type_serialization() {
        let types = vec![EdgeType::Fixed, EdgeType::Conditional, EdgeType::FanOut, EdgeType::FanIn];
        
        for t in types {
            let json = serde_json::to_string(&t).unwrap();
            let parsed: EdgeType = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, t);
        }
    }
    
    #[test]
    fn test_graph_definition_builder() {
        let def = GraphDefinition::new("entry".to_string())
            .with_id("test-id".to_string())
            .with_name("Test Graph".to_string())
            .with_recursion_limit(50);
        
        assert_eq!(def.id, "test-id");
        assert_eq!(def.name, Some("Test Graph".to_string()));
        assert_eq!(def.entry_point, "entry");
        assert_eq!(def.recursion_limit, 50);
    }
    
    #[tokio::test]
    async fn test_memory_persistence() {
        let persistence = MemoryPersistence::new();
        let def = GraphDefinition::new("entry".to_string())
            .with_id("test-001".to_string());
        
        persistence.save("test-001", &def).await.unwrap();
        assert!(persistence.exists("test-001").await.unwrap());
        
        let loaded = persistence.load("test-001").await.unwrap();
        assert_eq!(loaded.id, "test-001");
        
        persistence.delete("test-001").await.unwrap();
        assert!(!persistence.exists("test-001").await.unwrap());
    }
}