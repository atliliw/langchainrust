// src/vector_stores/mongo_document_store.rs
//! MongoDB 文档存储实现
//!
//! 生产环境推荐使用 MongoDB 作为 ChunkedDocumentStore 后端：
//! - 支持持久化
//! - 支持分片和复制集
//! - 文档结构天然匹配
//! - 支持索引查询

use super::{Document, VectorStoreError};
use super::document_store::{ChunkDocument, ChunkedDocumentStoreTrait};
use crate::retrieval::{RecursiveCharacterSplitter, TextSplitter};
use async_trait::async_trait;
use mongodb::{
    bson::doc,
    options::{ClientOptions},
    Client, Collection,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MongoParentDoc {
    #[serde(rename = "_id")]
    id: String,
    content: String,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MongoChunkDoc {
    #[serde(rename = "_id")]
    chunk_id: String,
    parent_id: String,
    content: String,
    segment: i32,
    metadata: HashMap<String, String>,
}

impl From<MongoParentDoc> for Document {
    fn from(m: MongoParentDoc) -> Self {
        Document {
            content: m.content,
            metadata: m.metadata,
            id: Some(m.id),
        }
    }
}

impl From<Document> for MongoParentDoc {
    fn from(d: Document) -> Self {
        MongoParentDoc {
            id: d.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            content: d.content,
            metadata: d.metadata,
        }
    }
}

impl From<MongoChunkDoc> for ChunkDocument {
    fn from(m: MongoChunkDoc) -> Self {
        ChunkDocument {
            chunk_id: m.chunk_id,
            parent_id: m.parent_id,
            content: m.content,
            segment: m.segment as usize,
            metadata: m.metadata,
        }
    }
}

impl From<ChunkDocument> for MongoChunkDoc {
    fn from(c: ChunkDocument) -> Self {
        MongoChunkDoc {
            chunk_id: c.chunk_id,
            parent_id: c.parent_id,
            content: c.content,
            segment: c.segment as i32,
            metadata: c.metadata,
        }
    }
}

/// MongoDB 存储配置
#[derive(Debug, Clone)]
pub struct MongoStoreConfig {
    pub uri: String,
    pub database: String,
    pub parent_collection: String,
    pub chunk_collection: String,
}

impl Default for MongoStoreConfig {
    fn default() -> Self {
        Self {
            uri: "mongodb://localhost:27017".to_string(),
            database: "langchainrust".to_string(),
            parent_collection: "parent_docs".to_string(),
            chunk_collection: "chunks".to_string(),
        }
    }
}

impl MongoStoreConfig {
    pub fn new(uri: impl Into<String>, database: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            database: database.into(),
            parent_collection: "parent_docs".to_string(),
            chunk_collection: "chunks".to_string(),
        }
    }
    
    pub fn with_collections(mut self, parent: impl Into<String>, chunk: impl Into<String>) -> Self {
        self.parent_collection = parent.into();
        self.chunk_collection = chunk.into();
        self
    }
}

/// MongoDB ChunkedDocumentStore 实现
pub struct MongoChunkedDocumentStore {
    client: Client,
    parent_collection: Collection<MongoParentDoc>,
    chunk_collection: Collection<MongoChunkDoc>,
}

impl MongoChunkedDocumentStore {
    pub async fn new(config: MongoStoreConfig) -> Result<Self, VectorStoreError> {
        let client_options = ClientOptions::parse(&config.uri)
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        
        let client = Client::with_options(client_options)
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        
        let db = client.database(&config.database);
        let parent_collection = db.collection(&config.parent_collection);
        let chunk_collection = db.collection(&config.chunk_collection);
        
        Ok(Self {
            client,
            parent_collection,
            chunk_collection,
        })
    }
    
    pub async fn create_indexes(&self) -> Result<(), VectorStoreError> {
        self.chunk_collection
            .create_index(
                mongodb::IndexModel::builder()
                    .keys(doc! { "parent_id": 1 })
                    .build(),
                None,
            )
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        Ok(())
    }
    
    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[async_trait]
impl ChunkedDocumentStoreTrait for MongoChunkedDocumentStore {
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let parent_id = document.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        
        let mongo_parent = MongoParentDoc {
            id: parent_id.clone(),
            content: document.content.clone(),
            metadata: document.metadata.clone(),
        };
        
        self.parent_collection
            .insert_one(mongo_parent, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks = splitter.split_text(&document.content);
        
        let mut chunk_ids = Vec::new();
        
        for (segment, chunk_content) in chunks.into_iter().enumerate() {
            let chunk_id = format!("{}_{}", parent_id, segment);
            
            let mongo_chunk = MongoChunkDoc {
                chunk_id: chunk_id.clone(),
                parent_id: parent_id.clone(),
                content: chunk_content,
                segment: segment as i32,
                metadata: HashMap::new(),
            };
            
            self.chunk_collection
                .insert_one(mongo_chunk, None)
                .await
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            
            chunk_ids.push(chunk_id);
        }
        
        Ok((parent_id, chunk_ids))
    }
    
    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError> {
        let mut results = Vec::new();
        for doc in documents {
            let result = self.add_parent_document(doc, chunk_size).await?;
            results.push(result);
        }
        Ok(results)
    }
    
    async fn get_parent_document(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let result = self.parent_collection
            .find_one(doc! { "_id": parent_id }, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        Ok(result.map(|m| m.into()))
    }
    
    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let result = self.chunk_collection
            .find_one(doc! { "_id": chunk_id }, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        Ok(result.map(|m| m.into()))
    }
    
    async fn get_chunk_document(&self, chunk_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let chunk = self.get_chunk(chunk_id).await?;
        Ok(chunk.map(|c| c.to_document()))
    }
    
    async fn get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let options = mongodb::options::FindOptions::builder()
            .sort(doc! { "segment": 1 })
            .build();
        
        let mut cursor = self.chunk_collection
            .find(doc! { "parent_id": parent_id }, options)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        let mut chunks = Vec::new();
        while cursor.advance().await.map_err(|e| VectorStoreError::StorageError(e.to_string()))? {
            let doc = cursor.deserialize_current()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunks.push(doc.into());
        }
        
        Ok(chunks)
    }
    
    async fn get_chunk_documents_for_parent(&self, parent_id: &str) -> Result<Vec<Document>, VectorStoreError> {
        let chunks = self.get_chunks_for_parent(parent_id).await?;
        Ok(chunks.into_iter().map(|c| c.to_document()).collect())
    }
    
    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError> {
        self.chunk_collection
            .delete_many(doc! { "parent_id": parent_id }, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        self.parent_collection
            .delete_one(doc! { "_id": parent_id }, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        Ok(())
    }
    
    async fn parent_count(&self) -> usize {
        self.parent_collection
            .count_documents(doc! {}, None)
            .await
            .unwrap_or(0) as usize
    }
    
    async fn chunk_count(&self) -> usize {
        self.chunk_collection
            .count_documents(doc! {}, None)
            .await
            .unwrap_or(0) as usize
    }
    
    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let mut cursor = self.chunk_collection
            .find(doc! {}, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        let mut chunks = Vec::new();
        while cursor.advance().await.map_err(|e| VectorStoreError::StorageError(e.to_string()))? {
            let doc = cursor.deserialize_current()
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunks.push(doc.into());
        }
        
        Ok(chunks)
    }
    
    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.parent_collection
            .delete_many(doc! {}, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        self.chunk_collection
            .delete_many(doc! {}, None)
            .await
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        
        Ok(())
    }
    
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                self.add_parent_document(document, chunk_size)
            )
        })
    }
    
    fn get_parent_document_blocking(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                self.get_parent_document(parent_id)
            )
        })
    }
    
    fn get_chunk_blocking(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                self.get_chunk(chunk_id)
            )
        })
    }
    
    fn blocking_get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                self.get_chunks_for_parent(parent_id)
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = MongoStoreConfig::new("mongodb://localhost:27017", "test_db");
        assert_eq!(config.uri, "mongodb://localhost:27017");
        assert_eq!(config.database, "test_db");
    }
    
    #[test]
    fn test_mongo_parent_doc_conversion() {
        let doc = Document::new("test content").with_id("test_id");
        let mongo: MongoParentDoc = doc.clone().into();
        assert_eq!(mongo.id, "test_id");
        assert_eq!(mongo.content, "test content");
        
        let back: Document = mongo.into();
        assert_eq!(back.content, "test content");
    }
    
    #[test]
    fn test_mongo_chunk_doc_conversion() {
        let chunk = ChunkDocument::new("chunk_0".to_string(), "parent_1".to_string(), "content".to_string(), 0);
        let mongo: MongoChunkDoc = chunk.clone().into();
        assert_eq!(mongo.chunk_id, "chunk_0");
        assert_eq!(mongo.parent_id, "parent_1");
        
        let back: ChunkDocument = mongo.into();
        assert_eq!(back.chunk_id, "chunk_0");
    }
}