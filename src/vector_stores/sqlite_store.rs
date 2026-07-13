// src/vector_stores/sqlite_store.rs
//! SQLite 文档存储实现

use async_trait::async_trait;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use std::path::Path;

use super::{Document, VectorStoreError};
use super::document_store::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use crate::retrieval::TextSplitter;

#[derive(Debug, Clone)]
pub struct SQLiteStoreConfig {
    pub db_path: String,
}

impl Default for SQLiteStoreConfig {
    fn default() -> Self {
        Self { db_path: "langchainrust.db".to_string() }
    }
}

impl SQLiteStoreConfig {
    pub fn new(path: impl Into<String>) -> Self {
        Self { db_path: path.into() }
    }
}

pub struct SQLiteDocumentStore {
    conn: Arc<Mutex<Connection>>,
}

impl SQLiteDocumentStore {
    pub fn new(config: SQLiteStoreConfig) -> Result<Self, VectorStoreError> {
        let conn = Connection::open(&config.db_path)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL,
                content TEXT NOT NULL, segment INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_segment ON chunks(parent_id, segment);"
        ).map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }
}

#[async_trait]
impl DocumentStore for SQLiteDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());
        let conn = self.conn.lock().await;
        conn.execute("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, document.content, meta])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(id)
    }

    async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>, VectorStoreError> {
        let conn = self.conn.lock().await;
        let mut ids = Vec::new();
        for doc in documents {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            let meta = serde_json::to_string(&doc.metadata).unwrap_or_else(|_| "{}".to_string());
            conn.execute("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
                rusqlite::params![id, doc.content, meta])
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT id, content, metadata FROM documents WHERE id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![id], |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let meta_str: String = row.get(2)?;
            Ok(Document {
                id: Some(id), content,
                metadata: serde_json::from_str(&meta_str).unwrap_or_default(),
            })
        });
        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM chunks WHERE parent_id = ?1", rusqlite::params![id])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        conn.execute("DELETE FROM documents WHERE id = ?1", rusqlite::params![id])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().await;
        conn.execute_batch("DELETE FROM chunks; DELETE FROM documents;")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl ChunkedDocumentStoreTrait for SQLiteDocumentStore {
    async fn add_parent_document(&self, document: Document, chunk_size: usize) -> Result<(String, Vec<String>), VectorStoreError> {
        let splitter = crate::retrieval::RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks_text = splitter.split_text(&document.content);
        let parent_id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());

        let conn = self.conn.lock().await;
        conn.execute("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![parent_id, document.content, meta])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut chunk_ids = Vec::new();
        for (i, text) in chunks_text.iter().enumerate() {
            let cid = format!("{}:chunk:{}", parent_id, i);
            conn.execute("INSERT OR REPLACE INTO chunks (chunk_id, parent_id, content, segment, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![cid, parent_id, text, i, "{}"])
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunk_ids.push(cid);
        }
        Ok((parent_id, chunk_ids))
    }

    async fn add_parent_documents(&self, documents: Vec<Document>, chunk_size: usize) -> Result<Vec<(String, Vec<String>)>, VectorStoreError> {
        let mut results = Vec::new();
        for doc in documents { results.push(self.add_parent_document(doc, chunk_size).await?); }
        Ok(results)
    }

    async fn get_parent_document(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        self.get_document(parent_id).await
    }

    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE chunk_id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![chunk_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?, parent_id: row.get(1)?,
                content: row.get(2)?, segment: row.get(3)?,
                metadata: serde_json::from_str::<std::collections::HashMap<String, String>>(&row.get::<_, String>(4)?).unwrap_or_default(),
            })
        });
        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    async fn get_chunk_document(&self, chunk_id: &str) -> Result<Option<Document>, VectorStoreError> {
        Ok(self.get_chunk(chunk_id).await?.map(|c| c.to_document()))
    }

    async fn get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE parent_id = ?1 ORDER BY segment")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt.query_map(rusqlite::params![parent_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?, parent_id: row.get(1)?,
                content: row.get(2)?, segment: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            })
        }).map_err(|e| VectorStoreError::StorageError(e.to_string()))?
        .filter_map(|r| r.ok()).collect();
        Ok(chunks)
    }

    async fn get_chunk_documents_for_parent(&self, parent_id: &str) -> Result<Vec<Document>, VectorStoreError> {
        Ok(self.get_chunks_for_parent(parent_id).await?.into_iter().map(|c| c.to_document()).collect())
    }

    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError> {
        self.delete_document(parent_id).await
    }

    async fn parent_count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)).unwrap_or(0)
    }

    async fn chunk_count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).unwrap_or(0)
    }

    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt.query_map([], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?, parent_id: row.get(1)?,
                content: row.get(2)?, segment: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            })
        }).map_err(|e| VectorStoreError::StorageError(e.to_string()))?
        .filter_map(|r| r.ok()).collect();
        Ok(chunks)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().await;
        conn.execute_batch("DELETE FROM chunks; DELETE FROM documents;")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn save(&self, _path: impl AsRef<Path> + Send) -> Result<(), VectorStoreError> {
        Ok(()) // SQLite auto-saves
    }

    // Blocking methods
    fn add_parent_document_blocking(&self, document: Document, chunk_size: usize) -> Result<(String, Vec<String>), VectorStoreError> {
        let conn = self.conn.blocking_lock();
        let splitter = crate::retrieval::RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks_text = splitter.split_text(&document.content);
        let parent_id = document.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());

        conn.execute("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![parent_id, document.content, meta])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut chunk_ids = Vec::new();
        for (i, text) in chunks_text.iter().enumerate() {
            let cid = format!("{}:chunk:{}", parent_id, i);
            conn.execute("INSERT OR REPLACE INTO chunks (chunk_id, parent_id, content, segment, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![cid, parent_id, text, i, "{}"])
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunk_ids.push(cid);
        }
        Ok((parent_id, chunk_ids))
    }

    fn get_parent_document_blocking(&self, parent_id: &str) -> Result<Option<Document>, VectorStoreError> {
        let conn = self.conn.blocking_lock();
        let mut stmt = conn.prepare("SELECT id, content, metadata FROM documents WHERE id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![parent_id], |row| {
            Ok(Document {
                id: Some(row.get(0)?), content: row.get(1)?,
                metadata: serde_json::from_str(&row.get::<_, String>(2)?).unwrap_or_default(),
            })
        });
        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    fn get_chunk_blocking(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let conn = self.conn.blocking_lock();
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE chunk_id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![chunk_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?, parent_id: row.get(1)?,
                content: row.get(2)?, segment: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            })
        });
        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    fn blocking_get_chunks_for_parent(&self, parent_id: &str) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.conn.blocking_lock();
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE parent_id = ?1 ORDER BY segment")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt.query_map(rusqlite::params![parent_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?, parent_id: row.get(1)?,
                content: row.get(2)?, segment: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
            })
        }).map_err(|e| VectorStoreError::StorageError(e.to_string()))?
        .filter_map(|r| r.ok()).collect();
        Ok(chunks)
    }
}
