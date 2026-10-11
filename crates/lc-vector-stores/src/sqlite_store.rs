// lc-vector-stores/src/sqlite_store.rs
//! SQLite document store implementation

use async_trait::async_trait;
use lc_shared::splitter::{RecursiveCharacterSplitter, TextSplitter};
use rusqlite::Connection;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::ops::{Deref, DerefMut};
use uuid::Uuid;

use crate::document_store::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use crate::{Document, VectorStoreError};

/// Parses the metadata column; on corruption, logs and falls back to an empty map instead of silently swallowing the error.
fn parse_metadata_or_default(raw: &str) -> std::collections::HashMap<String, serde_json::Value> {
    serde_json::from_str(raw).unwrap_or_else(|e| {
        log::warn!(
            "SQLite metadata in store is corrupted, falling back to empty map: {}",
            e
        );
        std::collections::HashMap::new()
    })
}

/// SQLite document store configuration
#[derive(Debug, Clone)]
pub struct SQLiteStoreConfig {
    /// SQLite database file path
    pub db_path: String,
}

impl Default for SQLiteStoreConfig {
    fn default() -> Self {
        Self {
            db_path: "langchainrust.db".to_string(),
        }
    }
}

impl SQLiteStoreConfig {
    /// Creates a config from a database file path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            db_path: path.into(),
        }
    }
}

/// SQLite document store implementation
///
/// Uses a small pool of connections to the same file in WAL mode. A single `Mutex<Connection>` (as
/// before) serialized every read AND write behind one lock for the duration of each operation; the
/// pool instead lets operations draw their own connection for the brief window they need it, so
/// concurrent readers run in parallel while the single SQLite writer queues on the file lock
/// (`busy_timeout`) rather than colliding with `SQLITE_BUSY` — the read/write concurrency that a
/// lone connection could never provide.
pub struct SQLiteDocumentStore {
    pool: Arc<Mutex<VecDeque<Connection>>>,
    db_path: String,
    cap: usize,
}

/// RAII handle returning a pooled connection to the store when dropped, on success *or* failure,
/// so every return path (`?`, early `return`, match arms) releases the connection without an
/// explicit call per branch.
struct PooledConn<'a> {
    conn: Option<Connection>,
    store: &'a SQLiteDocumentStore,
}

impl Deref for PooledConn<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("pooled connection is always present")
    }
}

impl DerefMut for PooledConn<'_> {
    fn deref_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("pooled connection is always present")
    }
}

impl Drop for PooledConn<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.store.release(conn);
        }
    }
}

/// Opens a fresh SQLite connection with WAL + a write busy timeout and the schema, so any lazily
/// opened pool member is ready to use immediately.
fn open_configured(db_path: &str) -> Result<Connection, VectorStoreError> {
    let conn = Connection::open(db_path)
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 5000;
         CREATE TABLE IF NOT EXISTS documents (
             id TEXT PRIMARY KEY, content TEXT NOT NULL,
             metadata TEXT NOT NULL DEFAULT '{}'
         );
         CREATE TABLE IF NOT EXISTS chunks (
             chunk_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL,
             content TEXT NOT NULL, segment INTEGER NOT NULL,
             metadata TEXT NOT NULL DEFAULT '{}'
         );
         CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
         CREATE INDEX IF NOT EXISTS idx_chunks_segment ON chunks(parent_id, segment);",
    )
    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
    Ok(conn)
}

impl SQLiteDocumentStore {
    /// Opens the SQLite database and creates the required tables and indexes.
    pub fn new(config: SQLiteStoreConfig) -> Result<Self, VectorStoreError> {
        let warm = open_configured(&config.db_path)?;
        Ok(Self {
            pool: Arc::new(Mutex::new(VecDeque::from([warm]))),
            db_path: config.db_path,
            cap: 4,
        })
    }

    /// Checks a connection out of the pool, opening a fresh one (with schema + WAL) only when the
    /// pool is empty.
    fn checkout(&self) -> Result<Connection, VectorStoreError> {
        if let Some(conn) = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_back()
        {
            Ok(conn)
        } else {
            open_configured(&self.db_path)
        }
    }

    /// Returns a connection to the pool, honoring the cap so heavy concurrency cannot open an
    /// unbounded number of connections; overflow connections are closed.
    fn release(&self, conn: Connection) {
        let mut guard = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if guard.len() < self.cap {
            guard.push_back(conn);
        }
    }

    /// Borrows a pooled connection for the duration of the calling method, returning it on drop.
    fn pooled(&self) -> Result<PooledConn<'_>, VectorStoreError> {
        Ok(PooledConn {
            conn: Some(self.checkout()?),
            store: self,
        })
    }

    /// Explicitly triggers persistence to disk.
    ///
    /// C3: the fake default `ChunkedDocumentStoreTrait::save` has been removed from the trait;
    /// SQLite persists automatically on every write, so this method is a compatibility no-op
    /// (aligned with Redis's `save_to_disk`).
    pub async fn save(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

#[async_trait]
impl DocumentStore for SQLiteDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());
        let conn = self.pooled()?;
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, document.content, meta],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(id)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut conn = self.pooled()?;
        // batch the whole vector in one transaction instead of one auto-commit fsync per row,
        // so a large `add_documents` doesn't hammer the write lock.
        let tx = conn
            .transaction()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let mut ids = Vec::new();
        for doc in documents {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            let meta = serde_json::to_string(&doc.metadata).unwrap_or_else(|_| "{}".to_string());
            tx.execute(
                "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
                rusqlite::params![id, doc.content, meta],
            )
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            ids.push(id);
        }
        tx.commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn
            .prepare("SELECT id, content, metadata FROM documents WHERE id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![id], |row| {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let meta_str: String = row.get(2)?;
            Ok(Document {
                id: Some(id),
                content,
                metadata: parse_metadata_or_default(&meta_str),
            })
        });
        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let conn = self.pooled()?;
        conn.execute(
            "DELETE FROM chunks WHERE parent_id = ?1",
            rusqlite::params![id],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        conn.execute("DELETE FROM documents WHERE id = ?1", rusqlite::params![id])
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }

    async fn count(&self) -> usize {
        // M-23: `count` returns usize (cannot `?`), so a failed pool checkout
        // surfaces as a logged error + 0, matching the error handling below.
        let conn = match self.pooled() {
            Ok(conn) => conn,
            Err(e) => {
                log::error!(
                    "SQLite pool checkout failed in count(documents), returning 0: {}",
                    e
                );
                return 0;
            }
        };
        match conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)) {
            Ok(count) => count,
            // M3: the trait returns usize so errors cannot propagate — stop silently swallowing
            // errors and returning 0; log an error to surface storage failures, consistent with
            // the "no silent row drops" pattern already used in this file.
            Err(e) => {
                log::error!("SQLite count(documents) query failed, returning 0: {}", e);
                0
            }
        }
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let conn = self.pooled()?;
        conn.execute_batch("DELETE FROM chunks; DELETE FROM documents;")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl ChunkedDocumentStoreTrait for SQLiteDocumentStore {
    async fn add_parent_document(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks_text = splitter.split_text(&document.content);
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());

        let mut conn = self.pooled()?;
        // batch the parent doc + chunk writes in one transaction: one fsync for the whole
        // re-ingest instead of one per statement (the "heavy write path").
        let tx = conn
            .transaction()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![parent_id, document.content, meta],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // H5: replace the parent's prior chunk set first so a re-ingest that shrank /
        // changed segmentation leaves no stale `{parent}::{n}` rows behind
        // (INSERT OR REPLACE only overwrites equal ids; higher segment ids from a
        // previous, larger set would otherwise linger and stay retrievable).
        tx.execute(
            "DELETE FROM chunks WHERE parent_id = ?1",
            rusqlite::params![parent_id],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut chunk_ids = Vec::new();
        for (i, text) in chunks_text.iter().enumerate() {
            let cid = crate::chunk_id(parent_id, i);
            tx.execute("INSERT OR REPLACE INTO chunks (chunk_id, parent_id, content, segment, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![cid, parent_id, text, i, "{}"])
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunk_ids.push(cid);
        }
        tx.commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok((parent_id, chunk_ids))
    }

    async fn add_parent_documents(
        &self,
        documents: Vec<Document>,
        chunk_size: usize,
    ) -> Result<Vec<(String, Vec<String>)>, VectorStoreError> {
        let mut results = Vec::new();
        for doc in documents {
            results.push(self.add_parent_document(doc, chunk_size).await?);
        }
        Ok(results)
    }

    async fn get_parent_document(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        self.get_document(parent_id).await
    }

    async fn get_chunk(&self, chunk_id: &str) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE chunk_id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![chunk_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?,
                parent_id: row.get(1)?,
                content: row.get(2)?,
                segment: row.get(3)?,
                metadata: parse_metadata_or_default(&row.get::<_, String>(4)?),
            })
        });
        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    async fn get_chunk_document(
        &self,
        chunk_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        Ok(self.get_chunk(chunk_id).await?.map(|c| c.to_document()))
    }

    async fn get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE parent_id = ?1 ORDER BY segment")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt
            .query_map(rusqlite::params![parent_id], |row| {
                Ok(ChunkDocument {
                    chunk_id: row.get(0)?,
                    parent_id: row.get(1)?,
                    content: row.get(2)?,
                    segment: row.get(3)?,
                    metadata: parse_metadata_or_default(&row.get::<_, String>(4)?),
                })
            })
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
            .filter_map(|r| match r {
                Ok(chunk) => Some(chunk),
                Err(e) => {
                    // stop silently dropping rows: log to surface storage degradation
                    log::error!(
                        "SQLite store: a row failed to deserialize and was dropped from the query result: {}",
                        e
                    );
                    None
                }
            })
            .collect();
        Ok(chunks)
    }

    async fn get_chunk_documents_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<Document>, VectorStoreError> {
        Ok(self
            .get_chunks_for_parent(parent_id)
            .await?
            .into_iter()
            .map(|c| c.to_document())
            .collect())
    }

    async fn delete_parent_document(&self, parent_id: &str) -> Result<(), VectorStoreError> {
        self.delete_document(parent_id).await
    }

    async fn parent_count(&self) -> usize {
        // The trait signature returns a bare `usize`, so a failed pool checkout
        // cannot be foisted onto the caller via `?`; surface it as an honest
        // error log and return 0, matching the query-failure branch below.
        let conn = match self.pooled() {
            Ok(conn) => conn,
            Err(e) => {
                log::error!("SQLite parent_count checkout failed, returning 0: {}", e);
                return 0;
            }
        };
        match conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0)) {
            Ok(count) => count,
            // M3: do not silently swallow errors and return 0; log an error to surface storage failures.
            Err(e) => {
                log::error!("SQLite parent_count query failed, returning 0: {}", e);
                0
            }
        }
    }

    async fn chunk_count(&self) -> usize {
        // Same checkout handling as `parent_count`: the `usize` return type forbids `?`,
// so a failed pool checkout is logged and yields 0 rather than panicking or
// invoking the `?` operator on a non-Result function.
        let conn = match self.pooled() {
            Ok(conn) => conn,
            Err(e) => {
                log::error!("SQLite chunk_count checkout failed, returning 0: {}", e);
                return 0;
            }
        };
        match conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)) {
            Ok(count) => count,
            // M3: do not silently swallow errors and return 0; log an error to surface storage failures.
            Err(e) => {
                log::error!("SQLite chunk_count query failed, returning 0: {}", e);
                0
            }
        }
    }

    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn
            .prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt
            .query_map([], |row| {
                Ok(ChunkDocument {
                    chunk_id: row.get(0)?,
                    parent_id: row.get(1)?,
                    content: row.get(2)?,
                    segment: row.get(3)?,
                    metadata: parse_metadata_or_default(&row.get::<_, String>(4)?),
                })
            })
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
            .filter_map(|r| match r {
                Ok(chunk) => Some(chunk),
                Err(e) => {
                    // stop silently dropping rows: log to surface storage degradation
                    log::error!(
                        "SQLite store: a row failed to deserialize and was dropped from the query result: {}",
                        e
                    );
                    None
                }
            })
            .collect();
        Ok(chunks)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let conn = self.pooled()?;
        conn.execute_batch("DELETE FROM chunks; DELETE FROM documents;")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok(())
    }

    // Blocking methods
    fn add_parent_document_blocking(
        &self,
        document: Document,
        chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        let mut conn = self.pooled()?;
        let splitter = RecursiveCharacterSplitter::new(chunk_size, chunk_size / 10);
        let chunks_text = splitter.split_text(&document.content);
        let parent_id = document
            .id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let meta = serde_json::to_string(&document.metadata).unwrap_or_else(|_| "{}".to_string());

        // batch the re-ingest in one transaction (one fsync instead of one per statement)
        let tx = conn
            .transaction()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?1, ?2, ?3)",
            rusqlite::params![parent_id, document.content, meta],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // H5: drop the parent's prior chunk set so a shrunken re-ingest leaves none behind.
        tx.execute(
            "DELETE FROM chunks WHERE parent_id = ?1",
            rusqlite::params![parent_id],
        )
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        let mut chunk_ids = Vec::new();
        for (i, text) in chunks_text.iter().enumerate() {
            let cid = crate::chunk_id(parent_id, i);
            tx.execute("INSERT OR REPLACE INTO chunks (chunk_id, parent_id, content, segment, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![cid, parent_id, text, i, "{}"])
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            chunk_ids.push(cid);
        }
        tx.commit()
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        Ok((parent_id, chunk_ids))
    }

    fn get_parent_document_blocking(
        &self,
        parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn
            .prepare("SELECT id, content, metadata FROM documents WHERE id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![parent_id], |row| {
            Ok(Document {
                id: Some(row.get(0)?),
                content: row.get(1)?,
                metadata: parse_metadata_or_default(&row.get::<_, String>(2)?),
            })
        });
        match result {
            Ok(doc) => Ok(Some(doc)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    fn get_chunk_blocking(
        &self,
        chunk_id: &str,
    ) -> Result<Option<ChunkDocument>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE chunk_id = ?1")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let result = stmt.query_row(rusqlite::params![chunk_id], |row| {
            Ok(ChunkDocument {
                chunk_id: row.get(0)?,
                parent_id: row.get(1)?,
                content: row.get(2)?,
                segment: row.get(3)?,
                metadata: parse_metadata_or_default(&row.get::<_, String>(4)?),
            })
        });
        match result {
            Ok(chunk) => Ok(Some(chunk)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(VectorStoreError::StorageError(e.to_string())),
        }
    }

    fn blocking_get_chunks_for_parent(
        &self,
        parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let conn = self.pooled()?;
        let mut stmt = conn.prepare("SELECT chunk_id, parent_id, content, segment, metadata FROM chunks WHERE parent_id = ?1 ORDER BY segment")
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let chunks = stmt
            .query_map(rusqlite::params![parent_id], |row| {
                Ok(ChunkDocument {
                    chunk_id: row.get(0)?,
                    parent_id: row.get(1)?,
                    content: row.get(2)?,
                    segment: row.get(3)?,
                    metadata: parse_metadata_or_default(&row.get::<_, String>(4)?),
                })
            })
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
            .filter_map(|r| match r {
                Ok(chunk) => Some(chunk),
                Err(e) => {
                    // stop silently dropping rows: log to surface storage degradation
                    log::error!(
                        "SQLite store: a row failed to deserialize and was dropped from the query result: {}",
                        e
                    );
                    None
                }
            })
            .collect();
        Ok(chunks)
    }
}
