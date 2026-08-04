// lc-vector-stores/src/redis_store.rs
//! Redis 文档存储实现

use async_trait::async_trait;
use lc_shared::splitter::{RecursiveCharacterSplitter, TextSplitter};
use std::path::Path;

use crate::document_store::{ChunkDocument, ChunkedDocumentStoreTrait, DocumentStore};
use crate::{Document, VectorStoreError};

#[derive(Debug, Clone)]
pub struct RedisStoreConfig {
    pub url: String,
    pub key_prefix: String,
}

impl Default for RedisStoreConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            key_prefix: "langchainrust".to_string(),
        }
    }
}

impl RedisStoreConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }
}

pub struct RedisDocumentStore {
    config: RedisStoreConfig,
    #[allow(dead_code)]
    client: redis::Client,
}

impl RedisDocumentStore {
    pub async fn new(config: RedisStoreConfig) -> Result<Self, VectorStoreError> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        // 验证连接
        let _ = client
            .get_connection()
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        Ok(Self { config, client })
    }

    fn doc_key(&self, id: &str) -> String {
        format!("{}:doc:{}", self.config.key_prefix, id)
    }
    fn chunk_key(&self, id: &str) -> String {
        format!("{}:chunk:{}", self.config.key_prefix, id)
    }
    fn parent_chunks_key(&self, pid: &str) -> String {
        format!("{}:pchunks:{}", self.config.key_prefix, pid)
    }
    fn doc_ids_key(&self) -> String {
        format!("{}:doc_ids", self.config.key_prefix)
    }
    fn parent_ids_key(&self) -> String {
        format!("{}:parent_ids", self.config.key_prefix)
    }
    fn all_chunks_key(&self) -> String {
        format!("{}:all_chunks", self.config.key_prefix)
    }
}

#[async_trait]
impl DocumentStore for RedisDocumentStore {
    async fn add_document(&self, document: Document) -> Result<String, VectorStoreError> {
        let id = document
            .id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let json = serde_json::to_string(&document)
            .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
        let config = self.config.clone();
        let id2 = id.clone();

        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SET")
                .arg(format!("{}:doc:{}", config.key_prefix, id2))
                .arg(&json)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            redis::cmd("SADD")
                .arg(format!("{}:doc_ids", config.key_prefix))
                .arg(&id2)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))??;

        Ok(id)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
    ) -> Result<Vec<String>, VectorStoreError> {
        let mut ids = Vec::new();
        for doc in documents {
            ids.push(self.add_document(doc).await?);
        }
        Ok(ids)
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let result = self.get_str(&self.doc_key(id)).await?;
        match result {
            Some(json) => Ok(serde_json::from_str(&json).ok()),
            None => Ok(None),
        }
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        self.del(&self.doc_key(id)).await?;
        self.srem(&self.doc_ids_key(), id).await
    }

    async fn count(&self) -> usize {
        self.scard(&self.doc_ids_key()).await.unwrap_or(0)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // M29: use SCAN + DEL with prefix instead of FLUSHDB to avoid wiping other keys
        self.clear_with_prefix().await
    }
}

impl RedisDocumentStore {
    async fn get_str(&self, key: &str) -> Result<Option<String>, VectorStoreError> {
        let config = self.config.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || -> Result<Option<String>, VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("GET")
                .arg(&key)
                .query(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    async fn del(&self, key: &str) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("DEL")
                .arg(&key)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    #[allow(dead_code)]
    async fn sadd(&self, key: &str, member: &str) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        let (k, m) = (key.to_string(), member.to_string());
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SADD")
                .arg(&k)
                .arg(&m)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    async fn srem(&self, key: &str, member: &str) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        let (k, m) = (key.to_string(), member.to_string());
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SREM")
                .arg(&k)
                .arg(&m)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    async fn smembers(&self, key: &str) -> Result<Vec<String>, VectorStoreError> {
        let config = self.config.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<String>, VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SMEMBERS")
                .arg(&key)
                .query(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    async fn scard(&self, key: &str) -> Result<usize, VectorStoreError> {
        let config = self.config.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || -> Result<usize, VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SCARD")
                .arg(&key)
                .query(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    #[allow(dead_code)]
    async fn flushdb(&self) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("FLUSHDB")
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    /// M29: Clear only keys with the configured prefix using SCAN + DEL
    async fn clear_with_prefix(&self) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

            let pattern = format!("{}:*", config.key_prefix);
            let mut cursor: u64 = 0;

            loop {
                let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(100)
                    .query(&mut conn)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

                if !keys.is_empty() {
                    redis::cmd("DEL")
                        .arg(&keys)
                        .query::<()>(&mut conn)
                        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                }

                cursor = next_cursor;
                if cursor == 0 {
                    break;
                }
            }

            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }
}

#[async_trait]
impl ChunkedDocumentStoreTrait for RedisDocumentStore {
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
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let doc_json = serde_json::to_string(&Document {
            id: Some(parent_id.clone()),
            ..document
        })
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

        // 使用 spawn_blocking 执行所有 Redis 操作
        let config = self.config.clone();
        let pid = parent_id.clone();
        let chunks = chunks_text.clone();

        tokio::task::spawn_blocking(move || -> Result<(String, Vec<String>), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

            redis::cmd("SET")
                .arg(format!("{}:doc:{}", config.key_prefix, pid))
                .arg(&doc_json)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

            let mut chunk_ids = Vec::new();
            for (i, text) in chunks.iter().enumerate() {
                let cid = format!("{}:chunk:{}", pid, i);
                let chunk = ChunkDocument::new(cid.clone(), pid.clone(), text.clone(), i);
                let cjson = serde_json::to_string(&chunk)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                redis::cmd("SET")
                    .arg(format!("{}:chunk:{}", config.key_prefix, cid))
                    .arg(&cjson)
                    .query::<()>(&mut conn)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                redis::cmd("SADD")
                    .arg(format!("{}:pchunks:{}", config.key_prefix, pid))
                    .arg(&cid)
                    .query::<()>(&mut conn)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                redis::cmd("SADD")
                    .arg(format!("{}:all_chunks", config.key_prefix))
                    .arg(&cid)
                    .query::<()>(&mut conn)
                    .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
                chunk_ids.push(cid);
            }
            redis::cmd("SADD")
                .arg(format!("{}:parent_ids", config.key_prefix))
                .arg(&pid)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;
            redis::cmd("SADD")
                .arg(format!("{}:doc_ids", config.key_prefix))
                .arg(&pid)
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))?;

            Ok((pid, chunk_ids))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
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
        match self.get_str(&self.chunk_key(chunk_id)).await? {
            Some(json) => Ok(serde_json::from_str(&json).ok()),
            None => Ok(None),
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
        let ids = self.smembers(&self.parent_chunks_key(parent_id)).await?;
        let mut chunks = Vec::new();
        for id in ids {
            if let Some(c) = self.get_chunk(&id).await? {
                chunks.push(c);
            }
        }
        chunks.sort_by_key(|c| c.segment);
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
        let chunks = self.get_chunks_for_parent(parent_id).await?;
        for chunk in &chunks {
            self.del(&self.chunk_key(&chunk.chunk_id)).await?;
        }
        self.del(&self.doc_key(parent_id)).await?;
        self.del(&self.parent_chunks_key(parent_id)).await?;
        self.srem(&self.doc_ids_key(), parent_id).await?;
        self.srem(&self.parent_ids_key(), parent_id).await
    }

    async fn parent_count(&self) -> usize {
        self.scard(&self.parent_ids_key()).await.unwrap_or(0)
    }

    async fn chunk_count(&self) -> usize {
        self.scard(&self.all_chunks_key()).await.unwrap_or(0)
    }

    async fn get_all_chunks(&self) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        let ids = self.smembers(&self.all_chunks_key()).await?;
        let mut chunks = Vec::new();
        for id in ids {
            if let Some(c) = self.get_chunk(&id).await? {
                chunks.push(c);
            }
        }
        Ok(chunks)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // M29: use SCAN + DEL with prefix instead of FLUSHDB to avoid wiping other keys
        self.clear_with_prefix().await
    }

    async fn save(&self, _path: impl AsRef<Path> + Send) -> Result<(), VectorStoreError> {
        let config = self.config.clone();
        tokio::task::spawn_blocking(move || -> Result<(), VectorStoreError> {
            let mut conn = redis::Client::open(config.url.as_str())
                .and_then(|c| c.get_connection())
                .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
            redis::cmd("SAVE")
                .query::<()>(&mut conn)
                .map_err(|e| VectorStoreError::StorageError(e.to_string()))
        })
        .await
        .map_err(|e| VectorStoreError::StorageError(e.to_string()))?
    }

    fn add_parent_document_blocking(
        &self,
        _document: Document,
        _chunk_size: usize,
    ) -> Result<(String, Vec<String>), VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "blocking not supported, use async API".to_string(),
        ))
    }

    fn get_parent_document_blocking(
        &self,
        _parent_id: &str,
    ) -> Result<Option<Document>, VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "blocking not supported, use async API".to_string(),
        ))
    }

    fn get_chunk_blocking(
        &self,
        _chunk_id: &str,
    ) -> Result<Option<ChunkDocument>, VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "blocking not supported, use async API".to_string(),
        ))
    }

    fn blocking_get_chunks_for_parent(
        &self,
        _parent_id: &str,
    ) -> Result<Vec<ChunkDocument>, VectorStoreError> {
        Err(VectorStoreError::StorageError(
            "blocking not supported, use async API".to_string(),
        ))
    }
}
