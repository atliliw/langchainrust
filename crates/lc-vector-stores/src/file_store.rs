// lc-vector-stores/src/file_store.rs
//! File-persistent vector store
//!
//! Persists documents and vectors to a local file (JSON serialization), suitable for personal
//! knowledge bases and offline scenarios. Fills the gap between InMemory (not persistent) and
//! external databases (too heavy), similar to SQLite vs MySQL.
//!
//! # Usage
//! ```ignore
//! use lc_vector_stores::FileVectorStore;
//! use std::path::PathBuf;
//!
//! let store = FileVectorStore::new(PathBuf::from("./my_vectors.json"), 1536).unwrap();
//! // add_documents / similarity_search share the same interface as InMemoryVectorStore
//! // every insert/delete is automatically persisted to disk
//! ```

use crate::{
    cosine_similarity, Document, MetadataFilter, SearchResult, VectorDocument, VectorStore,
    VectorStoreError,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Persistent data structure (serialized as JSON)
#[derive(serde::Serialize, serde::Deserialize)]
struct FileStoreData {
    /// Vector dimension (for validation)
    dimension: usize,
    /// Documents + vectors
    documents: HashMap<String, VectorDocument>,
}

/// File-persistent vector store
///
/// Vectors + metadata are serialized to JSON on disk, loaded at startup, and written back
/// automatically after inserts/deletes. Uses `RwLock` for concurrency safety: read operations
/// take a read lock, write operations take a write lock plus persistence.
pub struct FileVectorStore {
    /// Storage file path
    path: PathBuf,
    /// Vector dimension
    dimension: usize,
    /// In-memory data (kept in sync with the file)
    data: RwLock<FileStoreData>,
}

impl FileVectorStore {
    /// Creates or loads a file vector store
    ///
    /// If the file already exists, loads its data; otherwise creates an empty store.
    ///
    /// # Arguments
    /// * `path` - storage file path (a `.json` extension is recommended)
    /// * `dimension` - vector dimension (for validation; when a file exists the file wins)
    pub async fn new(path: PathBuf, dimension: usize) -> Result<Self, VectorStoreError> {
        let data = if path.exists() {
            let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
                VectorStoreError::StorageError(format!("failed to read file: {}", e))
            })?;
            serde_json::from_str::<FileStoreData>(&content).map_err(|e| {
                VectorStoreError::StorageError(format!("failed to parse file: {}", e))
            })?
        } else {
            // ensure the parent directory exists
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        VectorStoreError::StorageError(format!("failed to create directory: {}", e))
                    })?;
                }
            }
            let data = FileStoreData {
                dimension,
                documents: HashMap::new(),
            };
            // persist an empty file on first creation so path.exists() is true
            Self::persist(&data, &path).await?;
            data
        };

        Ok(Self {
            path,
            dimension: data.dimension,
            data: RwLock::new(data),
        })
    }

    /// Persists the current data to disk
    async fn persist(data: &FileStoreData, path: &PathBuf) -> Result<(), VectorStoreError> {
        let json = serde_json::to_string(data)
            .map_err(|e| VectorStoreError::StorageError(format!("failed to serialize: {}", e)))?;
        // write to a temporary file first, then rename, to avoid corruption from a mid-write power loss
        let tmp_path = path.with_extension("json.tmp");
        tokio::fs::write(&tmp_path, &json).await.map_err(|e| {
            VectorStoreError::StorageError(format!("failed to write temporary file: {}", e))
        })?;
        tokio::fs::rename(&tmp_path, path)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("failed to rename file: {}", e)))?;
        Ok(())
    }

    /// Returns the vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the storage file path
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[async_trait]
impl VectorStore for FileVectorStore {
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>, VectorStoreError> {
        if documents.len() != embeddings.len() {
            return Err(VectorStoreError::StorageError(
                "document count and embedding count mismatch".to_string(),
            ));
        }

        let mut data = self.data.write().await;
        let mut ids = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            // validate the dimension
            if !embedding.is_empty() && embedding.len() != data.dimension {
                return Err(VectorStoreError::StorageError(format!(
                    "embedding dimension {} does not match storage dimension {}",
                    embedding.len(),
                    data.dimension
                )));
            }

            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            let vector_doc = VectorDocument {
                document: Document {
                    id: Some(id.clone()),
                    content: doc.content,
                    metadata: doc.metadata,
                },
                embedding,
            };
            data.documents.insert(id.clone(), vector_doc);
            ids.push(id);
        }

        Self::persist(&data, &self.path).await?;
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let data = self.data.read().await;

        let mut results: Vec<SearchResult> = data
            .documents
            .values()
            .map(|vd| {
                let score = cosine_similarity(query_embedding, &vd.embedding).unwrap_or(0.0);
                SearchResult {
                    document: vd.document.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results.into_iter().take(k).collect())
    }

    /// S3: file-store metadata filtering — the same "filter first, then compute similarity" semantics as the in-memory store.
    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let data = self.data.read().await;

        let mut results: Vec<SearchResult> = data
            .documents
            .values()
            .filter(|vd| filter.is_none_or(|f| f.matches(&vd.document.metadata)))
            .map(|vd| {
                let score = cosine_similarity(query_embedding, &vd.embedding).unwrap_or(0.0);
                SearchResult {
                    document: vd.document.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results.into_iter().take(k).collect())
    }

    async fn get_document(&self, id: &str) -> Result<Option<Document>, VectorStoreError> {
        let data = self.data.read().await;
        Ok(data.documents.get(id).map(|vd| vd.document.clone()))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let data = self.data.read().await;
        Ok(data.documents.get(id).map(|vd| vd.embedding.clone()))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut data = self.data.write().await;
        data.documents
            .remove(id)
            .ok_or_else(|| VectorStoreError::DocumentNotFound(id.to_string()))?;
        Self::persist(&data, &self.path).await?;
        Ok(())
    }

    async fn count(&self) -> usize {
        let data = self.data.read().await;
        data.documents.len()
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut data = self.data.write().await;
        data.documents.clear();
        Self::persist(&data, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_store_path(dir: &TempDir) -> PathBuf {
        dir.path().join("test_vectors.json")
    }

    #[tokio::test]
    async fn test_new_creates_empty_store() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
        assert_eq!(store.count().await, 0);
        assert_eq!(store.dimension(), 3);
        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_add_and_search() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path, 3).await.unwrap();

        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.count().await, 3);

        let query = vec![0.9, 0.1, 0.0];
        let results = store.similarity_search(&query, 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("Rust"));
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_persistence_across_instances() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);

        // first instance: write
        {
            let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
            let doc = Document::new("persistent doc").with_id("p1");
            store
                .add_documents(vec![doc], vec![vec![1.0, 0.0, 0.0]])
                .await
                .unwrap();
        }

        // second instance: load and verify
        {
            let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
            assert_eq!(store.count().await, 1);
            let doc = store.get_document("p1").await.unwrap().unwrap();
            assert_eq!(doc.content, "persistent doc");
        }
    }

    #[tokio::test]
    async fn test_delete_persists() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);

        {
            let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
            let doc = Document::new("to delete").with_id("d1");
            store
                .add_documents(vec![doc], vec![vec![1.0, 0.0, 0.0]])
                .await
                .unwrap();
            store.delete_document("d1").await.unwrap();
        }

        let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_clear_persists() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);

        {
            let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
            let docs = vec![Document::new("a"), Document::new("b")];
            let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
            store.add_documents(docs, embeddings).await.unwrap();
            store.clear().await.unwrap();
        }

        let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path, 3).await.unwrap();

        let doc = Document::new("wrong dim");
        let wrong_embedding = vec![1.0, 0.0]; // dimension 2, storage dimension 3
        let result = store.add_documents(vec![doc], vec![wrong_embedding]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_embedding() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path, 3).await.unwrap();

        let doc = Document::new("embed test").with_id("e1");
        store
            .add_documents(vec![doc], vec![vec![0.5, 0.5, 0.0]])
            .await
            .unwrap();

        let emb = store.get_embedding("e1").await.unwrap().unwrap();
        assert_eq!(emb, vec![0.5, 0.5, 0.0]);
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path, 3).await.unwrap();

        let result = store.delete_document("no-such-id").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 0.0001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 0.0001);
    }

    /// S3: file-store metadata filtering — single condition + AND combination.
    #[tokio::test]
    async fn test_metadata_filter() {
        use crate::FilterOp;

        let dir = TempDir::new().unwrap();
        let path = test_store_path(&dir);
        let store = FileVectorStore::new(path, 3).await.unwrap();

        store
            .add_documents(
                vec![
                    Document::new("rust doc")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2024),
                    Document::new("python doc")
                        .with_metadata("lang", "python")
                        .with_metadata("year", 2023),
                ],
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];

        let eq = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&eq))
            .await
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].document.content, "rust doc");

        let and = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::field("year", FilterOp::Gt, 2020),
        ]);
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&and))
            .await
            .unwrap();
        assert_eq!(r.len(), 1);

        // filter: None behaves identically to similarity_search
        let none = store
            .similarity_search_with_filter(&query, 5, None)
            .await
            .unwrap();
        let base = store.similarity_search(&query, 5).await.unwrap();
        assert_eq!(none.len(), base.len());
    }
}
