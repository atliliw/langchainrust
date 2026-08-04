// lc-vector-stores/src/file_store.rs
//! 文件持久化向量存储
//!
//! 将文档和向量持久化到本地文件(JSON 序列化),适用于个人知识库和离线场景。
//! 填补 InMemory(不持久)与外部数据库(太重)之间的空缺,类似 SQLite 之于 MySQL。
//!
//! # 使用方式
//! ```ignore
//! use lc_vector_stores::FileVectorStore;
//! use std::path::PathBuf;
//!
//! let store = FileVectorStore::new(PathBuf::from("./my_vectors.json"), 1536).unwrap();
//! // add_documents / similarity_search 与 InMemoryVectorStore 接口一致
//! // 每次增删后自动持久化到磁盘
//! ```

use crate::{
    cosine_similarity, Document, SearchResult, VectorDocument, VectorStore, VectorStoreError,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;
use uuid::Uuid;

/// 持久化数据结构(序列化为 JSON)
#[derive(serde::Serialize, serde::Deserialize)]
struct FileStoreData {
    /// 向量维度(用于校验)
    dimension: usize,
    /// 文档 + 向量
    documents: HashMap<String, VectorDocument>,
}

/// 文件持久化向量存储
///
/// 向量 + 元数据序列化为 JSON 写入磁盘,启动时加载,增删后自动写回。
/// 使用 `RwLock` 保证并发安全:读操作用读锁,写操作用写锁 + 持久化。
pub struct FileVectorStore {
    /// 存储文件路径
    path: PathBuf,
    /// 向量维度
    dimension: usize,
    /// 内存中的数据(与文件同步)
    data: RwLock<FileStoreData>,
}

impl FileVectorStore {
    /// 创建或加载文件向量存储
    ///
    /// 如果文件已存在,加载其中数据;否则创建空存储。
    ///
    /// # Arguments
    /// * `path` - 存储文件路径(建议 `.json` 后缀)
    /// * `dimension` - 向量维度(用于校验,已有文件时以文件为准)
    pub async fn new(path: PathBuf, dimension: usize) -> Result<Self, VectorStoreError> {
        let data = if path.exists() {
            let content = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| VectorStoreError::StorageError(format!("读取文件失败: {}", e)))?;
            serde_json::from_str::<FileStoreData>(&content)
                .map_err(|e| VectorStoreError::StorageError(format!("解析文件失败: {}", e)))?
        } else {
            // 确保父目录存在
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await.map_err(|e| {
                        VectorStoreError::StorageError(format!("创建目录失败: {}", e))
                    })?;
                }
            }
            let data = FileStoreData {
                dimension,
                documents: HashMap::new(),
            };
            // 首次创建时也持久化空文件,确保 path.exists() 为 true
            Self::persist(&data, &path).await?;
            data
        };

        Ok(Self {
            path,
            dimension: data.dimension,
            data: RwLock::new(data),
        })
    }

    /// 持久化当前数据到磁盘
    async fn persist(data: &FileStoreData, path: &PathBuf) -> Result<(), VectorStoreError> {
        let json = serde_json::to_string(data)
            .map_err(|e| VectorStoreError::StorageError(format!("序列化失败: {}", e)))?;
        // 先写临时文件,再 rename,避免写一半断电损坏
        let tmp_path = path.with_extension("json.tmp");
        tokio::fs::write(&tmp_path, &json)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("写入临时文件失败: {}", e)))?;
        tokio::fs::rename(&tmp_path, path)
            .await
            .map_err(|e| VectorStoreError::StorageError(format!("重命名文件失败: {}", e)))?;
        Ok(())
    }

    /// 返回向量维度
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 返回存储文件路径
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
                "文档数量和嵌入向量数量不匹配".to_string(),
            ));
        }

        let mut data = self.data.write().await;
        let mut ids = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            // 校验维度
            if !embedding.is_empty() && embedding.len() != data.dimension {
                return Err(VectorStoreError::StorageError(format!(
                    "嵌入维度 {} 与存储维度 {} 不匹配",
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

        // 第一个实例:写入
        {
            let store = FileVectorStore::new(path.clone(), 3).await.unwrap();
            let doc = Document::new("persistent doc").with_id("p1");
            store
                .add_documents(vec![doc], vec![vec![1.0, 0.0, 0.0]])
                .await
                .unwrap();
        }

        // 第二个实例:加载并验证
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
        let wrong_embedding = vec![1.0, 0.0]; // 维度 2,存储维度 3
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
}
