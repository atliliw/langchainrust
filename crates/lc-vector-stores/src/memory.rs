// lc-vector-stores/src/memory.rs
//! 内存向量存储
//!
//! 将文档和向量存储在内存中，适用于小规模数据和测试。

use crate::{
    cosine_similarity, Document, MetadataFilter, SearchResult, VectorDocument, VectorStore,
    VectorStoreError,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// 内存向量存储
pub struct InMemoryVectorStore {
    /// 文档存储
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,
}

impl InMemoryVectorStore {
    /// 创建新的内存向量存储
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
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

        let mut store = self.documents.write().await;
        let mut ids = Vec::new();

        for (doc, embedding) in documents.into_iter().zip(embeddings.into_iter()) {
            let id = doc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());

            let vector_doc = VectorDocument {
                document: Document {
                    id: Some(id.clone()),
                    content: doc.content,
                    metadata: doc.metadata,
                },
                embedding,
            };

            store.insert(id.clone(), vector_doc);
            ids.push(id);
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Q2: 不再硬过滤 score > 0 —— 全负分语料下也应返回 top-k;
        // 是否设阈值由调用方通过 similarity_search_with_min_score 显式决定。
        self.similarity_search_with_min_score(query_embedding, k, None)
            .await
    }

    async fn similarity_search_with_min_score(
        &self,
        query_embedding: &[f32],
        k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let store = self.documents.read().await;

        // 计算所有文档的相似度,先按阈值过滤再取 top-k (Q2)
        let mut results: Vec<SearchResult> = store
            .values()
            .filter_map(|vd| {
                let score = cosine_similarity(query_embedding, &vd.embedding).unwrap_or(0.0);
                if min_score.is_none_or(|t| score >= t) {
                    Some(SearchResult {
                        document: vd.document.clone(),
                        score,
                    })
                } else {
                    None
                }
            })
            .collect();

        // 按相似度降序排序
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 返回前 k 个结果
        Ok(results.into_iter().take(k).collect())
    }

    async fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let store = self.documents.read().await;

        // S3: 内存元数据过滤 —— 先按过滤条件筛文档,再算相似度取 top-k。
        let mut results: Vec<SearchResult> = store
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
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.document.clone()))
    }

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, VectorStoreError> {
        let store = self.documents.read().await;
        Ok(store.get(id).map(|vd| vd.embedding.clone()))
    }

    async fn delete_document(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().await;
        store.remove(id);
        Ok(())
    }

    async fn count(&self) -> usize {
        let store = self.documents.read().await;
        store.len()
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut store = self.documents.write().await;
        store.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_search() {
        let store = InMemoryVectorStore::new();

        // 添加文档
        let docs = vec![
            Document::new("Rust is a systems programming language"),
            Document::new("Python is a scripting language"),
            Document::new("JavaScript is used for web development"),
        ];

        // 创建简单的模拟嵌入向量
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // Rust 相关
            vec![0.0, 1.0, 0.0], // Python 相关
            vec![0.0, 0.0, 1.0], // JavaScript 相关
        ];

        let ids = store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.count().await, 3);

        // 搜索相似文档
        let query = vec![0.9, 0.1, 0.0]; // 更接近 Rust
        let results = store.similarity_search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].document.content.contains("Rust"));
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let store = InMemoryVectorStore::new();

        let doc = Document::new("Test document").with_id("test-id");
        let embeddings = vec![vec![1.0, 0.0, 0.0]];

        store.add_documents(vec![doc], embeddings).await.unwrap();

        // 获取文档
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test document");

        // 删除文档
        store.delete_document("test-id").await.unwrap();
        assert_eq!(store.count().await, 0);

        // 再次获取应该返回 None
        let retrieved = store.get_document("test-id").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_clear() {
        let store = InMemoryVectorStore::new();

        let docs = vec![Document::new("Doc 1"), Document::new("Doc 2")];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        store.add_documents(docs, embeddings).await.unwrap();
        assert_eq!(store.count().await, 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await, 0);
    }

    /// Q1: 未配置嵌入器时,similarity_search_text 应显式报 EmbeddingError,
    /// 而不是静默成功或 panic。
    #[tokio::test]
    async fn test_similarity_search_text_without_embedder_errors() {
        let store = InMemoryVectorStore::new();
        let err = store.similarity_search_text("hello", 3).await.unwrap_err();
        assert!(matches!(err, VectorStoreError::EmbeddingError(_)));
    }

    /// Q2: 全非正分语料下 similarity_search 仍返回 top-k(不再被 score>0 硬过滤清空);
    /// similarity_search_with_min_score 按阈值显式过滤。
    #[tokio::test]
    async fn test_negative_scores_not_dropped() {
        let store = InMemoryVectorStore::new();
        store
            .add_documents(
                vec![
                    Document::new("orthogonal-up"),
                    Document::new("opposite"),
                    Document::new("orthogonal-down"),
                ],
                vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]],
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0];

        // 旧实现 score > 0.0 硬过滤,该语料下会返回空;现在返回 top-k(3 条,全部非正分)。
        let results = store.similarity_search(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.score <= 0.0));

        // 显式阈值:score >= -0.5 → 排除 score = -1.0 的那条
        let filtered = store
            .similarity_search_with_min_score(&query, 3, Some(-0.5))
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2);

        // min_score = None 时与 similarity_search 行为一致
        let all = store
            .similarity_search_with_min_score(&query, 3, None)
            .await
            .unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 0.0001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() - 0.0).abs() < 0.0001);
    }

    /// S3: 内存元数据过滤 —— 单条件 + AND/OR 组合;`filter: None` 与旧路径一致。
    #[tokio::test]
    async fn test_metadata_filter() {
        use crate::FilterOp;

        let store = InMemoryVectorStore::new();
        store
            .add_documents(
                vec![
                    Document::new("rust doc")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2024),
                    Document::new("python doc")
                        .with_metadata("lang", "python")
                        .with_metadata("year", 2023),
                    Document::new("rust legacy")
                        .with_metadata("lang", "rust")
                        .with_metadata("year", 2020),
                ],
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.9, 0.1, 0.0],
                ],
            )
            .await
            .unwrap();

        let query = vec![1.0, 0.0, 0.0];

        // 单条件
        let eq = MetadataFilter::field("lang", FilterOp::Eq, "rust");
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&eq))
            .await
            .unwrap();
        assert_eq!(r.len(), 2);
        assert!(r
            .iter()
            .all(|s| s.document.metadata.get("lang").and_then(|v| v.as_str()) == Some("rust")));

        // AND 组合
        let and = MetadataFilter::and(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "rust"),
            MetadataFilter::field("year", FilterOp::Gte, 2021),
        ]);
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&and))
            .await
            .unwrap();
        assert_eq!(r.len(), 1);
        assert!(r[0].document.content.contains("rust doc"));

        // OR 组合
        let or = MetadataFilter::or(vec![
            MetadataFilter::field("lang", FilterOp::Eq, "python"),
            MetadataFilter::field("year", FilterOp::Lt, 2021),
        ]);
        let r = store
            .similarity_search_with_filter(&query, 5, Some(&or))
            .await
            .unwrap();
        assert_eq!(r.len(), 2);

        // filter: None 与 similarity_search 行为一致(回归)
        let none = store
            .similarity_search_with_filter(&query, 5, None)
            .await
            .unwrap();
        let base = store.similarity_search(&query, 5).await.unwrap();
        assert_eq!(none.len(), base.len());
    }
}
