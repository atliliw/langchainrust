use crate::retrieval::document::DocumentChunk;
use crate::retrieval::traits::VectorStore;
use async_trait::async_trait;
use std::collections::HashMap;
use std::error::Error;
use uuid::Uuid;

/// 内存中的向量存储实现
pub struct InMemoryVectorStore {
    vectors: HashMap<String, (DocumentChunk, Vec<f32>)>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    /// 计算余弦相似度
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let norm = (norm_a.sqrt() * norm_b.sqrt()).max(f32::EPSILON);
        dot_product / norm
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
        &mut self,
        documents: Vec<(DocumentChunk, Vec<f32>)>,
    ) -> Result<(), Box<dyn Error>> {
        for (chunk, embedding) in documents {
            let id = Uuid::new_v4().to_string();
            self.vectors.insert(id, (chunk, embedding));
        }
        Ok(())
    }

    async fn similarity_search(
        &self,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<(DocumentChunk, f32)>, Box<dyn Error>> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        let mut scored_results: Vec<(DocumentChunk, f32)> = self
            .vectors
            .values()
            .map(|(chunk, embedding)| {
                let score = Self::cosine_similarity(&query, embedding);
                (chunk.clone(), score)
            })
            .collect();

        // 按相似度降序排序
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回前k个结果
        Ok(scored_results.into_iter().take(k).collect())
    }

    async fn similarity_search_with_filter(
        &self,
        query: Vec<f32>,
        k: usize,
        filter: HashMap<String, String>,
    ) -> Result<Vec<(DocumentChunk, f32)>, Box<dyn Error>> {
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        let mut scored_results: Vec<(DocumentChunk, f32)> = self
            .vectors
            .values()
            .filter(|(chunk, _)| {
                // 应用过滤器
                filter
                    .iter()
                    .all(|(key, value)| chunk.metadata.get(key).map_or(false, |v| v == value))
            })
            .map(|(chunk, embedding)| {
                let score = Self::cosine_similarity(&query, embedding);
                (chunk.clone(), score)
            })
            .collect();

        // 按相似度降序排序
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回前k个结果
        Ok(scored_results.into_iter().take(k).collect())
    }

    async fn delete_documents(&mut self, _ids: Vec<String>) -> Result<(), Box<dyn Error>> {
        // 简化实现：由于ID管理复杂，暂时提供空实现
        // 在实际应用中，需要在DocumentChunk中添加ID字段
        Ok(())
    }
}
