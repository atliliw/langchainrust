use async_trait::async_trait;
use std::error::Error;

/// 简单的Mock嵌入模型实现
pub struct MockEmbeddingModel {
    dimension: usize,
}

impl MockEmbeddingModel {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// 基于文本的简单哈希嵌入生成
    fn text_to_embedding(&self, text: &str
    ) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // 生成确定性但看似随机的浮点值
        let mut embedding = Vec::with_capacity(self.dimension);
        let mut seed = hash;

        for _ in 0..self.dimension {
            // 使用线性同余生成器生成伪随机数
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let normalized = (seed & 0xFFFFFF) as f32 / 0xFFFFFF as f32;
            embedding.push(normalized * 2.0 - 1.0); // 转换到[-1, 1]范围
        }

        // 标准化向量
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }
}

#[async_trait]
impl super::traits::EmbeddingModel for MockEmbeddingModel {
    async fn embed(&self, text: &str
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(self.text_to_embedding(text))
    }

    async fn embed_batch(
        &self,
        texts: Vec<&str>
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.text_to_embedding(text));
        }
        Ok(embeddings)
    }

    fn embedding_dimension(&self) -> usize {
        self.dimension
    }
}

