// src/embeddings/local.rs
//! 本地嵌入实现
//!
//! 包含两种实现:
//! - `BagOfWordsEmbeddings`: 轻量词频 hash 嵌入(纯 Rust, 无外部依赖), 始终可用
//! - `LocalEmbeddings`: 基于 ONNX Runtime 的神经网络嵌入(需 `local-embeddings` feature)
//!
//! `BagOfWordsEmbeddings` 适合离线、隐私、零成本场景的粗粒度检索;
//! `LocalEmbeddings` 适合需要高质量语义嵌入的场景(如 BGE/E5 模型).

use async_trait::async_trait;

#[cfg(feature = "local-embeddings")]
use std::path::Path;

use super::{Embeddings, EmbeddingError};

// ---------------------------------------------------------------------------
// BagOfWordsEmbeddings — 词频 hash + L2 归一化(始终可用)
// ---------------------------------------------------------------------------

/// 轻量本地嵌入(词频 hash + L2 归一化)
///
/// 基于词频 + 哈希的本地嵌入, 不调用任何 API, 适合离线、隐私、零成本场景的粗粒度检索。
///
/// 注: 这是轻量实现(词袋 hash), 语义质量有限; 若需高质量神经网络嵌入
/// (BGE/E5 via `ort`), 请启用 `local-embeddings` feature 并使用 [`LocalEmbeddings`]。
pub struct BagOfWordsEmbeddings {
    dim: usize,
}

impl BagOfWordsEmbeddings {
    /// 创建指定维度的本地嵌入
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(1),
        }
    }

    /// 默认维度 256
    pub fn default_dim() -> Self {
        Self::new(256)
    }

    /// 分词: 英文按非字母数字切分(小写化), 中文/非 ASCII 按单字
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                if c.is_ascii() {
                    current.push(c.to_ascii_lowercase());
                } else {
                    // 非ASCII(中文等)单字成 token
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                    tokens.push(c.to_string());
                }
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// FNV-1a 哈希
    fn hash(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// 计算嵌入向量(词频 hash + L2 归一化)
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in Self::tokenize(text) {
            let idx = (Self::hash(&token) as usize) % self.dim;
            v[idx] += 1.0;
        }
        // L2 归一化
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for BagOfWordsEmbeddings {
    fn default() -> Self {
        Self::default_dim()
    }
}

#[async_trait]
impl Embeddings for BagOfWordsEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.embed(text))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "local-bow"
    }
}

// ---------------------------------------------------------------------------
// LocalEmbeddings — ONNX Runtime 神经网络嵌入(需 local-embeddings feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "local-embeddings")]
mod nn {
    use super::*;
    use ort::value::Tensor;
    use std::cell::RefCell;

    /// 基于 ONNX Runtime 的本地神经网络嵌入
    ///
    /// 支持任何 ONNX 格式的嵌入模型(如 BGE/E5), 在本地运行推理,
    /// 无需调用外部 API, 适合隐私敏感或离线场景。
    ///
    /// # Example
    ///
    /// ```ignore
    /// use langchainrust::embeddings::LocalEmbeddings;
    ///
    /// let embedder = LocalEmbeddings::from_file("model.onnx")?;
    /// let vec = embedder.embed_query("hello world").await?;
    /// ```
    pub struct LocalEmbeddings {
        // ort 2.0.0-rc.12 的 Session::run() 需要 &mut self,
        // 使用 RefCell 以便在 &self 方法中获取可变引用
        session: RefCell<ort::session::Session>,
        dim: usize,
        model_name: String,
    }

    impl LocalEmbeddings {
        /// 从 ONNX 模型文件加载
        ///
        /// # 参数
        /// * `model_path` - ONNX 模型文件路径
        ///
        /// # 返回
        /// 加载成功返回 `LocalEmbeddings` 实例, 失败返回 `EmbeddingError`
        pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
            let path = model_path.as_ref();
            let model_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let session = ort::session::Session::builder()
                .map_err(|e| {
                    EmbeddingError::ApiError(format!("创建 ONNX SessionBuilder 失败: {}", e))
                })?
                .commit_from_file(path)
                .map_err(|e| {
                    EmbeddingError::ApiError(format!(
                        "加载 ONNX 模型失败 ({}): {}",
                        path.display(),
                        e
                    ))
                })?;

            // 从模型输出推断维度
            let dim = Self::infer_dimension(&session)?;

            Ok(Self {
                session: RefCell::new(session),
                dim,
                model_name,
            })
        }

        /// 从 ONNX session 的输出信息推断嵌入维度
        fn infer_dimension(session: &ort::session::Session) -> Result<usize, EmbeddingError> {
            let outputs = session.outputs();
            if outputs.is_empty() {
                return Err(EmbeddingError::ParseError(
                    "ONNX 模型没有输出节点".to_string(),
                ));
            }

            // 取第一个输出的 ValueType, 从 shape 中推断维度
            let dtype = outputs[0].dtype();
            let shape = dtype
                .tensor_shape()
                .ok_or_else(|| EmbeddingError::ParseError("输出不是 Tensor 类型".to_string()))?;

            // shape 通常是 [-1, seq_len, dim] 或 [-1, dim]
            // 动态维度用 -1 表示, 取最后一个确定的(正数)维度作为嵌入维度
            let dim = shape
                .iter()
                .rev()
                .find_map(|&d| if d > 0 { Some(d as usize) } else { None })
                .ok_or_else(|| {
                    EmbeddingError::ParseError(format!(
                        "无法从模型输出形状推断嵌入维度: {:?}",
                        *shape
                    ))
                })?;

            Ok(dim)
        }

        /// 简单空白分词器
        ///
        /// 将文本按空白字符分割为 token ID 序列。
        /// 这是一个基础实现; 生产环境建议使用 `tokenizers` crate 进行子词分词。
        fn simple_tokenize(text: &str) -> Vec<i64> {
            // 简单按空白分词, 将每个 token 的字节哈希映射为 ID
            text.split_whitespace()
                .map(|word| {
                    let mut h: u64 = 0xcbf29ce484222325;
                    for b in word.bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    // 映射到合理的 token ID 范围 (0..30522 类 BERT 词表大小)
                    (h % 30522) as i64
                })
                .collect()
        }

        /// 运行 ONNX 推理, 返回原始输出数据
        fn run_inference(
            &self,
            input_ids: &[i64],
        ) -> Result<(Vec<usize>, Vec<f32>), EmbeddingError> {
            let seq_len = input_ids.len();
            if seq_len == 0 {
                return Err(EmbeddingError::EmptyInput);
            }

            // 构造 input_ids 张量: shape [1, seq_len]
            let input_shape = vec![1i64, seq_len as i64];
            let input_data = input_ids.to_vec();

            let input_tensor =
                Tensor::from_array((input_shape, input_data)).map_err(|e| {
                    EmbeddingError::ApiError(format!("构造输入张量失败: {}", e))
                })?;

            // 获取输入名称
            let session = self.session.borrow();
            let input_name = session
                .inputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "input_ids".to_string());

            // run() 需要 &mut self, 通过 RefCell 获取可变引用
            drop(session);
            let mut session = self.session.borrow_mut();
            let outputs = session
                .run(ort::inputs![input_name.as_str() => input_tensor]?)
                .map_err(|e| EmbeddingError::ApiError(format!("ONNX 推理失败: {}", e)))?;

            // 取第一个输出
            let output_value = outputs
                .get(0)
                .ok_or_else(|| EmbeddingError::ParseError("ONNX 模型无输出".to_string()))?;

            // 提取张量数据
            let (shape, data) = output_value
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::ParseError(format!("提取输出张量失败: {}", e)))?;

            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let data_vec = data.to_vec();

            Ok((shape_vec, data_vec))
        }

        /// Mean pooling: 对序列维度取平均
        ///
        /// 输入形状: [1, seq_len, dim] -> 输出: [dim]
        /// 或 [1, dim] -> 输出: [dim]
        fn mean_pool(shape: &[usize], data: &[f32]) -> Result<Vec<f32>, EmbeddingError> {
            match shape.len() {
                3 => {
                    let dim = shape[2];
                    let seq_len = shape[1];
                    let mut result = vec![0.0f32; dim];

                    for s in 0..seq_len {
                        for d in 0..dim {
                            result[d] += data[s * dim + d];
                        }
                    }

                    for v in &mut result {
                        *v /= seq_len as f32;
                    }

                    Ok(result)
                }
                2 => {
                    let dim = shape[1];
                    Ok(data[..dim].to_vec())
                }
                _ => Err(EmbeddingError::ParseError(format!(
                    "不支持的输出维度数: {}",
                    shape.len()
                ))),
            }
        }

        /// L2 归一化
        fn l2_normalize(vec: &mut [f32]) {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }
        }

        /// 对单个文本执行完整的嵌入流程: 分词 -> 推理 -> mean pool -> L2 归一化
        fn embed_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if text.trim().is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }

            let input_ids = Self::simple_tokenize(text);
            if input_ids.is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }

            let (shape, raw_data) = self.run_inference(&input_ids)?;
            let mut pooled = Self::mean_pool(&shape, &raw_data)?;
            Self::l2_normalize(&mut pooled);
            Ok(pooled)
        }
    }

    #[async_trait]
    impl Embeddings for LocalEmbeddings {
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            // ONNX 推理是 CPU 密集型, 在阻塞线程池中执行
            let text = text.to_string();
            tokio::task::spawn_blocking(move || self.embed_single(&text))
                .await
                .map_err(|e| EmbeddingError::ApiError(format!("任务执行失败: {}", e)))?
        }

        async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            // 逐个推理(批量推理需要模型支持多 batch 输入)
            let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
            tokio::task::spawn_blocking(move || {
                let mut results = Vec::with_capacity(texts.len());
                for text in &texts {
                    results.push(self.embed_single(text)?);
                }
                Ok(results)
            })
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("任务执行失败: {}", e)))?
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }
    }
}

// 当 local-embeddings feature 启用时, 重新导出 LocalEmbeddings
#[cfg(feature = "local-embeddings")]
pub use nn::LocalEmbeddings;

// ---------------------------------------------------------------------------
// 向后兼容: LocalEmbeddings 在无 feature 时指向 BagOfWordsEmbeddings
// ---------------------------------------------------------------------------

/// 无 `local-embeddings` feature 时, `LocalEmbeddings` 是 `BagOfWordsEmbeddings` 的类型别名,
/// 保持向后兼容。
///
/// 启用 `local-embeddings` feature 后, `LocalEmbeddings` 变为基于 ONNX Runtime 的神经网络实现。
#[cfg(not(feature = "local-embeddings"))]
pub type LocalEmbeddings = BagOfWordsEmbeddings;

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::cosine_similarity;

    // ---- BagOfWordsEmbeddings 测试 ----

    #[tokio::test]
    async fn test_bow_dimension() {
        let e = BagOfWordsEmbeddings::new(128);
        let v = e.embed_query("hello world").await.unwrap();
        assert_eq!(v.len(), 128);
        assert_eq!(e.dimension(), 128);
    }

    #[tokio::test]
    async fn test_bow_same_text_same_vector() {
        let e = BagOfWordsEmbeddings::new(64);
        let a = e.embed_query("rust programming").await.unwrap();
        let b = e.embed_query("rust programming").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn test_bow_different_text_different_vector() {
        let e = BagOfWordsEmbeddings::new(64);
        let a = e.embed_query("rust programming").await.unwrap();
        let b = e.embed_query("cooking recipe pasta").await.unwrap();
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn test_bow_shared_words_more_similar() {
        let e = BagOfWordsEmbeddings::new(256);
        let base = e.embed_query("rust programming language").await.unwrap();
        let similar = e.embed_query("rust programming tutorial").await.unwrap();
        let different = e.embed_query("cooking pasta recipe").await.unwrap();

        let sim_similar = cosine_similarity(&base, &similar);
        let sim_different = cosine_similarity(&base, &different);
        assert!(
            sim_similar > sim_different,
            "共享词应更相似: {} vs {}",
            sim_similar,
            sim_different
        );
    }

    #[tokio::test]
    async fn test_bow_normalized() {
        let e = BagOfWordsEmbeddings::new(64);
        let v = e.embed_query("some text here").await.unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
    }

    #[tokio::test]
    async fn test_bow_empty_text_zero_vector() {
        let e = BagOfWordsEmbeddings::new(64);
        let v = e.embed_query("").await.unwrap();
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[tokio::test]
    async fn test_bow_chinese_tokenize() {
        let e = BagOfWordsEmbeddings::new(128);
        let a = e.embed_query("机器学习").await.unwrap();
        let b = e.embed_query("机器学习").await.unwrap();
        assert_eq!(a, b);
        let c = e.embed_query("深度学习").await.unwrap();
        let sim = cosine_similarity(&a, &c);
        assert!(sim > 0.0, "共享\"学习\"应有正相似度: {}", sim);
    }

    #[test]
    fn test_bow_tokenize_english() {
        let t = BagOfWordsEmbeddings::tokenize("Hello, World! 123");
        assert!(t.contains(&"hello".to_string()));
        assert!(t.contains(&"world".to_string()));
        assert!(t.contains(&"123".to_string()));
    }

    #[test]
    fn test_bow_tokenize_chinese() {
        let t = BagOfWordsEmbeddings::tokenize("机器学习");
        assert!(t.contains(&"机".to_string()));
        assert!(t.contains(&"学".to_string()));
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn test_bow_model_name() {
        let e = BagOfWordsEmbeddings::default_dim();
        assert_eq!(e.model_name(), "local-bow");
    }

    // ---- LocalEmbeddings 向后兼容测试 (无 feature 时为 BagOfWordsEmbeddings 别名) ----

    #[tokio::test]
    async fn test_local_embeddings_backward_compat() {
        // 无 feature 时 LocalEmbeddings = BagOfWordsEmbeddings
        let e = LocalEmbeddings::new(64);
        let v = e.embed_query("test backward compat").await.unwrap();
        assert_eq!(v.len(), 64);
        assert_eq!(e.model_name(), "local-bow");
    }

    // ---- ONNX LocalEmbeddings 测试 (需 local-embeddings feature) ----

    #[cfg(feature = "local-embeddings")]
    mod nn_tests {
        use super::*;

        #[test]
        fn test_l2_normalize() {
            let mut v = vec![3.0, 4.0];
            LocalEmbeddings::l2_normalize(&mut v);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
            assert!((v[0] - 0.6).abs() < 1e-5);
            assert!((v[1] - 0.8).abs() < 1e-5);
        }

        #[test]
        fn test_l2_normalize_zero() {
            let mut v = vec![0.0, 0.0, 0.0];
            LocalEmbeddings::l2_normalize(&mut v);
            assert!(v.iter().all(|x| *x == 0.0));
        }

        #[test]
        fn test_mean_pool_3d() {
            // shape [1, 2, 3]: 2 个 token, 3 维
            let shape = vec![1usize, 2, 3];
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let result = LocalEmbeddings::mean_pool(&shape, &data).unwrap();
            assert_eq!(result.len(), 3);
            assert!((result[0] - 2.5).abs() < 1e-5);
            assert!((result[1] - 3.5).abs() < 1e-5);
            assert!((result[2] - 4.5).abs() < 1e-5);
        }

        #[test]
        fn test_mean_pool_2d() {
            // shape [1, 3]: 直接提取
            let shape = vec![1usize, 3];
            let data = vec![1.0, 2.0, 3.0];
            let result = LocalEmbeddings::mean_pool(&shape, &data).unwrap();
            assert_eq!(result.len(), 3);
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!((result[1] - 2.0).abs() < 1e-5);
            assert!((result[2] - 3.0).abs() < 1e-5);
        }

        #[test]
        fn test_simple_tokenize() {
            let tokens = LocalEmbeddings::simple_tokenize("hello world test");
            assert_eq!(tokens.len(), 3);
            // 相同词应产生相同 token ID
            let tokens2 = LocalEmbeddings::simple_tokenize("hello");
            assert_eq!(tokens[0], tokens2[0]);
        }

        #[test]
        fn test_simple_tokenize_empty() {
            let tokens = LocalEmbeddings::simple_tokenize("");
            assert!(tokens.is_empty());
        }
    }
}
