// lc-embeddings/src/local.rs
//! Local embedding implementations
//!
//! Contains two implementations:
//! - `BagOfWordsEmbeddings`: Lightweight word-frequency hash embedding (pure Rust, no external deps), always available
//! - `LocalEmbeddings`: ONNX Runtime-based neural network embedding (requires `local-embeddings` feature)
//!
//! `BagOfWordsEmbeddings` is suitable for offline, privacy, zero-cost coarse-grained retrieval;
//! `LocalEmbeddings` is suitable for high-quality semantic embedding scenarios (e.g., BGE/E5 models).

use async_trait::async_trait;

#[cfg(feature = "local-embeddings")]
use std::path::Path;

use crate::{EmbeddingError, Embeddings};

// ---------------------------------------------------------------------------
// BagOfWordsEmbeddings — word-frequency hash + L2 normalization (always available)
// ---------------------------------------------------------------------------

/// Lightweight local embedding (word-frequency hash + L2 normalization)
///
/// Based on word frequency + hashing, no API calls, suitable for offline, privacy, zero-cost coarse-grained retrieval.
///
/// Note: This is a lightweight implementation (bag-of-words hash) with limited semantic quality;
/// for high-quality neural network embeddings (BGE/E5 via `ort`), enable the `local-embeddings` feature
/// and use [`LocalEmbeddings`].
pub struct BagOfWordsEmbeddings {
    dim: usize,
}

impl BagOfWordsEmbeddings {
    /// Create local embedding with specified dimension
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    /// Default dimension 256
    pub fn default_dim() -> Self {
        Self::new(256)
    }

    /// Tokenize: English by non-alphanumeric split (lowercased), Chinese/non-ASCII by single character
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for c in text.chars() {
            if c.is_alphanumeric() {
                if c.is_ascii() {
                    current.push(c.to_ascii_lowercase());
                } else {
                    // Non-ASCII (Chinese etc.) single character as token
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

    /// FNV-1a hash
    fn hash(s: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Compute embedding vector (word-frequency hash + L2 normalization)
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for token in Self::tokenize(text) {
            let idx = (Self::hash(&token) as usize) % self.dim;
            v[idx] += 1.0;
        }
        // L2 normalization
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
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
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
// LocalEmbeddings — ONNX Runtime neural network embedding (requires local-embeddings feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "local-embeddings")]
mod nn {
    use super::*;
    use ort::value::Tensor;
    use std::sync::RwLock;

    /// ONNX Runtime-based local neural network embedding
    ///
    /// Supports any ONNX format embedding model (e.g., BGE/E5), runs inference locally,
    /// no external API calls needed, suitable for privacy-sensitive or offline scenarios.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lc_embeddings::LocalEmbeddings;
    ///
    /// let embedder = LocalEmbeddings::from_file("model.onnx")?;
    /// let vec = embedder.embed_query("hello world").await?;
    /// ```
    pub struct LocalEmbeddings {
        // ort 2.0.0-rc.12's Session::run() requires &mut self,
        // use RwLock to get mutable reference in &self methods while satisfying Send + Sync
        session: RwLock<ort::session::Session>,
        dim: usize,
        model_name: String,
    }

    impl LocalEmbeddings {
        /// Load from ONNX model file
        ///
        /// # Arguments
        /// * `model_path` - Path to ONNX model file
        ///
        /// # Returns
        /// `LocalEmbeddings` instance on success, `EmbeddingError` on failure
        pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
            let path = model_path.as_ref();
            let model_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let session = ort::session::Session::builder()
                .map_err(|e| {
                    EmbeddingError::ApiError(format!("Failed to create ONNX SessionBuilder: {}", e))
                })?
                .commit_from_file(path)
                .map_err(|e| {
                    EmbeddingError::ApiError(format!(
                        "Failed to load ONNX model ({}): {}",
                        path.display(),
                        e
                    ))
                })?;

            // Infer dimension from model output
            let dim = Self::infer_dimension(&session)?;

            Ok(Self {
                session: RwLock::new(session),
                dim,
                model_name,
            })
        }

        /// Infer embedding dimension from ONNX session output info
        fn infer_dimension(session: &ort::session::Session) -> Result<usize, EmbeddingError> {
            let outputs = session.outputs();
            if outputs.is_empty() {
                return Err(EmbeddingError::ParseError(
                    "ONNX model has no output nodes".to_string(),
                ));
            }

            // Get first output's ValueType, infer dimension from shape
            let dtype = outputs[0].dtype();
            let shape = dtype.tensor_shape().ok_or_else(|| {
                EmbeddingError::ParseError("Output is not a Tensor type".to_string())
            })?;

            // shape is typically [-1, seq_len, dim] or [-1, dim]
            // Dynamic dimensions use -1, take the last positive dimension as embedding dimension
            let dim = shape
                .iter()
                .rev()
                .find_map(|&d| if d > 0 { Some(d as usize) } else { None })
                .ok_or_else(|| {
                    EmbeddingError::ParseError(format!(
                        "Cannot infer embedding dimension from model output shape: {:?}",
                        *shape
                    ))
                })?;

            Ok(dim)
        }

        /// Simple whitespace tokenizer
        ///
        /// Splits text by whitespace into token ID sequence.
        /// This is a basic implementation; production use should consider the `tokenizers` crate for subword tokenization.
        fn simple_tokenize(text: &str) -> Vec<i64> {
            // Simple whitespace tokenization, map each token's byte hash to an ID
            text.split_whitespace()
                .map(|word| {
                    let mut h: u64 = 0xcbf29ce484222325;
                    for b in word.bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    // Map to reasonable token ID range (0..30522 similar to BERT vocab size)
                    (h % 30522) as i64
                })
                .collect()
        }

        /// Run ONNX inference, return raw output data
        fn run_inference(
            &self,
            input_ids: &[i64],
        ) -> Result<(Vec<usize>, Vec<f32>), EmbeddingError> {
            let seq_len = input_ids.len();
            if seq_len == 0 {
                return Err(EmbeddingError::EmptyInput);
            }

            // Construct input_ids tensor: shape [1, seq_len]
            let input_shape = vec![1i64, seq_len as i64];
            let input_data = input_ids.to_vec();

            let input_tensor = Tensor::from_array((input_shape, input_data)).map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to construct input tensor: {}", e))
            })?;

            // Get input name
            let session = self.session.read().map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to acquire session read lock: {}", e))
            })?;
            let input_name = session
                .inputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "input_ids".to_string());

            // run() requires &mut self, acquire write lock via RwLock
            drop(session);
            let mut session = self.session.write().map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to acquire session write lock: {}", e))
            })?;
            let outputs = session
                .run(ort::inputs![input_name.as_str() => input_tensor]?)
                .map_err(|e| EmbeddingError::ApiError(format!("ONNX inference failed: {}", e)))?;

            // Get first output
            let output_value = outputs.get(0).ok_or_else(|| {
                EmbeddingError::ParseError("ONNX model has no output".to_string())
            })?;

            // Extract tensor data
            let (shape, data) = output_value.try_extract_tensor::<f32>().map_err(|e| {
                EmbeddingError::ParseError(format!("Failed to extract output tensor: {}", e))
            })?;

            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let data_vec = data.to_vec();

            Ok((shape_vec, data_vec))
        }

        /// Mean pooling: average over sequence dimension
        ///
        /// Input shape: [1, seq_len, dim] -> output: [dim]
        /// or [1, dim] -> output: [dim]
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
                    "Unsupported output dimension count: {}",
                    shape.len()
                ))),
            }
        }

        /// L2 normalization
        fn l2_normalize(vec: &mut [f32]) {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }
        }

        /// Execute full embedding pipeline for a single text: tokenize -> inference -> mean pool -> L2 normalize
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
            // ONNX inference is CPU-intensive, run in blocking thread pool
            let text = text.to_string();
            tokio::task::spawn_blocking(move || self.embed_single(&text))
                .await
                .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?
        }

        async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            // Sequential inference (batch inference requires model support for multi-batch input)
            let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
            tokio::task::spawn_blocking(move || {
                let mut results = Vec::with_capacity(texts.len());
                for text in &texts {
                    results.push(self.embed_single(text)?);
                }
                Ok(results)
            })
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn model_name(&self) -> &str {
            &self.model_name
        }
    }
}

// When local-embeddings feature is enabled, re-export LocalEmbeddings
#[cfg(feature = "local-embeddings")]
pub use nn::LocalEmbeddings;

// ---------------------------------------------------------------------------
// Backward compatibility: LocalEmbeddings without feature points to BagOfWordsEmbeddings
// ---------------------------------------------------------------------------

/// Without the `local-embeddings` feature, `LocalEmbeddings` is a type alias for `BagOfWordsEmbeddings`,
/// maintaining backward compatibility.
///
/// With the `local-embeddings` feature enabled, `LocalEmbeddings` becomes the ONNX Runtime-based neural network implementation.
#[cfg(not(feature = "local-embeddings"))]
pub type LocalEmbeddings = BagOfWordsEmbeddings;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosine_similarity;

    // ---- BagOfWordsEmbeddings tests ----

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

        let sim_similar = cosine_similarity(&base, &similar).unwrap_or(0.0);
        let sim_different = cosine_similarity(&base, &different).unwrap_or(0.0);
        assert!(
            sim_similar > sim_different,
            "Shared words should be more similar: {} vs {}",
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
    async fn test_bow_empty_text_returns_error() {
        let e = BagOfWordsEmbeddings::new(64);
        let result = e.embed_query("").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EmbeddingError::EmptyInput));
    }

    #[tokio::test]
    async fn test_bow_chinese_tokenize() {
        let e = BagOfWordsEmbeddings::new(128);
        let a = e.embed_query("机器学习").await.unwrap();
        let b = e.embed_query("机器学习").await.unwrap();
        assert_eq!(a, b);
        let c = e.embed_query("深度学习").await.unwrap();
        let sim = cosine_similarity(&a, &c).unwrap_or(0.0);
        assert!(
            sim > 0.0,
            "Shared '学习' should have positive similarity: {}",
            sim
        );
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

    // ---- LocalEmbeddings backward compatibility test (without feature, is BagOfWordsEmbeddings alias) ----

    #[tokio::test]
    async fn test_local_embeddings_backward_compat() {
        // Without feature, LocalEmbeddings = BagOfWordsEmbeddings
        let e = LocalEmbeddings::new(64);
        let v = e.embed_query("test backward compat").await.unwrap();
        assert_eq!(v.len(), 64);
        assert_eq!(e.model_name(), "local-bow");
    }

    // ---- ONNX LocalEmbeddings tests (requires local-embeddings feature) ----

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
            // shape [1, 2, 3]: 2 tokens, 3 dimensions
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
            // shape [1, 3]: direct extraction
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
            // Same word should produce same token ID
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
