// lc-embeddings/src/fastembed_emb.rs
//! FastEmbed local embeddings — ONNX Runtime with pre-downloaded models.
//!
//! FastEmbed (from Qdrant) provides fast, local embedding generation using
//! ONNX Runtime. No API calls needed — models run entirely on-device.
//!
//! Requires the `fastembed` feature to be enabled.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_embeddings::FastEmbedEmbeddings;
//! use fastembed::EmbeddingModel;
//!
//! let embedder = FastEmbedEmbeddings::with_model(EmbeddingModel::BGESmallENV15)?;
//! let vec = embedder.embed_query("hello world").await?;
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::{EmbeddingError, Embeddings};

/// FastEmbed embedding provider.
///
/// Wraps the `fastembed::TextEmbedding` ONNX Runtime engine.
///
/// ## §3.4 / R-5 — bounded worker pool instead of a single global lock
///
/// `TextEmbedding::embed` needs `&mut self` and is not `Clone`, so parallel inference
/// requires *independent* pre-constructed model instances — reconstructing one per request is
/// not viable (`TextEmbedding::try_new` resolves the model repo, reads the `.onnx` weights and
/// builds an ONNX session, and loads the tokenizer; doing that per call costs more than the
/// lock it replaces). Previously every call shared one `Arc<Mutex<TextEmbedding>>` and
/// serialized, losing all cross-call concurrency.
///
/// Instead this holds a **pool** of pre-constructed workers (`Arc<Mutex<TextEmbedding>>` each),
/// gated by a [`Semaphore`](tokio::sync::Semaphore). Each embedding call reserves one worker
/// for the duration of its blocked inference, so up to `pool_size` independent calls run
/// concurrently while each batch still processes whole under a single lock. The public API is
/// unchanged: the plain constructors keep a default pool of `1` (identical memory footprint to
/// the old single-lock behavior) and [`with_pool_size`](Self::with_pool_size) opts into
/// concurrency — each extra worker costs one more loaded model, so raise it deliberately.
pub struct FastEmbedEmbeddings {
    pool: Arc<ModelPool>,
    model_name: String,
    dimension: usize,
}

/// A bounded set of pre-constructed FastEmbed workers, guarded by a semaphore so at most
/// `pool_size` embeddings are in flight at once and every live call has its own worker.
struct ModelPool {
    /// Bounds concurrent embeddings at `pool_size` (default `DEFAULT_POOL_SIZE`).
    sem: Arc<tokio::sync::Semaphore>,
    /// Idle workers ready to be claimed by an embedding call.
    free: Arc<Mutex<VecDeque<Arc<Mutex<TextEmbedding>>>>>,
}

impl ModelPool {
    fn new(workers: Vec<Arc<Mutex<TextEmbedding>>>) -> Self {
        let size = workers.len();
        let free = Arc::new(Mutex::new(workers.into()));
        Self {
            sem: Arc::new(tokio::sync::Semaphore::new(size)),
            free,
        }
    }

    /// Claim an idle worker. The caller must already hold a semaphore permit, so an idle
    /// worker is guaranteed to exist (concurrency never exceeds `pool_size`).
    fn take_worker(&self) -> Result<PooledModel, EmbeddingError> {
        let mut free = self.free.lock().map_err(|e| {
            EmbeddingError::ApiError(format!("FastEmbed pool lock poisoned: {}", e))
        })?;
        let worker = free.pop_back().ok_or_else(|| {
            EmbeddingError::ApiError("FastEmbed pool exhausted (invariant broken)".to_string())
        })?;
        Ok(PooledModel {
            free: Arc::clone(&self.free),
            worker,
        })
    }
}

/// RAII claim on one pooled worker: returns the worker to the idle queue on drop (even on
/// error / poisoning), so a worker is never leaked out of the pool.
struct PooledModel {
    free: Arc<Mutex<VecDeque<Arc<Mutex<TextEmbedding>>>>>,
    worker: Arc<Mutex<TextEmbedding>>,
}

impl Drop for PooledModel {
    fn drop(&mut self) {
        if let Ok(mut free) = self.free.lock() {
            free.push_back(self.worker.clone());
        }
    }
}

impl std::fmt::Debug for FastEmbedEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedEmbeddings")
            .field("model", &self.model_name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

/// Default worker count for the plain constructors: `1`, identical to the legacy single-lock
/// behavior. Each worker loads its own model session, so raising it multiplies resident memory.
const DEFAULT_POOL_SIZE: usize = 1;

impl FastEmbedEmbeddings {
    /// Creates a new FastEmbedEmbeddings with the given init options.
    pub fn new(options: TextInitOptions) -> Result<Self, EmbeddingError> {
        Self::with_pool_size(DEFAULT_POOL_SIZE, options)
    }

    /// Creates a new FastEmbedEmbeddings with `pool_size` pre-constructed model workers
    /// (R-5). `pool_size` concurrent embedding calls then run in parallel instead of
    /// serializing on one global lock. Each worker is an independent ONNX session + tokenizer,
    /// so memory scales with `pool_size` — start small and measure.
    pub fn with_pool_size(
        pool_size: usize,
        options: TextInitOptions,
    ) -> Result<Self, EmbeddingError> {
        if pool_size == 0 {
            return Err(EmbeddingError::ApiError(
                "FastEmbed pool size must be >= 1".to_string(),
            ));
        }
        // P1-7: model_name 存用户友好名称（如 "BGE-small-en-v1.5"），
        // 而非内部 Debug 标识（如 "BGESmallENV15"）。
        let model_name = Self::friendly_model_name(options.model_name.clone()).to_string();
        // P1-2: 维度未知即报错，不得回落默认 384 撒谎。
        let dimension = Self::infer_dimension(options.model_name.clone())?;

        let mut workers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let model = TextEmbedding::try_new(options.clone()).map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to initialize FastEmbed model: {e}"))
            })?;
            workers.push(Arc::new(Mutex::new(model)));
        }

        Ok(Self {
            pool: Arc::new(ModelPool::new(workers)),
            model_name,
            dimension,
        })
    }

    /// Creates with the default model (BAAI/bge-small-en-v1.5).
    pub fn default_model() -> Result<Self, EmbeddingError> {
        Self::new(
            TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
        )
    }

    /// Creates with a specific model.
    pub fn with_model(model: EmbeddingModel) -> Result<Self, EmbeddingError> {
        Self::new(TextInitOptions::new(model).with_show_download_progress(false))
    }

    /// Infers embedding dimension from the model type.
    ///
    /// P1-2: 维度表穷举已知模型；fastembed 新增未知模型会在编译期以
    /// "non-exhaustive match" 显式暴露，而不是运行时静默回落默认值撒谎。
    fn infer_dimension(model: EmbeddingModel) -> Result<usize, EmbeddingError> {
        let dim = match model {
            EmbeddingModel::BGESmallENV15 | EmbeddingModel::BGESmallENV15Q => 384,
            EmbeddingModel::BGEBaseENV15 | EmbeddingModel::BGEBaseENV15Q => 768,
            EmbeddingModel::BGELargeENV15 | EmbeddingModel::BGELargeENV15Q => 1024,
            EmbeddingModel::AllMiniLML6V2 | EmbeddingModel::AllMiniLML6V2Q => 384,
            EmbeddingModel::AllMiniLML12V2 | EmbeddingModel::AllMiniLML12V2Q => 384,
            EmbeddingModel::AllMpnetBaseV2 => 768,
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::NomicEmbedTextV15 | EmbeddingModel::NomicEmbedTextV15Q => 768,
            EmbeddingModel::MxbaiEmbedLargeV1 | EmbeddingModel::MxbaiEmbedLargeV1Q => 1024,
            EmbeddingModel::GTEBaseENV15 | EmbeddingModel::GTEBaseENV15Q => 768,
            EmbeddingModel::GTELargeENV15 | EmbeddingModel::GTELargeENV15Q => 1024,
            EmbeddingModel::MultilingualE5Small => 384,
            EmbeddingModel::MultilingualE5Base => 768,
            EmbeddingModel::MultilingualE5Large => 1024,
            EmbeddingModel::BGESmallZHV15 => 512,
            EmbeddingModel::BGELargeZHV15 => 1024,
            EmbeddingModel::BGEM3 => 1024,
            EmbeddingModel::ClipVitB32 => 512,
            EmbeddingModel::JinaEmbeddingsV2BaseEN => 768,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 768,
            EmbeddingModel::ParaphraseMLMiniLML12V2 | EmbeddingModel::ParaphraseMLMiniLML12V2Q => {
                384
            }
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,
            EmbeddingModel::ModernBertEmbedLarge => 1024,
            EmbeddingModel::EmbeddingGemma300M
            | EmbeddingModel::EmbeddingGemma300MQ
            | EmbeddingModel::EmbeddingGemma300MQ4 => 768,
            EmbeddingModel::SnowflakeArcticEmbedXS | EmbeddingModel::SnowflakeArcticEmbedXSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedS | EmbeddingModel::SnowflakeArcticEmbedSQ => 384,
            EmbeddingModel::SnowflakeArcticEmbedM | EmbeddingModel::SnowflakeArcticEmbedMQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedMLong
            | EmbeddingModel::SnowflakeArcticEmbedMLongQ => 768,
            EmbeddingModel::SnowflakeArcticEmbedL | EmbeddingModel::SnowflakeArcticEmbedLQ => 1024,
        };
        Ok(dim)
    }

    /// Returns a user-friendly model name (P1-7), e.g. `"BGE-small-en-v1.5"`
    /// instead of the internal Debug identifier `"BGESmallENV15"`.
    fn friendly_model_name(model: EmbeddingModel) -> &'static str {
        match model {
            EmbeddingModel::AllMiniLML6V2 => "all-MiniLM-L6-v2",
            EmbeddingModel::AllMiniLML6V2Q => "all-MiniLM-L6-v2 (quantized)",
            EmbeddingModel::AllMiniLML12V2 => "all-MiniLM-L12-v2",
            EmbeddingModel::AllMiniLML12V2Q => "all-MiniLM-L12-v2 (quantized)",
            EmbeddingModel::AllMpnetBaseV2 => "all-mpnet-base-v2",
            EmbeddingModel::BGEBaseENV15 => "BGE-base-en-v1.5",
            EmbeddingModel::BGEBaseENV15Q => "BGE-base-en-v1.5 (quantized)",
            EmbeddingModel::BGELargeENV15 => "BGE-large-en-v1.5",
            EmbeddingModel::BGELargeENV15Q => "BGE-large-en-v1.5 (quantized)",
            EmbeddingModel::BGESmallENV15 => "BGE-small-en-v1.5",
            EmbeddingModel::BGESmallENV15Q => "BGE-small-en-v1.5 (quantized)",
            EmbeddingModel::NomicEmbedTextV1 => "nomic-embed-text-v1",
            EmbeddingModel::NomicEmbedTextV15 => "nomic-embed-text-v1.5",
            EmbeddingModel::NomicEmbedTextV15Q => "nomic-embed-text-v1.5 (quantized)",
            EmbeddingModel::ParaphraseMLMiniLML12V2 => "paraphrase-multilingual-MiniLM-L12-v2",
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => {
                "paraphrase-multilingual-MiniLM-L12-v2 (quantized)"
            }
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => "paraphrase-multilingual-mpnet-base-v2",
            EmbeddingModel::BGESmallZHV15 => "BGE-small-zh-v1.5",
            EmbeddingModel::BGELargeZHV15 => "BGE-large-zh-v1.5",
            EmbeddingModel::BGEM3 => "BGE-m3",
            EmbeddingModel::ModernBertEmbedLarge => "modernbert-embed-large",
            EmbeddingModel::MultilingualE5Small => "multilingual-e5-small",
            EmbeddingModel::MultilingualE5Base => "multilingual-e5-base",
            EmbeddingModel::MultilingualE5Large => "multilingual-e5-large",
            EmbeddingModel::MxbaiEmbedLargeV1 => "mxbai-embed-large-v1",
            EmbeddingModel::MxbaiEmbedLargeV1Q => "mxbai-embed-large-v1 (quantized)",
            EmbeddingModel::GTEBaseENV15 => "gte-base-en-v1.5",
            EmbeddingModel::GTEBaseENV15Q => "gte-base-en-v1.5 (quantized)",
            EmbeddingModel::GTELargeENV15 => "gte-large-en-v1.5",
            EmbeddingModel::GTELargeENV15Q => "gte-large-en-v1.5 (quantized)",
            EmbeddingModel::ClipVitB32 => "clip-ViT-B-32-text",
            EmbeddingModel::JinaEmbeddingsV2BaseCode => "jina-embeddings-v2-base-code",
            EmbeddingModel::JinaEmbeddingsV2BaseEN => "jina-embeddings-v2-base-en",
            EmbeddingModel::EmbeddingGemma300M => "embeddinggemma-300m",
            EmbeddingModel::EmbeddingGemma300MQ4 => "embeddinggemma-300m (q4)",
            EmbeddingModel::EmbeddingGemma300MQ => "embeddinggemma-300m (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedXS => "snowflake-arctic-embed-xs",
            EmbeddingModel::SnowflakeArcticEmbedXSQ => "snowflake-arctic-embed-xs (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedS => "snowflake-arctic-embed-s",
            EmbeddingModel::SnowflakeArcticEmbedSQ => "snowflake-arctic-embed-s (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedM => "snowflake-arctic-embed-m",
            EmbeddingModel::SnowflakeArcticEmbedMQ => "snowflake-arctic-embed-m (quantized)",
            EmbeddingModel::SnowflakeArcticEmbedMLong => "snowflake-arctic-embed-m-long",
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => {
                "snowflake-arctic-embed-m-long (quantized)"
            }
            EmbeddingModel::SnowflakeArcticEmbedL => "snowflake-arctic-embed-l",
            EmbeddingModel::SnowflakeArcticEmbedLQ => "snowflake-arctic-embed-l (quantized)",
        }
    }
}

#[async_trait]
impl Embeddings for FastEmbedEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let text = text.to_string();
        let pool = Arc::clone(&self.pool);
        // §4 (return-side): the model's dimension comes from the static
        // `infer_dimension` table; the ONNX runtime's actual output width must be
        // checked against it rather than trusted.
        let expected_dim = self.dimension;

        // R-5: reserve a pool permit *before* the blocking task (the semaphore is async, so it
        // cannot be awaited inside `spawn_blocking`). The permit guarantees `take_worker` finds
        // an idle worker and bounds total concurrent embeddings at `pool_size`.
        let permit = Arc::clone(&pool.sem)
            .acquire_owned()
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("FastEmbed pool closed: {}", e)))?;

        let embedding_result =
            tokio::task::spawn_blocking(move || -> Result<Vec<f32>, EmbeddingError> {
                // Claim an idle worker (permit held ⇒ available). Dropping `claim` at the end of
                // the closure returns the worker to the pool even on the `?` error paths below.
                let claim = pool.take_worker()?;
                let mut model = claim.worker.lock().map_err(|e| {
                    EmbeddingError::ApiError(format!("FastEmbed model lock poisoned: {}", e))
                })?;
                let result = model.embed(vec![text.as_str()], None).map_err(|e| {
                    EmbeddingError::ApiError(format!("FastEmbed inference failed: {}", e))
                })?;
                drop(model);

                let mut embedding = result
                    .first()
                    .ok_or_else(|| EmbeddingError::ApiError("No embedding returned".to_string()))?
                    .clone();
                // §3.1: returned width must equal the static-model-table dimension.
                crate::ensure_dimension(&embedding, expected_dim)?;
                // A12: uniform L2 normalization, matching every other backend (P2-8).
                crate::l2_normalize(&mut embedding);
                Ok(embedding)
            })
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?;

        // Return the permit once inference is done, releasing pool capacity.
        drop(permit);
        embedding_result
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // P1-1: 空切片不是错误（无事可做），含空/全空白文本才报错——契约统一。
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let text_vec: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let pool = Arc::clone(&self.pool);
        // §4 (return-side): same width check as embed_query — the static table's
        // dimension is separate from what the ONNX model actually emits.
        let expected_dim = self.dimension;

        // R-5: reserve a pool permit before the blocking task (see `embed_query`).
        let permit = Arc::clone(&pool.sem)
            .acquire_owned()
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("FastEmbed pool closed: {}", e)))?;

        let batch_result = tokio::task::spawn_blocking(move || {
            // Claim an idle worker; returning `claim` at the end rebuilds it into the pool on
            // every path (including `?`).
            let claim = pool.take_worker()?;
            let mut model = claim.worker.lock().map_err(|e| {
                EmbeddingError::ApiError(format!("FastEmbed model lock poisoned: {}", e))
            })?;
            // 批次大小从源 Vec 取:新版 fastembed 的 `embed` 按值消费迭代器
            // (str_vec 被 move),事后再借用 str_vec.len() 是 E0382(all-features
            // 编译失败)。text_vec 仍在作用域内,长度恒等。
            let expected = text_vec.len();
            let str_vec: Vec<&str> = text_vec.iter().map(|s| s.as_str()).collect();
            let result = model.embed(str_vec, None).map_err(|e| {
                EmbeddingError::ApiError(format!("FastEmbed batch inference failed: {}", e))
            })?;
            drop(model);

            // 0.21.0 BUG-2 (P0-1 alignment contract): provider 返回的向量数与
            // 请求批次数不符必须显式报错，禁止静默返回对齐错误的向量。
            // 同仓 ONNX 路径（local/nn）、openai/cohere/openai_compat 均已强制。
            // 注意 `aligned_batch` 本身已返回 Result（spawn_blocking 闭包直接
            // 透传），不要再包 `Ok`——否则三层嵌套导致 fastembed feature 编译失败
            // （docs.rs all-features E0308，0.22.5 修复）。
            let vectors: Vec<Vec<f32>> = result.into_iter().map(|v| v.to_vec()).collect();
            // A12: uniform L2 normalization, matching every other backend (P2-8).
            let normalized: Vec<Vec<f32>> = {
                let mut out = vectors;
                for v in out.iter_mut() {
                    crate::l2_normalize(v);
                }
                out
            };
            // §3.1: returned widths must equal the static-model-table dimension.
            crate::ensure_all_dimensions(&normalized, expected_dim)?;
            Self::aligned_batch(normalized, expected)
        })
        .await
        .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {}", e)))?;

        drop(permit);
        batch_result
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl FastEmbedEmbeddings {
    /// 0.21.0 BUG-2 (P0-1 alignment contract): the provider returned a different
    /// number of vectors than the requested batch → explicit `BatchMismatch`,
    /// never a silently mis-aligned vector list. Pure helper so the contract is
    /// unit-testable without loading a real model (same pattern as `pool_rows`
    /// in the ONNX path, `local/nn/mod.rs`).
    fn aligned_batch(
        result: Vec<Vec<f32>>,
        expected: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if result.len() != expected {
            return Err(EmbeddingError::BatchMismatch {
                expected,
                actual: result.len(),
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(model: EmbeddingModel) -> usize {
        FastEmbedEmbeddings::infer_dimension(model).unwrap()
    }

    #[test]
    fn test_infer_dimension_bge_small() {
        assert_eq!(dim(EmbeddingModel::BGESmallENV15), 384);
    }

    #[test]
    fn test_infer_dimension_bge_base() {
        assert_eq!(dim(EmbeddingModel::BGEBaseENV15), 768);
    }

    #[test]
    fn test_infer_dimension_bge_large() {
        assert_eq!(dim(EmbeddingModel::BGELargeENV15), 1024);
    }

    #[test]
    fn test_infer_dimension_mini_lm() {
        assert_eq!(dim(EmbeddingModel::AllMiniLML6V2), 384);
    }

    #[test]
    fn test_infer_dimension_nomic() {
        assert_eq!(dim(EmbeddingModel::NomicEmbedTextV15), 768);
    }

    #[test]
    fn test_infer_dimension_mxbai() {
        assert_eq!(dim(EmbeddingModel::MxbaiEmbedLargeV1), 1024);
    }

    #[test]
    fn test_infer_dimension_gte_base() {
        assert_eq!(dim(EmbeddingModel::GTEBaseENV15), 768);
    }

    #[test]
    fn test_infer_dimension_multilingual_e5() {
        assert_eq!(dim(EmbeddingModel::MultilingualE5Base), 768);
    }

    #[test]
    fn test_infer_dimension_snowflake_xs() {
        assert_eq!(dim(EmbeddingModel::SnowflakeArcticEmbedXS), 384);
    }

    #[test]
    fn test_infer_dimension_modern_bert() {
        // 权威维度来自 fastembed 5.17.4 ModelInfo：modernbert-embed-large = 1024。
        assert_eq!(dim(EmbeddingModel::ModernBertEmbedLarge), 1024);
    }

    /// P1-7: model_name() 返回用户友好名称而非内部 Debug 标识。
    #[test]
    fn test_friendly_model_name() {
        assert_eq!(
            FastEmbedEmbeddings::friendly_model_name(EmbeddingModel::BGESmallENV15),
            "BGE-small-en-v1.5"
        );
        assert_eq!(
            FastEmbedEmbeddings::friendly_model_name(EmbeddingModel::SnowflakeArcticEmbedMLong),
            "snowflake-arctic-embed-m-long"
        );
    }

    /// R-5: a pool of zero workers is rejected up front, before any model load.
    #[test]
    fn test_with_pool_size_zero_rejected() {
        let err = FastEmbedEmbeddings::with_pool_size(
            0,
            TextInitOptions::new(EmbeddingModel::BGESmallENV15),
        )
        .unwrap_err();
        assert!(
            matches!(err, EmbeddingError::ApiError(_)),
            "pool size 0 must be rejected without loading a model"
        );
    }

    /// 0.21.0 BUG-2 (P0-1 alignment contract): returned row count == requested
    /// batch → pass through unchanged.
    #[test]
    fn test_aligned_batch_ok() {
        let batch = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let aligned = FastEmbedEmbeddings::aligned_batch(batch.clone(), 2).unwrap();
        assert_eq!(aligned, batch);
    }

    /// 0.21.0 BUG-2 (P0-1 alignment contract): model output rows != input batch
    /// → explicit `BatchMismatch` (expected/actual reported), never a silent
    /// mis-aligned vector list.
    #[test]
    fn test_aligned_batch_mismatch() {
        let batch = vec![vec![1.0, 2.0]];
        let err = FastEmbedEmbeddings::aligned_batch(batch, 2).unwrap_err();
        assert!(matches!(
            err,
            EmbeddingError::BatchMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }
}
