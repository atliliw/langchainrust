// lc-embeddings/src/candle_emb.rs
//! Candle local inference backend (0.21.0 S5.3, feature = `local-candle`).
//!
//! Runs BERT-family embedding models (`bge-*`, Qwen3-Embedding-Mini, ...)
//! through `candle` — pure Rust, no ONNX Runtime. CPU-only here (quantized /
//! CUDA / Metal backends are candle capabilities this version does not wire).
//!
//! Two loading paths, mirroring the ONNX backend's local-file contract:
//! - [`CandleEmbeddings::from_dir`]: local `config.json` + `tokenizer.json`
//!   + `model.safetensors` (offline deployment);
//! - [`CandleEmbeddings::from_hf_hub`]: downloads the three files from a
//!   HuggingFace repo id via `hf-hub` (cache-aware).
//!
//! The `local-candle` backend is an *optional alternative* to fastembed(ort),
//! not a replacement — CI-gated like `local-embeddings`; real-model tests are
//! `#[ignore]`d.

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use std::sync::Arc;

/// Max texts per forward pass (memory bound; pad-aligned batching).
const MAX_BATCH: usize = 16;

/// Candle (pure-Rust) embedding backend for BERT-family models.
pub struct CandleEmbeddings {
    model: Arc<BertModel>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_name: String,
}

impl std::fmt::Debug for CandleEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbeddings")
            .field("model", &self.model_name)
            .field("dim", &self.dim)
            .finish()
    }
}

impl CandleEmbeddings {
    /// Loads from a local model directory containing `config.json`,
    /// `tokenizer.json` and `model.safetensors`.
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        let dir = model_dir.as_ref();
        let model_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("candle-bert")
            .to_string();
        Self::load(
            &dir.join("config.json"),
            &dir.join("tokenizer.json"),
            &dir.join("model.safetensors"),
            model_name,
        )
    }

    /// Downloads `config.json` / `tokenizer.json` / `model.safetensors` from a
    /// HuggingFace repo (blocking, cache-aware) and loads the model.
    pub fn from_hf_hub(repo_id: &str) -> Result<Self, EmbeddingError> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            EmbeddingError::Config(format!("failed to initialize hf-hub client: {e}"))
        })?;
        let repo = api.model(repo_id.to_string());
        let config = repo.get("config.json").map_err(hf_err)?;
        let tokenizer = repo.get("tokenizer.json").map_err(hf_err)?;
        let weights = repo.get("model.safetensors").map_err(hf_err)?;
        Self::load(&config, &tokenizer, &weights, repo_id.to_string())
    }

    fn load(
        config_path: &Path,
        tokenizer_path: &Path,
        weights_path: &Path,
        model_name: String,
    ) -> Result<Self, EmbeddingError> {
        let config: Config =
            serde_json::from_reader(std::fs::File::open(config_path).map_err(|e| {
                EmbeddingError::Config(format!("failed to open {}: {e}", config_path.display()))
            })?)
            .map_err(|e| {
                EmbeddingError::Config(format!("failed to parse {}: {e}", config_path.display()))
            })?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            EmbeddingError::Config(format!(
                "failed to load tokenizer {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, &device).map_err(|e| {
            EmbeddingError::Config(format!(
                "failed to load weights {}: {e}",
                weights_path.display()
            ))
        })?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbeddingError::Config(format!("failed to build BERT model: {e}")))?;

        Ok(Self {
            dim: config.hidden_size,
            model: Arc::new(model),
            tokenizer,
            model_name,
        })
    }
}

fn hf_err(e: hf_hub::api::sync::ApiError) -> EmbeddingError {
    EmbeddingError::Config(format!("hf-hub download failed: {e}"))
}

fn tensor_err(e: candle_core::Error) -> EmbeddingError {
    EmbeddingError::ApiError(format!("candle tensor error: {e}"))
}

/// Mask-weighted mean pooling over the sequence axis.
///
/// Pure (0.21.0 S5.3): `hidden` is a flattened `[batch, seq, hidden]` tensor;
/// each row averages the positions where `mask == 1` and is returned
/// un-normalized (the caller L2-normalizes uniformly, P2-8). Positions beyond
/// a row's actual length are excluded by the mask, so pad rows never leak in.
pub(crate) fn mean_pool_rows(
    hidden: &[f32],
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    masks: &[Vec<i64>],
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let expected = batch * seq_len * hidden_size;
    if hidden.len() < expected {
        return Err(EmbeddingError::ParseError(format!(
            "hidden tensor too short: need {expected} floats, got {}",
            hidden.len()
        )));
    }
    let mut out = vec![vec![0.0f32; hidden_size]; batch];
    for b in 0..batch {
        let mask = masks
            .get(b)
            .ok_or_else(|| EmbeddingError::ParseError("missing attention mask row".to_string()))?;
        let mut count = 0usize;
        for s in 0..seq_len {
            if mask.get(s).copied().unwrap_or(0) == 0 {
                continue;
            }
            count += 1;
            let base = (b * seq_len + s) * hidden_size;
            for d in 0..hidden_size {
                out[b][d] += hidden[base + d];
            }
        }
        if count > 0 {
            for v in &mut out[b] {
                *v /= count as f32;
            }
        }
    }
    Ok(out)
}

#[async_trait]
impl Embeddings for CandleEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.trim().is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        self.embed_documents(&[text])
            .await
            .map(|mut v| v.pop().unwrap_or_default())
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Pad-aligned chunked batching (same shape as the other providers).
        let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_BATCH) {
            let model = self.model.clone();
            let tokenizer = self.tokenizer.clone();
            let dim = self.dim;
            let chunk: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();

            // Rebuild a blocking embedder view: model + tokenizer are shared
            // Arc/clone-cheap; run on the blocking pool.
            let blocking = BlockingEmbedder {
                model,
                tokenizer,
                dim,
            };
            let rows = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                blocking.embed(&refs)
            })
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {e}")))??;
            all.extend(rows);
        }
        Ok(all)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Arc-shared view of the model for `spawn_blocking` (`'static` capture).
struct BlockingEmbedder {
    model: Arc<BertModel>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
}

impl BlockingEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingError::ApiError(format!("tokenizer failed: {e}")))?;

        let batch = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let pad_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0) as i64;

        let mut input_ids = vec![pad_id; batch * max_len];
        let mut attention = vec![0i64; batch * max_len];
        let token_types = vec![0i64; batch * max_len];
        for (b, enc) in encodings.iter().enumerate() {
            let row = b * max_len;
            for (i, id) in enc.get_ids().iter().enumerate() {
                input_ids[row + i] = *id as i64;
            }
            for (i, m) in enc.get_attention_mask().iter().enumerate() {
                attention[row + i] = *m as i64;
            }
        }

        let device = Device::Cpu;
        let input_ids = Tensor::new(&input_ids[..], &device)
            .and_then(|t| t.reshape((batch, max_len)))
            .map_err(tensor_err)?;
        let token_type_ids = Tensor::new(&token_types[..], &device)
            .and_then(|t| t.reshape((batch, max_len)))
            .map_err(tensor_err)?;
        let attention_mask = Tensor::new(&attention[..], &device)
            .and_then(|t| t.reshape((batch, max_len)))
            .map_err(tensor_err)?;

        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| EmbeddingError::ApiError(format!("candle forward failed: {e}")))?;
        let hidden_data = hidden
            .flatten_all()
            .map_err(tensor_err)?
            .to_vec1()
            .map_err(tensor_err)?;

        let masks: Vec<Vec<i64>> = (0..batch)
            .map(|b| attention[b * max_len..(b + 1) * max_len].to_vec())
            .collect();
        let mut rows = mean_pool_rows(&hidden_data, batch, max_len, self.dim, &masks)?;
        for row in &mut rows {
            crate::l2_normalize(row);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mask-weighted mean: only mask=1 positions average in.
    #[test]
    fn mean_pool_weights_by_mask() {
        // batch 2, seq 2, dim 2; row0 both real, row1 first real second pad.
        let hidden = vec![
            1.0, 1.0, 3.0, 3.0, // row0: [1,1] and [3,3] → mean [2,2]
            5.0, 7.0, 100.0, 100.0, // row1: [5,7] only (second masked) → [5,7]
        ];
        let masks = vec![vec![1, 1], vec![1, 0]];
        let rows = mean_pool_rows(&hidden, 2, 2, 2, &masks).unwrap();
        assert_eq!(rows[0], vec![2.0, 2.0]);
        assert_eq!(rows[1], vec![5.0, 7.0], "pad position excluded");
    }

    /// All-masked rows stay zero (no divide-by-zero / no NaN).
    #[test]
    fn mean_pool_all_masked_row_is_zero() {
        let hidden = vec![1.0, 2.0];
        let rows = mean_pool_rows(&hidden, 1, 1, 2, &vec![vec![0]]).unwrap();
        assert_eq!(rows[0], vec![0.0, 0.0]);
        assert!(rows[0].iter().all(|v| v.is_finite()));
    }

    /// Short tensor errors instead of panicking on the index.
    #[test]
    fn mean_pool_short_tensor_errors() {
        let err = mean_pool_rows(&[1.0, 2.0], 1, 2, 2, &vec![vec![1, 1]]).unwrap_err();
        assert!(matches!(err, EmbeddingError::ParseError(_)));
    }

    /// Config JSON parses into the candle `Config` (real struct, real serde).
    #[test]
    fn config_json_parses() {
        let json = r#"{
            "vocab_size": 30522, "hidden_size": 384, "num_hidden_layers": 6,
            "num_attention_heads": 12, "intermediate_size": 1536,
            "hidden_act": "gelu", "hidden_dropout_prob": 0.1,
            "max_position_embeddings": 512, "type_vocab_size": 2,
            "initializer_range": 0.02, "layer_norm_eps": 1e-12,
            "pad_token_id": 0, "classifier_dropout": null, "model_type": "bert"
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_hidden_layers, 6);
    }

    /// Construction fails fast with a clear Config error when files are missing.
    #[test]
    fn missing_files_fail_fast() {
        let dir = std::env::temp_dir().join("lc-candle-nonexistent-dir");
        let err = CandleEmbeddings::from_dir(&dir).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
    }

    /// Live end-to-end against a real HF model (network + ~120MB download).
    /// Run with: `cargo test -p lc-embeddings --features local-candle
    ///   candle_emb -- --ignored`
    #[tokio::test]
    #[ignore = "downloads a real BERT model from HuggingFace (~120MB)"]
    async fn real_model_end_to_end() {
        let embedder =
            CandleEmbeddings::from_hf_hub("BAAI/bge-small-en-v1.5").expect("model loads");
        assert_eq!(embedder.dimension(), 384);

        let vec = embedder
            .embed_query("Rust is a systems programming language.")
            .await
            .unwrap();
        assert_eq!(vec.len(), 384);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "L2-normalized, got {norm}");

        // Same text → identical vector; different text → different vector.
        let again = embedder
            .embed_query("Rust is a systems programming language.")
            .await
            .unwrap();
        assert_eq!(vec, again);
        let other = embedder.embed_query("banana pancake recipe").await.unwrap();
        let dot: f32 = vec.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        assert!(dot < 0.99, "distinct texts should not be near-identical");

        // Batch path returns aligned vectors.
        let batch = embedder
            .embed_documents(&["first doc", "second doc", "third one"])
            .await
            .unwrap();
        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|v| v.len() == 384));
    }
}
