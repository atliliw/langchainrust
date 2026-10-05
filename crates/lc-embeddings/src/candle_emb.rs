// lc-embeddings/src/candle_emb.rs
//! Candle local inference backend (0.21.0 S5.3, feature = `local-candle`).
//!
//! Runs BERT-family embedding models (`bge-*`, Qwen3-Embedding-Mini, ...)
//! through `candle` — pure Rust, no ONNX Runtime. CPU-only here (CUDA / Metal
//! backends are candle capabilities this version does not wire).
//!
//! Quantized inference (v0.26.0 T8.1): weights can be FP16-quantized
//! ([`Quantization::F16`]) or routed through candle's 8-bit GGML backend
//! ([`Quantization::Int8`], `Q8_0`). candle 0.11 ships no quantized BERT module
//! (`QuantizedBertModel`/`QLinear` were removed upstream) and its `BertModel`
//! cannot run a clean FP16 forward on CPU, so both modes quantize the weights
//! then dequantize to FP32 for the standard model build — output stays `Vec<f32>`
//! and near-identical to FP32. The default ([`Quantization::None`]) is unchanged
//! FP32 behavior. (True footprint reduction is deferred until candle restores a
//! native quantized BERT forward; these knobs wire and verify candle 0.11's
//! scaling/`Q8_0` backends today.)
//!
//! Two loading paths, mirroring the ONNX backend's local-file contract:
//! - [`CandleEmbeddings::from_dir`]: local `config.json` + `tokenizer.json`
//!   + `model.safetensors` (offline deployment);
//! - [`CandleEmbeddings::from_hf_hub`]: downloads the three files from a
//!   HuggingFace repo id via `hf-hub` (cache-aware).
//!
//! A deferred [`CandleEmbeddingsBuilder`] (`.with_quantization(..).build()`,
//! matching the sibling providers' `with_*` style) lets callers pick the mode
//! on either path; `from_*_with` are the equivalent one-shot constructors.
//!
//! The `local-candle` backend is an *optional alternative* to fastembed(ort),
//! not a replacement — CI-gated like `local-embeddings`; real-model tests are
//! `#[ignore]`d.

use crate::{EmbeddingError, Embeddings};
use async_trait::async_trait;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Max texts per forward pass (memory bound; pad-aligned batching).
const MAX_BATCH: usize = 16;

/// Candle (pure-Rust) embedding backend for BERT-family models.
pub struct CandleEmbeddings {
    model: Arc<BertModel>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_name: String,
    /// Model's positional embedding limit (`config.json::max_position_embeddings`).
    /// Overlong documents are truncated to this length instead of overflowing
    /// the position table in `forward()`.
    max_len: usize,
}

impl std::fmt::Debug for CandleEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleEmbeddings")
            .field("model", &self.model_name)
            .field("dim", &self.dim)
            .finish()
    }
}

/// Weight precision for the local-candle backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quantization {
    /// FP32 weights (default — matches prior behavior).
    #[default]
    None,
    /// FP16-quantized weights (rounded via half precision, dequantized to FP32).
    F16,
    /// Route weights through candle's 8-bit GGML backend (`Q8_0`) at load.
    Int8,
}

/// Deferred (not-yet-loaded) constructor for the candle backend.
///
/// Mirrors the `with_*` builder style used by the sibling providers
/// (`OpenAIEmbeddings::new(..).with_model(..)`, ...). The model is built on
/// [`CandleEmbeddingsBuilder::build`], so call [`with_quantization`] before it.
///
/// ```
/// # use lc_embeddings::{CandleEmbeddingsBuilder, Quantization};
/// let res = CandleEmbeddingsBuilder::from_dir(std::env::temp_dir())
///     .with_quantization(Quantization::F16)
///     .build();
/// assert!(res.is_err()); // temp_dir() has no model.safetensors — fails fast
/// ```
pub struct CandleEmbeddingsBuilder {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
    model_name: String,
    quantization: Quantization,
}

impl CandleEmbeddingsBuilder {
    /// Build from a local model directory containing `config.json`,
    /// `tokenizer.json` and `model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let model_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("candle-bert")
            .to_string();
        Self {
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
            weights: dir.join("model.safetensors"),
            model_name,
            quantization: Quantization::None,
        }
    }

    /// Downloads `config.json` / `tokenizer.json` / `model.safetensors` from a
    /// HuggingFace repo (blocking, cache-aware).
    pub fn from_hf_hub(repo_id: &str) -> Result<Self, EmbeddingError> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            EmbeddingError::Config(format!("failed to initialize hf-hub client: {e}"))
        })?;
        let repo = api.model(repo_id.to_string());
        let config = repo.get("config.json").map_err(hf_err)?;
        let tokenizer = repo.get("tokenizer.json").map_err(hf_err)?;
        let weights = repo.get("model.safetensors").map_err(hf_err)?;
        Ok(Self {
            config,
            tokenizer,
            weights,
            model_name: repo_id.to_string(),
            quantization: Quantization::None,
        })
    }

    /// Select the weight quantization mode (default [`Quantization::None`]).
    pub fn with_quantization(mut self, quant: Quantization) -> Self {
        self.quantization = quant;
        self
    }

    /// Run the (potentially blocking) load.
    pub fn build(&self) -> Result<CandleEmbeddings, EmbeddingError> {
        CandleEmbeddings::load(
            &self.config,
            &self.tokenizer,
            &self.weights,
            self.model_name.clone(),
            self.quantization,
        )
    }
}

impl CandleEmbeddings {
    /// Loads from a local model directory containing `config.json`,
    /// `tokenizer.json` and `model.safetensors` (FP32 weights).
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        CandleEmbeddingsBuilder::from_dir(model_dir).build()
    }

    /// Loads with an explicit weight mode from a local model directory.
    pub fn from_dir_with(
        model_dir: impl AsRef<Path>,
        quant: Quantization,
    ) -> Result<Self, EmbeddingError> {
        CandleEmbeddingsBuilder::from_dir(model_dir)
            .with_quantization(quant)
            .build()
    }

    /// Downloads `config.json` / `tokenizer.json` / `model.safetensors` from a
    /// HuggingFace repo (blocking, cache-aware) — FP32 weights.
    pub fn from_hf_hub(repo_id: &str) -> Result<Self, EmbeddingError> {
        CandleEmbeddingsBuilder::from_hf_hub(repo_id)?.build()
    }

    /// Like [`from_hf_hub`](CandleEmbeddings::from_hf_hub), with an explicit
    /// weight mode.
    pub fn from_hf_hub_with(
        repo_id: &str,
        quant: Quantization,
    ) -> Result<Self, EmbeddingError> {
        CandleEmbeddingsBuilder::from_hf_hub(repo_id)?
            .with_quantization(quant)
            .build()
    }

    fn load(
        config_path: &Path,
        tokenizer_path: &Path,
        weights_path: &Path,
        model_name: String,
        quant: Quantization,
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
        // Quantization mode → weight pre-pass before the (FP32) model build.
        // `None` is the unchanged FP32 path. `F16` FP16-roundtrips every weight;
        // `Int8` routes them through candle's Q8_0 backend. Both dequantize to
        // FP32 for the standard `BertModel` kernel — candle 0.11 has no native
        // quantized BERT forward (see the helper docs).
        let model = match quant {
            Quantization::None => {
                let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
                BertModel::load(vb, &config)
            }
            Quantization::F16 => {
                let tensors = quantize_f16(tensors)?;
                let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
                BertModel::load(vb, &config)
            }
            Quantization::Int8 => {
                let tensors = quantize_int8(tensors, &device)?;
                let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
                BertModel::load(vb, &config)
            }
        }
        .map_err(|e| EmbeddingError::Config(format!("failed to build BERT model: {e}")))?;

        Ok(Self {
            dim: config.hidden_size,
            model: Arc::new(model),
            tokenizer,
            model_name,
            max_len: config.max_position_embeddings,
        })
    }
}

fn hf_err(e: hf_hub::api::sync::ApiError) -> EmbeddingError {
    EmbeddingError::Config(format!("hf-hub download failed: {e}"))
}

fn tensor_err(e: candle_core::Error) -> EmbeddingError {
    EmbeddingError::ApiError(format!("candle tensor error: {e}"))
}

/// FP16 weight pre-pass for [`Quantization::F16`].
///
/// Casts every weight through `DType::F16` (half precision) and back to FP32.
/// candle 0.11 cannot run the stock `BertModel`'s forward at FP16 on CPU — its
/// `get_extended_attention_mask` broadcasts `f32::MIN` to the model dtype,
/// which overflows to `-inf` in FP16 and makes `0 * -inf = NaN` on every
/// attended position — so the FP16-quantized weights are surfaced as FP32 for
/// the standard kernel. (Mathematically this is exactly an FP16 rounding of
/// the weights: ~11-bit mantissa, near-FP32 output.)
fn quantize_f16(
    tensors: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, EmbeddingError> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        let restored = t.to_dtype(DType::F16).and_then(|h| h.to_dtype(DType::F32));
        let r = match restored {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("failed to FP16-cast weight `{}`: {e}", name);
                return Err(EmbeddingError::Config(msg));
            }
        };
        out.insert(name, r);
    }
    Ok(out)
}

/// Int8 (`Q8_0`) weight pre-pass for [`Quantization::Int8`].
///
/// Runs every weight tensor through candle's 8-bit GGML block quantization
/// (`QTensor::quantize(.., GgmlDType::Q8_0)`) and back (`dequantize`). candle
/// 0.11 ships no quantized BERT module (`QuantizedBertModel`/`QLinear` were
/// removed upstream), so the Q8_0 round-trip is applied at load and the
/// dequantized (approximately-identical, `Q8_0` is lossy) FP32 weights feed the
/// standard `BertModel::load`. Tensors whose last dim is not a multiple of the
/// Q8_0 block size (32), or scalars, are left untouched rather than failing the
/// load.
fn quantize_int8(
    tensors: HashMap<String, Tensor>,
    device: &Device,
) -> Result<HashMap<String, Tensor>, EmbeddingError> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        let restored = QTensor::quantize(&t, GgmlDType::Q8_0)
            .and_then(|q| q.dequantize(device))
            .unwrap_or(t);
        out.insert(name, restored);
    }
    Ok(out)
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
    for (b, out_row) in out.iter_mut().enumerate() {
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
                out_row[d] += hidden[base + d];
            }
        }
        if count > 0 {
            for v in out_row {
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
                max_len: self.max_len,
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
    /// Model positional embedding limit; docs over this are truncated.
    max_len: usize,
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
            .unwrap_or(0)
            // Truncate overlong documents to the model's positional embedding
            // limit rather than letting `max_seq_len > max_position_embeddings`
            // overflow in `forward()`. (Units: tokens; models like bge-small
            // cap at 512.) The row is padded to the capped width below, so
            // bounded inputs are unaffected.
            .min(self.max_len.max(1));
        let pad_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0) as i64;

        let mut input_ids = vec![pad_id; batch * max_len];
        let mut attention = vec![0i64; batch * max_len];
        let token_types = vec![0i64; batch * max_len];
        for (b, enc) in encodings.iter().enumerate() {
            let row = b * max_len;
            for (i, id) in enc.get_ids().iter().take(max_len).enumerate() {
                input_ids[row + i] = *id as i64;
            }
            for (i, m) in enc.get_attention_mask().iter().take(max_len).enumerate() {
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
        // Cast to FP32 before flattening/reading so quantized (FP16) models
        // also return their `Vec<f32>` embedding rows unchanged.
        let hidden_data = hidden
            .flatten_all()
            .map_err(tensor_err)?
            .to_dtype(DType::F32)
            .map_err(tensor_err)?
            .to_vec1::<f32>()
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
    use candle_transformers::models::bert::{HiddenAct, PositionEmbeddingType};

    const TINY_HIDDEN: usize = 16;
    const TINY_INTER: usize = 32;
    const TINY_VOCAB: usize = 40;

    /// Minimal BERT config matching the synthetic weight shapes.
    fn tiny_bert_config() -> Config {
        Config {
            vocab_size: TINY_VOCAB,
            hidden_size: TINY_HIDDEN,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: TINY_INTER,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }

    /// Synthetic FP32 weight set for a 1-layer BERT (F16 path casts on load).
    fn tiny_bert_weights(device: &Device) -> HashMap<String, Tensor> {
        let h = TINY_HIDDEN;
        let mut w = HashMap::new();
        fn add(w: &mut HashMap<String, Tensor>, path: &str, shape: (usize, usize), device: &Device) {
            w.insert(
                path.to_string(),
                Tensor::randn(0f32, 0.01f32, shape, device).unwrap(),
            );
        }
        fn add1(w: &mut HashMap<String, Tensor>, path: &str, n: usize, device: &Device) {
            w.insert(
                path.to_string(),
                Tensor::randn(0f32, 0.01f32, (n,), device).unwrap(),
            );
        }
        // embeddings
        add(&mut w, "embeddings.word_embeddings.weight", (TINY_VOCAB, h), device);
        add(&mut w, "embeddings.position_embeddings.weight", (16, h), device);
        add(&mut w, "embeddings.token_type_embeddings.weight", (2, h), device);
        add1(&mut w, "embeddings.LayerNorm.weight", h, device);
        add1(&mut w, "embeddings.LayerNorm.bias", h, device);
        // single encoder layer
        add(&mut w, "encoder.layer.0.attention.self.query.weight", (h, h), device);
        add1(&mut w, "encoder.layer.0.attention.self.query.bias", h, device);
        add(&mut w, "encoder.layer.0.attention.self.key.weight", (h, h), device);
        add1(&mut w, "encoder.layer.0.attention.self.key.bias", h, device);
        add(&mut w, "encoder.layer.0.attention.self.value.weight", (h, h), device);
        add1(&mut w, "encoder.layer.0.attention.self.value.bias", h, device);
        add(&mut w, "encoder.layer.0.attention.output.dense.weight", (h, h), device);
        add1(&mut w, "encoder.layer.0.attention.output.dense.bias", h, device);
        add1(&mut w, "encoder.layer.0.attention.output.LayerNorm.weight", h, device);
        add1(&mut w, "encoder.layer.0.attention.output.LayerNorm.bias", h, device);
        add(&mut w, "encoder.layer.0.intermediate.dense.weight", (TINY_INTER, h), device);
        add1(&mut w, "encoder.layer.0.intermediate.dense.bias", TINY_INTER, device);
        add(&mut w, "encoder.layer.0.output.dense.weight", (h, TINY_INTER), device);
        add1(&mut w, "encoder.layer.0.output.dense.bias", h, device);
        add1(&mut w, "encoder.layer.0.output.LayerNorm.weight", h, device);
        add1(&mut w, "encoder.layer.0.output.LayerNorm.bias", h, device);
        w
    }

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
        // 形参收 &[Vec<usize>],直接传切片字面量,无需 vec! 套 vec!
        // (stable clippy::useless_vec;该模块 Linux CI 才编译,本机 gnu 无 ort 预构建)。
        let rows = mean_pool_rows(&hidden, 1, 1, 2, &[vec![0]]).unwrap();
        assert_eq!(rows[0], vec![0.0, 0.0]);
        assert!(rows[0].iter().all(|v| v.is_finite()));
    }

    /// Short tensor errors instead of panicking on the index.
    #[test]
    fn mean_pool_short_tensor_errors() {
        let err = mean_pool_rows(&[1.0, 2.0], 1, 2, 2, &[vec![1, 1]]).unwrap_err();
        assert!(matches!(err, EmbeddingError::ParseError(_)));
    }

    /// The quantization knob defaults to disabled (status-quo F32).
    #[test]
    fn quantization_default_is_none() {
        assert_eq!(Quantization::default(), Quantization::None);
        let q = CandleEmbeddingsBuilder::from_dir(std::env::temp_dir()).with_quantization(Quantization::F16);
        assert_eq!(q.quantization, Quantization::F16);
    }

    /// A `BertModel` built from the (dequantized) weight set runs the real
    /// forward on CPU and reads back as a finite `Vec<f32>` — the path both
    /// quantized modes land on (they dequantize to FP32 for the standard model).
    #[test]
    fn model_forward_on_cpu_returns_f32() {
        let device = Device::Cpu;
        let weights = tiny_bert_weights(&device);
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
        let config = tiny_bert_config();
        let model = BertModel::load(vb, &config).unwrap();

        let ids = Tensor::new(&[[1i64, 2i64, 3i64]], &device).unwrap();
        let types = Tensor::new(&[[0i64, 0i64, 0i64]], &device).unwrap();
        let mask = Tensor::new(&[[1i64, 1i64, 1i64]], &device).unwrap();
        let hidden = model.forward(&ids, &types, Some(&mask)).unwrap();

        assert_eq!(hidden.dtype(), DType::F32);
        assert_eq!(hidden.shape().dims(), &[1usize, 3usize, 16usize]);
        let out = hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out.len(), 48);
        assert!(out.iter().all(|v| v.is_finite()), "all outputs finite");
    }

    /// The FP16 weight round-trip (`quantize_f16`) preserves shape and stays
    /// near-identical to the FP32 source (FP16 has ~11 bits of mantissa).
    #[test]
    fn f16_quantize_roundtrip_keeps_shape_and_values() {
        let device = Device::Cpu;
        let src = Tensor::randn(0f32, 0.1f32, (768, 384), &device).unwrap();
        let dq = src
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(dq.shape(), src.shape());
        let a = src.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = dq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-2, "FP16 round-trip max error {max_err}");
    }

    /// candle's int8 (`Q8_0`) round-trip preserves shape and is lossy-but-tight.
    #[test]
    fn q8_quantize_roundtrip_keeps_shape_and_values() {
        let device = Device::Cpu;
        let src = Tensor::randn(0f32, 1f32, (768usize, 384usize), &device).unwrap();
        let q = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
        let dq = q.dequantize(&device).unwrap();
        assert_eq!(dq.shape(), src.shape());
        let a = src.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = dq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let max_err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        // Q8_0 is per-32-element block quantization: lossy, but each element
        // stays within ~2% of its scaled magnitude (N(0,1) → ≤0.02 abs error).
        assert!(max_err < 5e-2, "Q8_0 round-trip max error {max_err}");
    }

    /// The quantized constructors thread the knob and fail fast on missing files.
    #[test]
    fn quantization_knob_fails_fast_on_missing_dir() {
        let dir = std::env::temp_dir().join("lc-candle-quant-nonexistent");
        for quant in [Quantization::F16, Quantization::Int8] {
            let err = CandleEmbeddings::from_dir_with(&dir, quant).unwrap_err();
            assert!(matches!(err, EmbeddingError::Config(_)));
        }
    }

    /// Live F16/Int8 parity against the F32 model (network + download, CI-gated).
    /// Run: `cargo test -p lc-embeddings --features local-candle candle_emb -- --ignored`
    #[tokio::test]
    #[ignore = "downloads a real BERT model from HuggingFace (~120MB) per mode"]
    async fn quantized_modes_match_f32_within_tolerance() {
        let text = "Rust is a systems programming language.";
        let f32v = CandleEmbeddings::from_hf_hub("BAAI/bge-small-en-v1.5")
            .unwrap()
            .embed_query(text)
            .await
            .unwrap();
        let f16 = CandleEmbeddings::from_hf_hub_with("BAAI/bge-small-en-v1.5", Quantization::F16)
            .unwrap()
            .embed_query(text)
            .await
            .unwrap();
        let i8 = CandleEmbeddings::from_hf_hub_with("BAAI/bge-small-en-v1.5", Quantization::Int8)
            .unwrap()
            .embed_query(text)
            .await
            .unwrap();
        for (name, v) in [("f16", &f16), ("int8", &i8)] {
            let dot: f32 = f32v.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            assert!(dot > 0.999, "{name} vs F32 cosine similarity {dot}");
        }
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
