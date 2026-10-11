// lc-embeddings/src/candle_emb.rs
//! Candle local inference backend (0.21.0 S5.3, feature = `local-candle`).
//!
//! Runs BERT-family embedding models (`bge-*`, Qwen3-Embedding-Mini, ...)
//! through `candle` — pure Rust, no ONNX Runtime. CPU-only here (CUDA / Metal
//! backends are candle capabilities this version does not wire).
//!
//! Quantized inference (v0.26.0 T8.1; **FP8 added v0.27.0 T3**): weights can be
//! FP16-quantized ([`Quantization::F16`]), routed through candle's 8-bit GGML
//! backend ([`Quantization::Int8`], `Q8_0`), or rounded through candle's native
//! 8-bit float dtype ([`Quantization::Fp8`], `DType::F8E4M3`). candle 0.11 ships
//! no quantized BERT module (`QuantizedBertModel`/`QLinear` were removed
//! upstream) and its `BertModel` cannot run a clean FP16 forward on CPU, so all
//! three quantize the weights then dequantize to FP32 for the standard model
//! build — output stays `Vec<f32>` and near-identical to FP32. The default
//! ([`Quantization::None`]) is unchanged FP32 behavior. (True footprint
//! reduction is deferred until candle restores a native quantized BERT forward;
//! these knobs wire and verify candle 0.11's scaling/`Q8_0`/`F8E4M3` backends
//! today.)
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

use crate::catalog::{ModelCatalog, ModelFetcher, ModelSource, ModelSpec};
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
    /// FP8-quantized weights (candle's `DType::F8E4M3`: 8-bit float, 4-bit
    /// exponent / 3-bit mantissa, saturating at ±448), dequantized to FP32.
    Fp8,
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
    pub fn from_hf_hub_with(repo_id: &str, quant: Quantization) -> Result<Self, EmbeddingError> {
        CandleEmbeddingsBuilder::from_hf_hub(repo_id)?
            .with_quantization(quant)
            .build()
    }

    /// Loads the model registered under `name` in a [`ModelCatalog`].
    ///
    /// This is the catalog's load-side hook: the entry supplies the directory,
    /// the load is the same `from_dir` path. A name that is not registered
    /// ([`EmbeddingError::ModelNotFound`]), an entry whose files are missing
    /// ([`EmbeddingError::ModelNotReady`]), an entry using the ONNX layout, or a
    /// recorded checksum that no longer matches the weights file are all
    /// explicit errors — there is no fallback model.
    pub fn from_catalog(catalog: &ModelCatalog, name: &str) -> Result<Self, EmbeddingError> {
        Self::from_catalog_with(catalog, name, Quantization::None)
    }

    /// Like [`from_catalog`](CandleEmbeddings::from_catalog), with an explicit
    /// weight mode.
    pub fn from_catalog_with(
        catalog: &ModelCatalog,
        name: &str,
        quant: Quantization,
    ) -> Result<Self, EmbeddingError> {
        let entry = catalog.resolve_ready(name)?;
        let config_path = entry.config_path().ok_or_else(|| {
            EmbeddingError::Config(format!(
                "model '{}' is registered with the {:?} layout; the candle backend needs \
                 config.json + model.safetensors + tokenizer.json",
                entry.name, entry.layout
            ))
        })?;
        // Integrity gate: only pays for the read when the entry recorded one.
        if entry.verify_checksum()? == Some(false) {
            return Err(EmbeddingError::Config(format!(
                "model '{}' at '{}' does not match the checksum recorded in the catalog \
                 (re-register it after replacing the weights)",
                entry.name,
                entry.dir.display()
            )));
        }
        Self::load(
            &config_path,
            &entry.tokenizer_path(),
            &entry.model_path(),
            entry.name.clone(),
            quant,
        )
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
            Quantization::Fp8 => {
                let tensors = quantize_fp8(tensors)?;
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

/// `hf-hub`-backed [`ModelFetcher`] for [`ModelCatalog::prefetch`].
///
/// Downloads the files of the spec's layout from its HuggingFace repo into
/// `base_dir/<name>` — the real pre-download path for a deployment that can
/// reach the network. Air-gapped hosts use
/// [`crate::catalog::LocalOnlyFetcher`] over directories that were staged in
/// by a provisioning step.
pub struct HfHubFetcher {
    base_dir: PathBuf,
}

impl HfHubFetcher {
    /// Downloads into `<base_dir>/<spec name>`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }
}

impl ModelFetcher for HfHubFetcher {
    fn fetch(&self, spec: &ModelSpec) -> Result<PathBuf, EmbeddingError> {
        let repo_id = match &spec.source {
            ModelSource::HuggingFace { repo_id } => repo_id.clone(),
            ModelSource::Local => {
                return Err(EmbeddingError::Config(format!(
                    "HfHubFetcher needs a HuggingFace source, but '{}' is declared as local",
                    spec.name
                )));
            }
        };
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            EmbeddingError::Config(format!("failed to initialize hf-hub client: {e}"))
        })?;
        let repo = api.model(repo_id);
        let dir = self.base_dir.join(&spec.name);
        std::fs::create_dir_all(&dir).map_err(|e| {
            EmbeddingError::Config(format!(
                "failed to create model directory '{}': {e}",
                dir.display()
            ))
        })?;
        for file in spec.layout.required_files() {
            let downloaded = repo.get(*file).map_err(hf_err)?;
            let target = dir.join(file);
            std::fs::copy(&downloaded, &target).map_err(|e| {
                EmbeddingError::Config(format!(
                    "failed to copy '{}' to '{}': {e}",
                    downloaded.display(),
                    target.display()
                ))
            })?;
        }
        Ok(dir)
    }
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

/// FP8 weight pre-pass for [`Quantization::Fp8`].
///
/// Rounds every weight through candle's 8-bit float dtype
/// (`DType::F8E4M3` — 4-bit exponent, 3-bit mantissa, largest magnitude 448)
/// and back to FP32: the same load-time shape as the FP16 and `Q8_0`
/// pre-passes, because candle 0.11's CPU backend implements the
/// `F32 ↔ F8E4M3` conversions (`candle-core-0.11.0/src/cpu_backend/mod.rs`,
/// the `(F32, F8E4M3)` and `(F8E4M3, F32)` arms) but ships no FP8 compute
/// kernel for `BertModel::forward`. The rounded weights therefore feed the
/// standard FP32 model and the output stays a near-identical `Vec<f32>`
/// (FP8 keeps ~3 mantissa bits, so this is a coarser rounding than FP16 —
/// use it to model FP8 weight error, not to shrink memory).
///
/// `float8`'s conversion (which candle calls) saturates with
/// `SaturationType::SatFinite`, so an out-of-range weight clamps to ±448
/// rather than becoming NaN/Inf.
fn quantize_fp8(
    tensors: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, EmbeddingError> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in tensors {
        let restored = t.to_dtype(DType::F8E4M3).and_then(|h| h.to_dtype(DType::F32));
        let r = match restored {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("failed to FP8-cast weight `{}`: {e}", name);
                return Err(EmbeddingError::Config(msg));
            }
        };
        out.insert(name, r);
    }
    Ok(out)
}

/// Mask-weighted mean pooling over the sequence axis, from a **flat** mask.
///
/// Pure (0.21.0 S5.3): `hidden` is a flattened `[batch, seq, hidden]` tensor;
/// each row averages the positions where the mask is non-zero and is returned
/// un-normalized (the caller L2-normalizes uniformly, P2-8). Positions beyond
/// a row's actual length are excluded by the mask, so pad rows never leak in.
///
/// `masks` is the row-major `[batch, seq_len]` attention mask (`row b` lives at
/// `masks[b * seq_len .. (b + 1) * seq_len]`). v0.27.0 T3 zero-copy: the caller
/// already holds the padded mask as one `Vec<i64>`, so pooling reads it
/// directly instead of first copying each row into its own `Vec<i64>` — that
/// per-row copy cost `batch` allocations plus a full copy of the mask on every
/// forward pass.
pub(crate) fn mean_pool_rows_flat(
    hidden: &[f32],
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    masks: &[i64],
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let expected = batch * seq_len * hidden_size;
    if hidden.len() < expected {
        return Err(EmbeddingError::ParseError(format!(
            "hidden tensor too short: need {expected} floats, got {}",
            hidden.len()
        )));
    }
    let mask_len = batch * seq_len;
    if masks.len() < mask_len {
        return Err(EmbeddingError::ParseError(format!(
            "attention mask too short: need {mask_len} entries, got {}",
            masks.len()
        )));
    }
    let mut out = vec![vec![0.0f32; hidden_size]; batch];
    for (b, out_row) in out.iter_mut().enumerate() {
        let mask = &masks[b * seq_len..(b + 1) * seq_len];
        let mut count = 0usize;
        for (s, m) in mask.iter().enumerate() {
            if *m == 0 {
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

/// Row-vector form of [`mean_pool_rows_flat`], for callers that already hold
/// per-row masks. Flattens once and delegates, so both entry points share one
/// implementation of the pooling math.
pub(crate) fn mean_pool_rows(
    hidden: &[f32],
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    masks: &[Vec<i64>],
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let mut flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
    for b in 0..batch {
        let row = masks
            .get(b)
            .ok_or_else(|| EmbeddingError::ParseError("missing attention mask row".to_string()))?;
        if row.len() < seq_len {
            return Err(EmbeddingError::ParseError(format!(
                "attention mask row {b} too short: need {seq_len} entries, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(&row[..seq_len]);
    }
    mean_pool_rows_flat(hidden, batch, seq_len, hidden_size, &flat)
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
            let max_len = self.max_len;
            // One owned copy of the chunk: `spawn_blocking` needs `'static`, so
            // the texts must be owned here. Nothing downstream copies them again
            // — `BlockingEmbedder::embed` hands this `Vec<String>` straight to
            // the tokenizer (v0.27.0 T3; it used to rebuild `Vec<&str>` twice
            // per chunk).
            let chunk: Vec<String> = chunk.iter().map(|s| s.to_string()).collect();

            // Rebuild a blocking embedder view: model + tokenizer are shared
            // Arc/clone-cheap; run on the blocking pool.
            let blocking = BlockingEmbedder {
                model,
                tokenizer,
                dim,
                max_len,
            };
            let rows = tokio::task::spawn_blocking(move || blocking.embed(chunk))
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
    /// Embeds an owned batch.
    ///
    /// Takes the `Vec<String>` by value so `encode_batch` (which is generic over
    /// `T: AsRef<str>` and consumes its input) receives it directly: the old
    /// `&[&str]` signature forced a `Vec<&str>` rebuild inside this call on top
    /// of the one the caller already built (v0.27.0 T3 zero-copy).
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
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

        // v0.27.0 T3 zero-copy: `attention` is already the flat [batch, max_len]
        // mask, so pooling reads it in place — no per-row `Vec<i64>` copies.
        let mut rows = mean_pool_rows_flat(&hidden_data, batch, max_len, self.dim, &attention)?;
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
        fn add(
            w: &mut HashMap<String, Tensor>,
            path: &str,
            shape: (usize, usize),
            device: &Device,
        ) {
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
        add(
            &mut w,
            "embeddings.word_embeddings.weight",
            (TINY_VOCAB, h),
            device,
        );
        add(
            &mut w,
            "embeddings.position_embeddings.weight",
            (16, h),
            device,
        );
        add(
            &mut w,
            "embeddings.token_type_embeddings.weight",
            (2, h),
            device,
        );
        add1(&mut w, "embeddings.LayerNorm.weight", h, device);
        add1(&mut w, "embeddings.LayerNorm.bias", h, device);
        // single encoder layer
        add(
            &mut w,
            "encoder.layer.0.attention.self.query.weight",
            (h, h),
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.attention.self.query.bias",
            h,
            device,
        );
        add(
            &mut w,
            "encoder.layer.0.attention.self.key.weight",
            (h, h),
            device,
        );
        add1(&mut w, "encoder.layer.0.attention.self.key.bias", h, device);
        add(
            &mut w,
            "encoder.layer.0.attention.self.value.weight",
            (h, h),
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.attention.self.value.bias",
            h,
            device,
        );
        add(
            &mut w,
            "encoder.layer.0.attention.output.dense.weight",
            (h, h),
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.attention.output.dense.bias",
            h,
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.attention.output.LayerNorm.weight",
            h,
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.attention.output.LayerNorm.bias",
            h,
            device,
        );
        add(
            &mut w,
            "encoder.layer.0.intermediate.dense.weight",
            (TINY_INTER, h),
            device,
        );
        add1(
            &mut w,
            "encoder.layer.0.intermediate.dense.bias",
            TINY_INTER,
            device,
        );
        add(
            &mut w,
            "encoder.layer.0.output.dense.weight",
            (h, TINY_INTER),
            device,
        );
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
        let q = CandleEmbeddingsBuilder::from_dir(std::env::temp_dir())
            .with_quantization(Quantization::F16);
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

    /// v0.27.0 T3: the FP8 (`F8E4M3`) round-trip preserves shape, stays finite
    /// and is coarser than FP16 (3 mantissa bits → up to ~6% relative error at
    /// N(0,1) scale; an element near 4 can be off by ~0.25).
    #[test]
    fn fp8_quantize_roundtrip_keeps_shape_and_values() {
        let device = Device::Cpu;
        let src = Tensor::randn(0f32, 1f32, (768usize, 384usize), &device).unwrap();
        let dq = src.to_dtype(DType::F8E4M3).unwrap().to_dtype(DType::F32).unwrap();
        assert_eq!(dq.shape(), src.shape());
        let a = src.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = dq.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let max_err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(b.iter().all(|v| v.is_finite()), "no NaN/Inf (SatFinite)");
        // Bound on the maximum relative+absolute error for N(0,1) inputs: the
        // half-ULP rounding step of a 3-mantissa-bit format is 2^5 = 32× FP16's,
        // so a value near 4 has a step of 2^2 × 2^-3 = 0.5 → ≤0.25 error.
        assert!(max_err < 0.5, "FP8 round-trip max error {max_err}");
    }

    /// v0.27.0 T3: the flat-mask pooling reads the same numbers as the
    /// row-vector form — the zero-copy refactor changes nothing about the
    /// result.
    #[test]
    fn mean_pool_rows_flat_matches_rowwise() {
        let hidden = vec![
            1.0, 1.0, 3.0, 3.0, // row0: [1,1] and [3,3] → mean [2,2]
            5.0, 7.0, 100.0, 100.0, // row1: [5,7] only (second masked) → [5,7]
        ];
        let row_masks = vec![vec![1, 1], vec![1, 0]];
        let flat_masks: Vec<i64> = row_masks.iter().flatten().copied().collect();
        let by_rows = mean_pool_rows(&hidden, 2, 2, 2, &row_masks).unwrap();
        let by_flat = mean_pool_rows_flat(&hidden, 2, 2, 2, &flat_masks).unwrap();
        assert_eq!(by_rows, by_flat);
        assert_eq!(by_flat[0], vec![2.0, 2.0]);
        assert_eq!(by_flat[1], vec![5.0, 7.0], "pad position excluded");
    }

    /// The quantized constructors thread the knob and fail fast on missing files.
    #[test]
    fn quantization_knob_fails_fast_on_missing_dir() {
        let dir = std::env::temp_dir().join("lc-candle-quant-nonexistent");
        for quant in [Quantization::F16, Quantization::Int8, Quantization::Fp8] {
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
