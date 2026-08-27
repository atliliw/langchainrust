use super::*;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

// ⚠️ Unverified statement (P2-2/3/4):
//
// This machine (Windows, no MSVC Build Tools, no x86_64-pc-windows-gnu prebuilt artifacts
// for ort-sys on the GNU toolchain) cannot compile the `local-embeddings` feature. All code
// below was written by cross-checking the API against the vendored ort 2.0.0-rc.13 and
// tokenizers 0.22.2 sources, but is **not verified by the compiler**. Run on a machine that
// supports ort:
//
//   cargo test -p lc-embeddings --features local-embeddings --lib
//   cargo clippy -p lc-embeddings --features local-embeddings --lib
//
// Key APIs cross-checked:
// - ort::session::Session::builder().commit_from_memory(&[u8]) (impl_commit.rs:93)
// - Session::run(impl Into<SessionInputs>) requires &mut self; SessionInputs: From<Vec<(K,V)>>
//   where K: Into<Cow<str>>, V: Into<SessionInputValue> (input.rs:62)
// - Tensor::from_array((Vec<i64>, Vec<i64>)); try_extract_tensor::<f32>()
// - tokenizers::Tokenizer::from_file/encode_batch(..., add_special_tokens)
// - Encoding::get_ids/get_attention_mask/get_type_ids; tokenizer.get_padding()/token_to_id()

/// Default per-batch text count cap for dynamic-batch models.
const DEFAULT_MAX_BATCH: usize = 32;
/// Default truncation cap (in tokens per text) when the sequence dimension is dynamic (excess truncated).
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Default session pool size: aligned with the CPU logical core count but capped at 8,
/// avoiding memory blow-up from many ONNX sessions each holding the full model weights.
fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(2)
}

/// Maps lock poisoning to `ApiError` (abnormal internal state; explicit error instead of panic).
fn lock_error<T>(_: std::sync::PoisonError<T>) -> EmbeddingError {
    EmbeddingError::ApiError("session pool lock poisoned".to_string())
}

/// Bounded ONNX Session concurrency pool (P2-3).
///
/// Each `ort::session::Session` holds its own ORT runtime environment; `Session::run`
/// requires `&mut self`, so concurrent inference across threads cannot share a single session.
/// The pool lets concurrent requests reuse up to `capacity` sessions. The model is kept in
/// memory as bytes, and sessions are created lazily via `commit_from_memory` on demand — this
/// avoids `commit_from_file` invalidating if the model file is later deleted, and keeps the
/// pool fully self-contained and shareable across threads. `acquire` blocks on a Condvar when
/// over capacity.
struct SessionPool {
    model_bytes: Arc<Vec<u8>>,
    idle: Mutex<Vec<ort::session::Session>>,
    live: Mutex<usize>,
    capacity: usize,
    notify: Condvar,
}

impl SessionPool {
    /// Pre-builds the first session (live=1); later sessions are created lazily on demand.
    fn new(model_bytes: Vec<u8>, capacity: usize) -> Result<Arc<Self>, EmbeddingError> {
        let capacity = capacity.max(1);
        let pool = Arc::new(Self {
            model_bytes: Arc::new(model_bytes),
            idle: Mutex::new(Vec::new()),
            live: Mutex::new(0),
            capacity,
            notify: Condvar::new(),
        });
        let session = pool.build_session()?;
        *pool.live.lock().map_err(lock_error)? = 1;
        pool.idle.lock().map_err(lock_error)?.push(session);
        Ok(pool)
    }

    fn build_session(&self) -> Result<ort::session::Session, EmbeddingError> {
        ort::session::Session::builder()
            .map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to create ONNX SessionBuilder: {e}"))
            })?
            .commit_from_memory(&self.model_bytes)
            .map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to load ONNX model from memory: {e}"))
            })
    }

    /// Lends out a session. Takes an idle one if available; lazily creates a new one if the
    /// pool is not full; otherwise blocks on the Condvar until one is returned. Callers should
    /// use this from within a `spawn_blocking` thread.
    fn acquire(self: &Arc<Self>) -> Result<SessionGuard, EmbeddingError> {
        // Fast path: the idle queue is non-empty.
        {
            let mut idle = self.idle.lock().map_err(lock_error)?;
            if let Some(session) = idle.pop() {
                return Ok(SessionGuard {
                    pool: self.clone(),
                    session: Some(session),
                });
            }
        }
        // Slow path: need to build a session, or wait for another caller to return one.
        let mut live = self.live.lock().map_err(lock_error)?;
        loop {
            // A session may have been returned while waiting; re-check the idle queue.
            {
                let mut idle = self.idle.lock().map_err(lock_error)?;
                if let Some(session) = idle.pop() {
                    return Ok(SessionGuard {
                        pool: self.clone(),
                        session: Some(session),
                    });
                }
            }
            if *live < self.capacity {
                *live += 1;
                match self.build_session() {
                    Ok(session) => {
                        return Ok(SessionGuard {
                            pool: self.clone(),
                            session: Some(session),
                        });
                    }
                    Err(e) => {
                        // Roll back the live count and wake a waiter so others can retry.
                        *live -= 1;
                        self.notify.notify_one();
                        return Err(e);
                    }
                }
            }
            live = self.notify.wait(live).map_err(lock_error)?;
        }
    }

    /// Returns a session: pushes it to the idle queue and wakes one waiter. Drops the session if the lock is poisoned.
    fn release(&self, session: ort::session::Session) {
        let Ok(mut idle) = self.idle.lock() else {
            return;
        };
        idle.push(session);
        drop(idle);
        self.notify.notify_one();
    }
}

/// RAII session loan handle: returning it to the pool automatically on scope exit (including `?` early returns) without leaking a session.
struct SessionGuard {
    pool: Arc<SessionPool>,
    session: Option<ort::session::Session>,
}

impl SessionGuard {
    fn session(&mut self) -> Result<&mut ort::session::Session, EmbeddingError> {
        self.session
            .as_mut()
            .ok_or_else(|| EmbeddingError::ApiError("session guard already released".to_string()))
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.release(session);
        }
    }
}

/// Internal shared state: pool + tokenizer + static metadata.
///
/// Wrapped in `Arc` so `spawn_blocking` closures capture an owned reference (satisfying
/// `'static`), fixing the latent compile error in the old `move || self.embed_single(&text)`
/// capturing `&self`.
struct LocalInner {
    pool: Arc<SessionPool>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_name: String,
    seq_limit: usize,
    max_batch: usize,
}

impl LocalInner {
    /// Real tokenizers encoding (P2-2): loads a HuggingFace WordPiece/BPE tokenizer, producing
    /// the `input_ids + attention_mask + token_type_ids` triple.
    fn tokenize(&self, texts: &[String]) -> Result<Vec<tokenizers::Encoding>, EmbeddingError> {
        self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingError::Config(format!("Tokenizer failed to encode input: {e}")))
    }

    /// Packs a batch of encodings into tensor data padded to the batch's longest sequence (P2-4).
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids, aligned sequence length max_len)`.
    /// Pure logic with no ONNX session dependency, directly unit-testable. Pad positions get mask=0, type_id=0.
    fn build_batch_tensors(
        encodings: &[tokenizers::Encoding],
        seq_limit: usize,
        pad_id: u32,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>, usize) {
        let batch = encodings.len();
        let lens: Vec<usize> = encodings
            .iter()
            .map(|e| e.get_ids().len().min(seq_limit))
            .collect();
        let max_len = lens.iter().copied().max().unwrap_or(0);
        let pad = pad_id as i64;
        let mut input_ids = vec![pad; batch * max_len];
        let mut attention_mask = vec![0i64; batch * max_len];
        let mut token_type_ids = vec![0i64; batch * max_len];
        for (b, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let len = lens[b];
            let row = b * max_len;
            for i in 0..len {
                input_ids[row + i] = ids[i] as i64;
                attention_mask[row + i] = mask.get(i).copied().unwrap_or(1) as i64;
                token_type_ids[row + i] = types.get(i).copied().unwrap_or(0) as i64;
            }
        }
        (input_ids, attention_mask, token_type_ids, max_len)
    }

    /// Batch inference: feeds pad-aligned tensors, extracts per-row vectors via masked mean pooling, L2-normalizes (P2-4).
    fn infer_rows(
        &self,
        encodings: &[tokenizers::Encoding],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if encodings.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let pad_id = Self::resolve_pad_id(&self.tokenizer);
        let (input_ids, attention_mask, token_type_ids, fed_seq_len) =
            Self::build_batch_tensors(encodings, self.seq_limit, pad_id);
        if fed_seq_len == 0 {
            return Err(EmbeddingError::EmptyInput);
        }
        let batch = encodings.len();
        let mask_rows: Vec<Vec<i64>> = (0..batch)
            .map(|b| attention_mask[b * fed_seq_len..(b + 1) * fed_seq_len].to_vec())
            .collect();

        let input_shape = vec![batch as i64, fed_seq_len as i64];
        let input_tensor = Tensor::from_array((input_shape, input_ids)).map_err(|e| {
            EmbeddingError::ApiError(format!("Failed to construct input_ids tensor: {e}"))
        })?;
        let attention_tensor =
            Tensor::from_array((input_shape.clone(), attention_mask)).map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to construct attention_mask tensor: {e}"))
            })?;
        let type_tensor =
            Tensor::from_array((input_shape.clone(), token_type_ids)).map_err(|e| {
                EmbeddingError::ApiError(format!("Failed to construct token_type_ids tensor: {e}"))
            })?;

        let mut guard = self.pool.acquire()?;
        let input_names: Vec<String> = {
            let session = guard.session()?;
            session
                .inputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect()
        };
        // Feed only the inputs the model declares, aligned by name; unknown input names error explicitly rather than being silently skipped (P2-2).
        let mut input_ids_slot = Some(input_tensor);
        let mut attention_slot = Some(attention_tensor);
        let mut type_slot = Some(type_tensor);
        let mut named: Vec<(String, Tensor<i64>)> = Vec::with_capacity(input_names.len());
        for name in input_names {
            let tensor = match name.as_str() {
                "input_ids" => input_ids_slot.take().ok_or_else(|| {
                    EmbeddingError::ParseError(
                        "ONNX model declares duplicate 'input_ids' input".to_string(),
                    )
                })?,
                "attention_mask" => attention_slot.take().ok_or_else(|| {
                    EmbeddingError::ParseError(
                        "ONNX model declares duplicate 'attention_mask' input".to_string(),
                    )
                })?,
                "token_type_ids" => type_slot.take().ok_or_else(|| {
                    EmbeddingError::ParseError(
                        "ONNX model declares duplicate 'token_type_ids' input".to_string(),
                    )
                })?,
                other => {
                    return Err(EmbeddingError::ParseError(format!(
                        "Unsupported ONNX model input '{other}': the `local-embeddings` \
                         feature only supports input_ids / attention_mask / token_type_ids"
                    )));
                }
            };
            named.push((name, tensor));
        }
        if named.is_empty() {
            return Err(EmbeddingError::ParseError(
                "ONNX model declares no supported inputs".to_string(),
            ));
        }

        let outputs = guard
            .session()?
            .run(named)
            .map_err(|e| EmbeddingError::ApiError(format!("ONNX inference failed: {e}")))?;

        let output_value = outputs
            .get(0)
            .ok_or_else(|| EmbeddingError::ParseError("ONNX model has no output".to_string()))?;
        let (shape, data) = output_value.try_extract_tensor::<f32>().map_err(|e| {
            EmbeddingError::ParseError(format!("Failed to extract output tensor: {e}"))
        })?;
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        let mut rows = Self::pool_rows(&shape_vec, data, &mask_rows, batch, fed_seq_len)?;
        for row in &mut rows {
            crate::l2_normalize(row);
        }
        Ok(rows)
    }

    /// Extracts per-row vectors from the output tensor.
    ///
    /// - 3D `[batch, seq, dim]`: masked mean pooling over the fed attention_mask
    ///   (pad positions with mask=0 do not participate in the mean); when the output sequence
    ///   length differs from the fed one, take the smaller.
    /// - 2D `[batch, dim]`: split by row directly.
    /// - Batch row count != input → explicit `BatchMismatch` (P0-1 alignment contract).
    fn pool_rows(
        shape: &[usize],
        data: &[f32],
        masks: &[Vec<i64>],
        batch: usize,
        fed_seq_len: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match shape.len() {
            3 => {
                let out_batch = shape[0];
                if out_batch != batch {
                    return Err(EmbeddingError::BatchMismatch {
                        expected: batch,
                        actual: out_batch,
                    });
                }
                let out_seq = shape[1];
                let dim = shape[2];
                let seq = out_seq.min(fed_seq_len);
                let mut result = vec![vec![0.0f32; dim]; batch];
                for b in 0..batch {
                    let mask = &masks[b];
                    let mut count = 0usize;
                    for s in 0..seq {
                        if s >= mask.len() || mask[s] == 0 {
                            continue; // pad positions do not participate in the mean
                        }
                        count += 1;
                        let base = (b * out_seq + s) * dim;
                        for d in 0..dim {
                            result[b][d] += data[base + d];
                        }
                    }
                    if count > 0 {
                        for d in 0..dim {
                            result[b][d] /= count as f32;
                        }
                    }
                }
                Ok(result)
            }
            2 => {
                let out_batch = shape[0];
                if out_batch != batch {
                    return Err(EmbeddingError::BatchMismatch {
                        expected: batch,
                        actual: out_batch,
                    });
                }
                let dim = shape[1];
                let mut result = Vec::with_capacity(batch);
                for b in 0..batch {
                    let base = b * dim;
                    result.push(data[base..base + dim].to_vec());
                }
                Ok(result)
            }
            _ => Err(EmbeddingError::ParseError(format!(
                "Unsupported output dimension count: {}",
                shape.len()
            ))),
        }
    }

    /// Runs the full embedding pipeline for a batch (per-chunk pad-aligned inference).
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.max_batch.max(1)) {
            let encodings = self.tokenize(chunk)?;
            results.extend(self.infer_rows(&encodings)?);
        }
        Ok(results)
    }

    /// Embeds a single text.
    fn embed_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut rows = self.embed_batch(&[text.to_string()])?;
        rows.pop().ok_or_else(|| {
            EmbeddingError::ParseError("model returned no embedding for single text".to_string())
        })
    }

    /// Infers the output dimension (last positive dimension of the output shape, as the old implementation).
    fn infer_dimension(session: &ort::session::Session) -> Result<usize, EmbeddingError> {
        let outputs = session.outputs();
        if outputs.is_empty() {
            return Err(EmbeddingError::ParseError(
                "ONNX model has no output nodes".to_string(),
            ));
        }
        let dtype = outputs[0].dtype();
        let shape = dtype
            .tensor_shape()
            .ok_or_else(|| EmbeddingError::ParseError("Output is not a Tensor type".to_string()))?;
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

    /// Infers the batch cap and sequence truncation cap from the first input's static shape (P2-4).
    ///
    /// - static batch=1 → only sequential execution (batch_cap=1);
    /// - static batch=N → per-batch cap `min(N, max_batch)`;
    /// - dynamic batch (-1) → use `max_batch` directly.
    /// Same for the sequence dimension: static uses the value, dynamic uses `max_seq_len`.
    fn infer_input_capability(
        session: &ort::session::Session,
        max_batch: usize,
        max_seq_len: usize,
    ) -> Result<(usize, usize), EmbeddingError> {
        let inputs = session.inputs();
        let input = inputs.first().ok_or_else(|| {
            EmbeddingError::ParseError("ONNX model has no input nodes".to_string())
        })?;
        let shape = input.dtype().tensor_shape().ok_or_else(|| {
            EmbeddingError::ParseError("Model input is not a Tensor type".to_string())
        })?;
        let batch_dim = shape.first().copied().unwrap_or(-1);
        let seq_dim = shape.get(1).copied().unwrap_or(-1);
        let batch_cap = if batch_dim > 0 {
            (batch_dim as usize).min(max_batch)
        } else {
            max_batch
        };
        let seq_limit = if seq_dim > 0 {
            seq_dim as usize
        } else {
            max_seq_len
        };
        Ok((batch_cap, seq_limit))
    }

    /// Resolves the pad token id: prefers the tokenizer's explicit padding config, then the
    /// `[PAD]` vocab entry, finally falls back to 0 (a conservative choice beyond [UNK]/[PAD]).
    fn resolve_pad_id(tokenizer: &tokenizers::Tokenizer) -> u32 {
        tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0)
    }
}

/// Discovers tokenizer.json next to the model: `<model-name>.json` first, then `tokenizer.json`.
fn discover_tokenizer(model_path: &Path) -> Result<PathBuf, EmbeddingError> {
    let dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    for candidate in [dir.join(format!("{stem}.json")), dir.join("tokenizer.json")] {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(EmbeddingError::Config(format!(
        "No tokenizer.json found next to ONNX model '{}'. The `local-embeddings` feature \
         uses the HuggingFace `tokenizers` crate and requires a real tokenizer.json \
         (WordPiece/BPE vocab). Place one as '{stem}.json' or 'tokenizer.json' next to the \
         model, or pass it explicitly via `LocalEmbeddings::from_file_with_tokenizer`. The old \
         byte-hash fake tokenizer was removed in P2-2: feeding fake token IDs to a neural \
         embedding model yields garbage vectors.",
        model_path.display()
    )))
}

/// ONNX Runtime-based local neural network embedding
///
/// Loads from an ONNX model + HuggingFace `tokenizers` (WordPiece/BPE), infers locally with
/// no external API calls — suitable for privacy-sensitive or offline scenarios.
///
/// P2-2: replaces the old byte-hash fake tokenizer with a real one, producing the
/// `input_ids + attention_mask + token_type_ids` triple.
/// P2-3: replaces `RwLock<Session>` with a bounded Session concurrency pool for multi-threaded inference.
/// P2-4: switches to pad-aligned batch inference + masked mean pooling.
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
    inner: Arc<LocalInner>,
}

impl LocalEmbeddings {
    /// Loads from an ONNX model file; auto-discovers tokenizer.json next to the model
    /// (`<model-name>.json` or `tokenizer.json`, see [`LocalEmbeddingsBuilder`]).
    pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
        Self::builder().model_path(model_path).build()
    }

    /// Loads from an ONNX model + an explicit tokenizer.json.
    pub fn from_file_with_tokenizer(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, EmbeddingError> {
        Self::builder()
            .model_path(model_path)
            .tokenizer_path(tokenizer_path)
            .build()
    }

    /// Creates a builder to customize session pool size / batch cap / sequence truncation cap.
    pub fn builder() -> LocalEmbeddingsBuilder {
        LocalEmbeddingsBuilder {
            model_path: None,
            tokenizer_path: None,
            pool_size: default_pool_size(),
            max_batch: DEFAULT_MAX_BATCH,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
        }
    }
}

/// Builder for [`LocalEmbeddings`], for customizing the ONNX session pool and batching strategy.
pub struct LocalEmbeddingsBuilder {
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    pool_size: usize,
    max_batch: usize,
    max_seq_len: usize,
}

impl LocalEmbeddingsBuilder {
    /// Sets the ONNX model path (required).
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the tokenizer.json path; when omitted, auto-discovers it next to the model.
    pub fn tokenizer_path(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the Session concurrency pool size (default = CPU logical cores, capped at 8).
    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Sets the max texts per inference (default 32).
    pub fn max_batch(mut self, n: usize) -> Self {
        self.max_batch = n;
        self
    }

    /// Sets the sequence truncation cap (default 512); excess tokens are truncated.
    pub fn max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = n;
        self
    }

    /// Builds `LocalEmbeddings`: loads the model bytes + tokenizer, pre-builds the session pool, infers dimension and batching capability.
    pub fn build(self) -> Result<LocalEmbeddings, EmbeddingError> {
        let model_path = self.model_path.ok_or_else(|| {
            EmbeddingError::Config(
                "model path is required: call LocalEmbeddings::builder().model_path(path)"
                    .to_string(),
            )
        })?;
        let model_bytes = std::fs::read(&model_path).map_err(|e| {
            EmbeddingError::ApiError(format!(
                "Failed to read ONNX model '{}': {e}",
                model_path.display()
            ))
        })?;
        let tokenizer_path = match self.tokenizer_path {
            Some(p) => p,
            None => discover_tokenizer(&model_path)?,
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            EmbeddingError::Config(format!(
                "Failed to load tokenizer '{}': {e}. Expected a HuggingFace tokenizer.json \
                 (WordPiece/BPE).",
                tokenizer_path.display()
            ))
        })?;

        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let pool = SessionPool::new(model_bytes, self.pool_size)?;

        // Borrow the first session to infer the output dimension and input batch/sequence capability.
        let (dim, max_batch, seq_limit) = {
            let mut guard = pool.acquire()?;
            let session = guard.session()?;
            let dim = LocalInner::infer_dimension(session)?;
            let (max_batch, seq_limit) =
                LocalInner::infer_input_capability(session, self.max_batch, self.max_seq_len)?;
            (dim, max_batch, seq_limit)
        };

        Ok(LocalEmbeddings {
            inner: Arc::new(LocalInner {
                pool,
                tokenizer,
                dim,
                model_name,
                seq_limit,
                max_batch,
            }),
        })
    }
}

#[async_trait]
impl Embeddings for LocalEmbeddings {
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // ONNX inference is CPU-bound; run it in the blocking thread pool. Capture the owned
        // `Arc<LocalInner>` to satisfy spawn_blocking's `'static` bound (fixing the latent
        // `&self` capture error in the old implementation).
        let inner = self.inner.clone();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || inner.embed_single(&text))
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {e}")))?
    }

    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // P1-1: any empty/all-whitespace text errors, consistent with the trait's default contract.
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(EmbeddingError::EmptyInput);
        }

        let inner = self.inner.clone();
        let texts: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        tokio::task::spawn_blocking(move || inner.embed_batch(&texts))
            .await
            .map_err(|e| EmbeddingError::ApiError(format!("Task execution failed: {e}")))?
    }

    fn dimension(&self) -> usize {
        self.inner.dim
    }

    fn model_name(&self) -> &str {
        &self.inner.model_name
    }
}

#[cfg(test)]
mod nn_tests;
