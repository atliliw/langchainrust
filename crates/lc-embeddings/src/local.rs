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
    use std::path::PathBuf;
    use std::sync::{Arc, Condvar, Mutex};

    // ⚠️ 未验证声明（P2-2/3/4）：
    //
    // 本机（Windows，无 MSVC Build Tools，GNU 链上 ort-sys 无 x86_64-pc-windows-gnu
    // 预编译产物）无法编译 `local-embeddings` feature。以下代码全部按 vendored 的
    // ort 2.0.0-rc.13 与 tokenizers 0.22.2 源码逐项核对 API 写成，但**未经编译器验证**。
    // 请在支持 ort 的环境执行：
    //
    //   cargo test -p lc-embeddings --features local-embeddings --lib
    //   cargo clippy -p lc-embeddings --features local-embeddings --lib
    //
    // 已核对的关键 API：
    // - ort::session::Session::builder().commit_from_memory(&[u8])（impl_commit.rs:93）
    // - Session::run(impl Into<SessionInputs>) 需 &mut self；SessionInputs: From<Vec<(K,V)>>
    //   其中 K: Into<Cow<str>>, V: Into<SessionInputValue>（input.rs:62）
    // - Tensor::from_array((Vec<i64>, Vec<i64>))；try_extract_tensor::<f32>()
    // - tokenizers::Tokenizer::from_file/encode_batch(..., add_special_tokens)
    // - Encoding::get_ids/get_attention_mask/get_type_ids；tokenizer.get_padding()/token_to_id()

    /// 动态 batch 模型的默认单批文本数上限。
    const DEFAULT_MAX_BATCH: usize = 32;
    /// 序列维为动态时，单条文本 token 数的默认截断上限（超出截断）。
    const DEFAULT_MAX_SEQ_LEN: usize = 512;

    /// 默认会话池大小：与 CPU 逻辑核数对齐但封顶 8，避免多个 ONNX session
    /// 各自持有整份模型权重导致内存失控。
    fn default_pool_size() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(2)
    }

    /// 锁中毒时映射为 `ApiError`（内部状态异常，显式报错而非 panic）。
    fn lock_error<T>(_: std::sync::PoisonError<T>) -> EmbeddingError {
        EmbeddingError::ApiError("session pool lock poisoned".to_string())
    }

    /// 有界 ONNX Session 并发池（P2-3）。
    ///
    /// 每个 `ort::session::Session` 持有独立的 ORT 运行环境；`Session::run` 要求
    /// `&mut self`，多线程并发推理不能共享单个 session，池让并发请求复用至多
    /// `capacity` 个会话。模型以字节保存在内存中，会话按需 `commit_from_memory`
    /// 惰性创建——既避免模型文件事后被删除导致 `commit_from_file` 失效，也让池
    /// 完全自持、可跨线程共享。`acquire` 超容量时经 Condvar 阻塞等待。
    struct SessionPool {
        model_bytes: Arc<Vec<u8>>,
        idle: Mutex<Vec<ort::session::Session>>,
        live: Mutex<usize>,
        capacity: usize,
        notify: Condvar,
    }

    impl SessionPool {
        /// 预建第一个会话（live=1），后续会话按需惰性创建。
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

        /// 借出一个会话。空闲队列有现成的直接拿；池未满则惰性新建一个；满则
        /// Condvar 阻塞等待归还。调用方应在 `spawn_blocking` 线程内使用本方法。
        fn acquire(self: &Arc<Self>) -> Result<SessionGuard, EmbeddingError> {
            // 快速路径：空闲队列非空。
            {
                let mut idle = self.idle.lock().map_err(lock_error)?;
                if let Some(session) = idle.pop() {
                    return Ok(SessionGuard {
                        pool: self.clone(),
                        session: Some(session),
                    });
                }
            }
            // 慢速路径：需要新建会话，或等待其他调用方归还。
            let mut live = self.live.lock().map_err(lock_error)?;
            loop {
                // 等待期间可能有会话被归还，重新检查空闲队列。
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
                            // 回滚 live 计数并唤醒等待者，让其他人有机会重试。
                            *live -= 1;
                            self.notify.notify_one();
                            return Err(e);
                        }
                    }
                }
                live = self.notify.wait(live).map_err(lock_error)?;
            }
        }

        /// 归还会话：入空闲队列并唤醒一个等待者。锁中毒时直接丢弃会话。
        fn release(&self, session: ort::session::Session) {
            let Ok(mut idle) = self.idle.lock() else {
                return;
            };
            idle.push(session);
            drop(idle);
            self.notify.notify_one();
        }
    }

    /// RAII 会话借出句柄：离开作用域（含 `?` 提前返回）自动归还池，不泄漏 session。
    struct SessionGuard {
        pool: Arc<SessionPool>,
        session: Option<ort::session::Session>,
    }

    impl SessionGuard {
        fn session(&mut self) -> Result<&mut ort::session::Session, EmbeddingError> {
            self.session.as_mut().ok_or_else(|| {
                EmbeddingError::ApiError("session guard already released".to_string())
            })
        }
    }

    impl Drop for SessionGuard {
        fn drop(&mut self) {
            if let Some(session) = self.session.take() {
                self.pool.release(session);
            }
        }
    }

    /// 内部共享状态：池 + tokenizer + 静态元信息。
    ///
    /// 用 `Arc` 包一层，让 `spawn_blocking` 闭包捕获 owned 引用（满足 `'static`），
    /// 修掉旧实现 `move || self.embed_single(&text)` 捕获 `&self` 导致的潜伏编译错误。
    struct LocalInner {
        pool: Arc<SessionPool>,
        tokenizer: tokenizers::Tokenizer,
        dim: usize,
        model_name: String,
        seq_limit: usize,
        max_batch: usize,
    }

    impl LocalInner {
        /// 真实 tokenizers 编码（P2-2）：加载 HuggingFace WordPiece/BPE tokenizer，
        /// 产出 `input_ids + attention_mask + token_type_ids` 三件套。
        fn tokenize(&self, texts: &[String]) -> Result<Vec<tokenizers::Encoding>, EmbeddingError> {
            self.tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| {
                    EmbeddingError::Config(format!("Tokenizer failed to encode input: {e}"))
                })
        }

        /// 把一批 encodings 整理成按 batch 内最长序列 pad 对齐的张量数据（P2-4）。
        ///
        /// 返回 `(input_ids, attention_mask, token_type_ids, 对齐后序列长 max_len)`。
        /// 纯逻辑、不依赖 ONNX 会话，可直接单测。pad 位置 mask=0、type_id=0。
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

        /// 批量推理：pad 对齐喂入，masked mean pooling 出每行向量，L2 归一化（P2-4）。
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
            let attention_tensor = Tensor::from_array((input_shape.clone(), attention_mask))
                .map_err(|e| {
                    EmbeddingError::ApiError(format!(
                        "Failed to construct attention_mask tensor: {e}"
                    ))
                })?;
            let type_tensor =
                Tensor::from_array((input_shape.clone(), token_type_ids)).map_err(|e| {
                    EmbeddingError::ApiError(format!(
                        "Failed to construct token_type_ids tensor: {e}"
                    ))
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
            // 只喂模型声明存在的输入，按名对齐；未知输入名显式报错而非静默跳过（P2-2）。
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

            let output_value = outputs.get(0).ok_or_else(|| {
                EmbeddingError::ParseError("ONNX model has no output".to_string())
            })?;
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

        /// 从输出张量提取每行向量。
        ///
        /// - 3D `[batch, seq, dim]`：按喂入的 attention_mask 做 masked mean pooling
        ///   （mask=0 的 pad 位置不参与均值）；输出序列长与喂入长不同时取较小者。
        /// - 2D `[batch, dim]`：直接按行切。
        /// - batch 行数与输入不一致 → 显式 `BatchMismatch`（P0-1 对齐契约）。
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
                                continue; // pad 位置不参与均值
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

        /// 批量执行完整嵌入管线（chunk 内逐批 pad 对齐推理）。
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

        /// 单条文本嵌入。
        fn embed_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            let mut rows = self.embed_batch(&[text.to_string()])?;
            rows.pop().ok_or_else(|| {
                EmbeddingError::ParseError(
                    "model returned no embedding for single text".to_string(),
                )
            })
        }

        /// 推断输出维度（取输出 shape 最后一个正维度，同旧实现）。
        fn infer_dimension(session: &ort::session::Session) -> Result<usize, EmbeddingError> {
            let outputs = session.outputs();
            if outputs.is_empty() {
                return Err(EmbeddingError::ParseError(
                    "ONNX model has no output nodes".to_string(),
                ));
            }
            let dtype = outputs[0].dtype();
            let shape = dtype.tensor_shape().ok_or_else(|| {
                EmbeddingError::ParseError("Output is not a Tensor type".to_string())
            })?;
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

        /// 从模型第一个输入的静态 shape 推断 batch 上限与序列截断上限（P2-4）。
        ///
        /// - 静态 batch=1 → 只能顺序执行（batch_cap=1）；
        /// - 静态 batch=N → 单批上限 `min(N, max_batch)`；
        /// - 动态 batch（-1）→ 直接用 `max_batch`。
        /// 序列维同理：静态给值则取该值，动态取 `max_seq_len`。
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

        /// 解析 pad token id：优先 tokenizer 显式 padding 配置，其次词表 `[PAD]`，
        /// 最后回退 0（[UNK]/[PAD] 之外的保守选择）。
        fn resolve_pad_id(tokenizer: &tokenizers::Tokenizer) -> u32 {
            tokenizer
                .get_padding()
                .map(|p| p.pad_id)
                .or_else(|| tokenizer.token_to_id("[PAD]"))
                .unwrap_or(0)
        }
    }

    /// 在模型同目录发现 tokenizer.json：`<模型名>.json` 优先，其次 `tokenizer.json`。
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
    /// 从 ONNX 模型 + HuggingFace `tokenizers`（WordPiece/BPE）加载，本地推理，
    /// 无外部 API 调用，适合隐私敏感或离线场景。
    ///
    /// P2-2：以真实 tokenizer 替换旧的字节 hash 伪 tokenizer，组齐
    /// `input_ids + attention_mask + token_type_ids`。
    /// P2-3：`RwLock<Session>` 改为有界 Session 并发池，支持多线程并发推理。
    /// P2-4：改为按最长序列 pad 对齐的批量推理 + masked mean pooling。
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
        /// 从 ONNX 模型文件加载；自动在模型同目录发现 tokenizer.json
        /// （`<模型名>.json` 或 `tokenizer.json`，见 [`LocalEmbeddingsBuilder`]）。
        pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, EmbeddingError> {
            Self::builder().model_path(model_path).build()
        }

        /// 从 ONNX 模型 + 显式 tokenizer.json 加载。
        pub fn from_file_with_tokenizer(
            model_path: impl AsRef<Path>,
            tokenizer_path: impl AsRef<Path>,
        ) -> Result<Self, EmbeddingError> {
            Self::builder()
                .model_path(model_path)
                .tokenizer_path(tokenizer_path)
                .build()
        }

        /// 创建构建器，可定制会话池大小 / 批量上限 / 序列截断上限。
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

    /// [`LocalEmbeddings`] 构建器，用于定制 ONNX 会话池与批量策略。
    pub struct LocalEmbeddingsBuilder {
        model_path: Option<PathBuf>,
        tokenizer_path: Option<PathBuf>,
        pool_size: usize,
        max_batch: usize,
        max_seq_len: usize,
    }

    impl LocalEmbeddingsBuilder {
        /// 设置 ONNX 模型路径（必填）。
        pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
            self.model_path = Some(path.as_ref().to_path_buf());
            self
        }

        /// 设置 tokenizer.json 路径；缺省时自动在模型同目录发现。
        pub fn tokenizer_path(mut self, path: impl AsRef<Path>) -> Self {
            self.tokenizer_path = Some(path.as_ref().to_path_buf());
            self
        }

        /// 设置 Session 并发池大小（默认 = CPU 逻辑核数，封顶 8）。
        pub fn pool_size(mut self, size: usize) -> Self {
            self.pool_size = size;
            self
        }

        /// 设置单次推理的最大文本数（默认 32）。
        pub fn max_batch(mut self, n: usize) -> Self {
            self.max_batch = n;
            self
        }

        /// 设置序列截断上限（默认 512），超出的 token 会被截断。
        pub fn max_seq_len(mut self, n: usize) -> Self {
            self.max_seq_len = n;
            self
        }

        /// 构建 `LocalEmbeddings`：加载模型字节 + tokenizer，预建会话池，推断维度与批量能力。
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

            // 借用首个会话推断输出维度与输入 batch/序列能力。
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
            // ONNX 推理是 CPU 密集，放入阻塞线程池。捕获 owned `Arc<LocalInner>`
            // 以满足 spawn_blocking 的 `'static` 约束（修掉旧实现 `&self` 捕获的潜伏错误）。
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
            // P1-1: 任一空/全空白文本都报错，与 trait 默认契约一致。
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
    mod nn_tests {
        use super::*;

        /// 构建一个最小 WordPiece tokenizer（JSON 走 `Tokenizer::from_bytes`，
        /// 与真实 tokenizer.json 的加载路径一致）。词表：`[UNK]=0, hello=1, world=2`，
        /// `with_pad_in_vocab=true` 时追加 `[PAD]=3`。
        fn tiny_tokenizer(with_pad_in_vocab: bool) -> tokenizers::Tokenizer {
            let mut vocab = serde_json::json!({
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
            });
            if with_pad_in_vocab {
                vocab["[PAD]"] = serde_json::json!(3);
            }
            let json = serde_json::json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": { "type": "Whitespace" },
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "WordPiece",
                    "vocab": vocab,
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100
                }
            });
            tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes())
                .expect("tiny tokenizer should deserialize")
        }

        #[test]
        fn test_l2_normalize() {
            let mut v = vec![3.0, 4.0];
            crate::l2_normalize(&mut v);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
            assert!((v[0] - 0.6).abs() < 1e-5);
            assert!((v[1] - 0.8).abs() < 1e-5);
        }

        #[test]
        fn test_l2_normalize_zero() {
            let mut v = vec![0.0, 0.0, 0.0];
            crate::l2_normalize(&mut v);
            assert!(v.iter().all(|x| *x == 0.0));
        }

        /// P2-2: 真实 WordPiece tokenizer 输出确定 token ID（无 post_processor，
        /// `add_special_tokens=true` 也不会追加 [CLS]/[SEP]）。
        #[test]
        fn test_tokenize_real_wordpiece() {
            let tok = tiny_tokenizer(false);
            let enc = tok.encode("hello world", true).unwrap();
            assert_eq!(enc.get_ids(), &[1u32, 2u32]);
            assert_eq!(enc.get_attention_mask(), &[1u32, 1u32]);
        }

        /// P2-2: 词表外单词回退到 [UNK]=0。
        #[test]
        fn test_tokenize_unknown_word_uses_unk() {
            let tok = tiny_tokenizer(false);
            let enc = tok.encode("zzzznotinvocab", true).unwrap();
            assert_eq!(enc.get_ids(), &[0u32]);
        }

        /// P2-4: 批量 pad 对齐——短行补 pad_id、mask 补 0，长行不动。
        #[test]
        fn test_build_batch_tensors_pads_to_longest() {
            let tok = tiny_tokenizer(false);
            let encodings = tok
                .encode_batch(vec!["hello".to_string(), "hello world".to_string()], true)
                .unwrap();
            let (input_ids, attention_mask, token_type_ids, max_len) =
                LocalInner::build_batch_tensors(&encodings, 8, 0);
            assert_eq!(max_len, 2);
            // "hello" → [1, PAD(0)]，"hello world" → [1, 2]
            assert_eq!(input_ids, vec![1, 0, 1, 2]);
            assert_eq!(attention_mask, vec![1, 0, 1, 1]);
            assert_eq!(token_type_ids, vec![0, 0, 0, 0]);
        }

        #[test]
        fn test_resolve_pad_id_with_pad_token() {
            let tok = tiny_tokenizer(true);
            assert_eq!(LocalInner::resolve_pad_id(&tok), 3);
        }

        #[test]
        fn test_resolve_pad_id_defaults_zero() {
            let tok = tiny_tokenizer(false);
            assert_eq!(LocalInner::resolve_pad_id(&tok), 0);
        }

        /// P2-4: 3D masked mean pooling——mask=0 的 pad 位置不参与均值。
        #[test]
        fn test_pool_rows_3d_masked() {
            // shape [2, 3, 2]：两行各 3 个位置、dim=2。
            let shape = vec![2usize, 3, 2];
            let data = vec![
                // row0: tokens [1,2,3]
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, // row1
                4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
            ];
            let masks = vec![vec![1, 1, 1], vec![1, 0, 0]];
            let rows = LocalInner::pool_rows(&shape, &data, &masks, 2, 3).unwrap();
            assert_eq!(rows.len(), 2);
            // row0 均值 = ((1+2+3)/3, (10+20+30)/3) = (2, 20)
            assert!((rows[0][0] - 2.0).abs() < 1e-5);
            assert!((rows[0][1] - 20.0).abs() < 1e-5);
            // row1 只取第一个位置 = (4, 40)
            assert!((rows[1][0] - 4.0).abs() < 1e-5);
            assert!((rows[1][1] - 40.0).abs() < 1e-5);
        }

        /// P2-4: 2D `[batch, dim]` 输出直接按行切。
        #[test]
        fn test_pool_rows_2d() {
            let shape = vec![2usize, 2];
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let masks = vec![vec![1], vec![1]];
            let rows = LocalInner::pool_rows(&shape, &data, &masks, 2, 1).unwrap();
            assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        }

        /// P0-1 对齐契约：模型输出行数与输入 batch 不一致 → 显式 BatchMismatch。
        #[test]
        fn test_pool_rows_batch_mismatch() {
            let shape = vec![3usize, 2, 2];
            let data = vec![0.0; 12];
            let masks = vec![vec![1], vec![1]];
            let err = LocalInner::pool_rows(&shape, &data, &masks, 2, 1).unwrap_err();
            assert!(matches!(
                err,
                EmbeddingError::BatchMismatch {
                    expected: 2,
                    actual: 3
                }
            ));
        }
    }
}

// When local-embeddings feature is enabled, re-export LocalEmbeddings and its builder
#[cfg(feature = "local-embeddings")]
pub use nn::{LocalEmbeddings, LocalEmbeddingsBuilder};

// ---------------------------------------------------------------------------
// Backward compatibility: LocalEmbeddings without feature points to BagOfWordsEmbeddings
// ---------------------------------------------------------------------------

/// Without the `local-embeddings` feature, `LocalEmbeddings` is a type alias for `BagOfWordsEmbeddings`,
/// maintaining backward compatibility.
///
/// With the `local-embeddings` feature enabled, `LocalEmbeddings` becomes the ONNX Runtime-based neural network implementation.
///
/// P2-1: 消除静默降级。无 feature 时 `LocalEmbeddings` 静默退化为词袋哈希嵌入,
/// 用户以为在用语义向量、实际是词频——"好像能用,但不对"。这里加
/// `#[deprecated]` 让降级在编译期可见:使用者需显式改用 `BagOfWordsEmbeddings`,
/// 或开启 `local-embeddings` feature 使用真正的 ONNX 神经嵌入。
#[cfg(not(feature = "local-embeddings"))]
#[deprecated(
    note = "LocalEmbeddings without the `local-embeddings` feature degrades to \
            BagOfWordsEmbeddings (bag-of-words hash), not semantic neural embedding. \
            Enable the `local-embeddings` feature, or use BagOfWordsEmbeddings explicitly."
)]
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

    /// P2-1: 该测试正是验证"无 feature 时 LocalEmbeddings = BagOfWordsEmbeddings",
    /// 是有意使用已弃用别名,`#[allow(deprecated)]` 豁免降级警告。
    #[allow(deprecated)]
    #[tokio::test]
    async fn test_local_embeddings_backward_compat() {
        // Without feature, LocalEmbeddings = BagOfWordsEmbeddings
        let e = LocalEmbeddings::new(64);
        let v = e.embed_query("test backward compat").await.unwrap();
        assert_eq!(v.len(), 64);
        assert_eq!(e.model_name(), "local-bow");
    }
}
