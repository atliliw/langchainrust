// lc-embeddings/src/catalog.rs
//! Persistent local model catalog (v0.27.0 T3, triage §5 "持久化模型目录").
//!
//! A [`ModelCatalog`] is a small JSON file on disk that records which local
//! models are available: `name → directory / layout / dimension / checksum /
//! timestamp / origin`. It exists so a deployment can **pre-download** (or
//! pre-stage) a fixed set of models once and then resolve them by name at
//! runtime, instead of re-discovering paths (or re-downloading) on every start.
//!
//! Design constraints this module holds to:
//!
//! - **No silent fallback.** A name that is not registered is
//!   [`EmbeddingError::ModelNotFound`]; a name whose files are missing on disk is
//!   [`EmbeddingError::ModelNotReady`] listing exactly what is missing. Neither
//!   case degrades to "use some other model".
//! - **Durable writes.** The file is written to a temp file in the same
//!   directory and `rename`d over the target, so a reader never observes a
//!   half-written catalog. Writers additionally serialize on a `<file>.lock`
//!   file (created with `create_new`, removed on drop) and re-read the on-disk
//!   state while holding it, so concurrent `register` calls merge instead of
//!   losing each other's entries.
//! - **Deterministic bytes.** Entries live in a [`BTreeMap`], so the JSON key
//!   order is stable across runs and machines (a `HashMap` would make the file
//!   churn on every rewrite and defeat diffing).
//! - **Offline first.** [`LocalOnlyFetcher`] never touches the network; a host
//!   with the model files already on disk registers metadata only. The
//!   `local-candle` feature adds an `hf-hub`-backed fetcher for real downloads.
//! - **No new dependencies**: `serde_json` (already a dependency) plus `std`.
//!
//! The catalog is deliberately *not* a model loader. It answers "where is the
//! model named X, and is it intact?" — the backend constructors
//! (`CandleEmbeddings::from_catalog`, `LocalEmbeddings::from_catalog`) turn that
//! answer into a loaded model.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::EmbeddingError;

/// Schema version written into the catalog file; a file declaring any other
/// version is rejected explicitly instead of being guessed at.
const CATALOG_FORMAT_VERSION: u32 = 1;

/// How long a writer waits for the catalog lock before failing.
const LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// Poll interval while waiting for the lock.
const LOCK_POLL: Duration = Duration::from_millis(5);

/// Read buffer size for [`file_checksum`].
const CHECKSUM_BUFFER: usize = 64 * 1024;

/// Environment variable that overrides [`ModelCatalog::default_path`].
pub const CATALOG_PATH_ENV: &str = "LC_EMBEDDINGS_MODEL_CATALOG";

/// Disambiguates temp files written by concurrent `save` calls in one process.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// On-disk layout of a registered model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLayout {
    /// Candle BERT directory: `config.json` + `tokenizer.json` +
    /// `model.safetensors` (see `CandleEmbeddings::from_dir`).
    CandleBert,
    /// ONNX Runtime directory: `model.onnx` + `tokenizer.json` (see
    /// `LocalEmbeddings::from_file_with_tokenizer`).
    Onnx,
}

impl ModelLayout {
    /// Every file that must exist inside the model directory for the model to
    /// be loadable.
    pub const fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::CandleBert => &["config.json", "tokenizer.json", "model.safetensors"],
            Self::Onnx => &["model.onnx", "tokenizer.json"],
        }
    }

    /// The weights file — the one [`file_checksum`] hashes.
    pub const fn weights_file(self) -> &'static str {
        match self {
            Self::CandleBert => "model.safetensors",
            Self::Onnx => "model.onnx",
        }
    }
}

/// Where a registered model's files came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    /// The files were already on disk when the entry was registered.
    Local,
    /// The files were (or will be) fetched from this HuggingFace repo id.
    HuggingFace {
        /// HuggingFace repository id, e.g. `BAAI/bge-small-en-v1.5`.
        repo_id: String,
    },
}

/// One registered model.
///
/// Fields are public so callers can build entries directly (offline metadata
/// registration on an air-gapped host) without a builder ceremony.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Logical name the catalog is resolved by (unique within a catalog).
    pub name: String,
    /// Local model directory holding the files of [`ModelEntry::layout`].
    pub dir: PathBuf,
    /// File layout expected inside [`ModelEntry::dir`].
    pub layout: ModelLayout,
    /// Embedding width the model produces.
    pub dimension: usize,
    /// Optional integrity string for the weights file, by convention
    /// `fnv1a64:<16 hex digits>` as produced by [`file_checksum`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    /// Unix timestamp (seconds) at registration.
    pub registered_at: u64,
    /// Where the files came from.
    pub source: ModelSource,
}

impl ModelEntry {
    /// Creates a local-source entry stamped with the current time.
    pub fn new(
        name: impl Into<String>,
        dir: impl Into<PathBuf>,
        layout: ModelLayout,
        dimension: usize,
    ) -> Self {
        Self {
            name: name.into(),
            dir: dir.into(),
            layout,
            dimension,
            checksum: None,
            registered_at: now_unix_secs(),
            source: ModelSource::Local,
        }
    }

    /// Sets the origin of the files.
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source;
        self
    }

    /// Sets the recorded integrity string (see [`file_checksum`]).
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Path of the weights file inside [`ModelEntry::dir`].
    pub fn model_path(&self) -> PathBuf {
        self.dir.join(self.layout.weights_file())
    }

    /// Path of the tokenizer file inside [`ModelEntry::dir`].
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join("tokenizer.json")
    }

    /// Path of `config.json`, for layouts that have one (`CandleBert`).
    pub fn config_path(&self) -> Option<PathBuf> {
        match self.layout {
            ModelLayout::CandleBert => Some(self.dir.join("config.json")),
            ModelLayout::Onnx => None,
        }
    }

    /// Files declared by the layout that are **not** present on disk.
    pub fn missing_files(&self) -> Vec<PathBuf> {
        self.layout
            .required_files()
            .iter()
            .map(|f| self.dir.join(f))
            .filter(|p| !p.is_file())
            .collect()
    }

    /// Whether every file of the layout exists locally.
    pub fn is_ready(&self) -> bool {
        self.missing_files().is_empty()
    }

    /// Computes the checksum of the weights file (`fnv1a64:<hex>`).
    ///
    /// Fails with an explicit error when the weights file is missing, so a
    /// catalog entry can never be stamped with a checksum of nothing.
    pub fn compute_checksum(&self) -> Result<String, EmbeddingError> {
        file_checksum(self.model_path())
    }

    /// Recomputes the weights checksum and compares it with the recorded one.
    ///
    /// Returns `Ok(None)` when the entry carries no checksum (nothing to
    /// verify — this is reported, not treated as a pass).
    pub fn verify_checksum(&self) -> Result<Option<bool>, EmbeddingError> {
        match &self.checksum {
            None => Ok(None),
            Some(recorded) => Ok(Some(file_checksum(self.model_path())? == *recorded)),
        }
    }
}

/// A model to materialize during [`ModelCatalog::prefetch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// Logical name to register the model under.
    pub name: String,
    /// Directory the fetcher should materialize the files into.
    pub dir: PathBuf,
    /// Expected file layout of `dir`.
    pub layout: ModelLayout,
    /// Embedding width the model produces.
    pub dimension: usize,
    /// Where the files come from.
    pub source: ModelSource,
}

impl ModelSpec {
    /// Creates a spec with [`ModelSource::Local`].
    pub fn new(
        name: impl Into<String>,
        dir: impl Into<PathBuf>,
        layout: ModelLayout,
        dimension: usize,
    ) -> Self {
        Self {
            name: name.into(),
            dir: dir.into(),
            layout,
            dimension,
            source: ModelSource::Local,
        }
    }

    /// Sets the origin of the files.
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source;
        self
    }
}

/// Resolves a [`ModelSpec`] to a local model directory, downloading if needed.
///
/// Implementations must return the directory that now holds the spec's files.
pub trait ModelFetcher {
    /// Materializes `spec` and returns its model directory.
    fn fetch(&self, spec: &ModelSpec) -> Result<PathBuf, EmbeddingError>;
}

/// Offline fetcher: accepts only directories that already exist locally and
/// never touches the network.
///
/// Pre-staged hosts (models copied in by a provisioning step) use this; a
/// missing directory is an explicit error rather than a silent no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalOnlyFetcher;

impl ModelFetcher for LocalOnlyFetcher {
    fn fetch(&self, spec: &ModelSpec) -> Result<PathBuf, EmbeddingError> {
        if !spec.dir.is_dir() {
            return Err(EmbeddingError::Config(format!(
                "offline prefetch: model directory '{}' for '{}' does not exist",
                spec.dir.display(),
                spec.name
            )));
        }
        Ok(spec.dir.clone())
    }
}

/// Outcome of a [`ModelCatalog::prefetch`] call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefetchReport {
    /// Names registered (or refreshed) by this call, in spec order.
    pub registered: Vec<String>,
    /// Names already registered and ready, left untouched, in spec order.
    pub skipped: Vec<String>,
}

impl PrefetchReport {
    /// Whether the call had nothing to do.
    pub fn is_empty(&self) -> bool {
        self.registered.is_empty() && self.skipped.is_empty()
    }
}

/// JSON document written to disk.
#[derive(Debug, Serialize, Deserialize)]
struct CatalogFile {
    version: u32,
    entries: BTreeMap<String, ModelEntry>,
}

/// A persistent `name → local model` directory backed by a JSON file.
///
/// Cheap to hold: entries are in memory, the file is touched only by
/// [`ModelCatalog::register`] / [`ModelCatalog::unregister`] /
/// [`ModelCatalog::prefetch`] (which persist immediately) and by
/// [`ModelCatalog::reload`].
#[derive(Debug)]
pub struct ModelCatalog {
    path: PathBuf,
    entries: BTreeMap<String, ModelEntry>,
}

impl ModelCatalog {
    /// Opens the catalog at `path`, loading it when the file exists.
    ///
    /// A missing file is an empty catalog (first run). A file that exists but
    /// cannot be read or parsed is an explicit error — never silently treated
    /// as "empty", which would drop every registration on the floor.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, EmbeddingError> {
        let path = path.into();
        let entries = Self::read_from(&path)?;
        Ok(Self { path, entries })
    }

    /// Default catalog location: [`CATALOG_PATH_ENV`] when set, otherwise
    /// `<home>/.lc/models/catalog.json`, falling back to the system temp
    /// directory when no home directory is known.
    pub fn default_path() -> PathBuf {
        if let Some(explicit) = std::env::var_os(CATALOG_PATH_ENV).filter(|v| !v.is_empty()) {
            return PathBuf::from(explicit);
        }
        let home = std::env::var_os("USERPROFILE")
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var_os("HOME").filter(|v| !v.is_empty()));
        match home {
            Some(home) => PathBuf::from(home)
                .join(".lc")
                .join("models")
                .join("catalog.json"),
            None => std::env::temp_dir().join("lc-embeddings-models.json"),
        }
    }

    /// Path of the backing JSON file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks a model up by name without any readiness check.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.get(name)
    }

    /// Iterates over the entries in name order (deterministic).
    pub fn iter(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.values()
    }

    /// Resolves a model by name.
    ///
    /// # Errors
    /// [`EmbeddingError::ModelNotFound`] when the name is not registered. The
    /// caller gets no fallback model — an unknown name is a configuration bug,
    /// not something to paper over with a default.
    pub fn resolve(&self, name: &str) -> Result<&ModelEntry, EmbeddingError> {
        self.entries
            .get(name)
            .ok_or_else(|| EmbeddingError::ModelNotFound {
                name: name.to_string(),
            })
    }

    /// Resolves a model by name **and** verifies its files are all present.
    ///
    /// # Errors
    /// [`EmbeddingError::ModelNotFound`] when the name is not registered, or
    /// [`EmbeddingError::ModelNotReady`] (listing the missing files) when it is
    /// registered but incomplete on disk.
    pub fn resolve_ready(&self, name: &str) -> Result<&ModelEntry, EmbeddingError> {
        let entry = self.resolve(name)?;
        let missing = entry.missing_files();
        if !missing.is_empty() {
            return Err(EmbeddingError::ModelNotReady {
                name: name.to_string(),
                missing: missing
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect(),
            });
        }
        Ok(entry)
    }

    /// Registers (or replaces) one entry and persists the catalog.
    ///
    /// The write happens under the catalog lock, after re-reading the on-disk
    /// state, so two concurrent registrations both survive.
    ///
    /// # Errors
    /// [`EmbeddingError::Config`] for an empty name or a zero dimension —
    /// garbage never reaches the file.
    pub fn register(&mut self, entry: ModelEntry) -> Result<(), EmbeddingError> {
        validate(&entry)?;
        let _lock = FileLock::acquire(&self.path, LOCK_TIMEOUT)?;
        self.reload_locked()?;
        self.entries.insert(entry.name.clone(), entry);
        self.write_atomically()
    }

    /// Removes an entry. Returns whether it was present; persists either way
    /// only when something changed.
    pub fn unregister(&mut self, name: &str) -> Result<bool, EmbeddingError> {
        let _lock = FileLock::acquire(&self.path, LOCK_TIMEOUT)?;
        self.reload_locked()?;
        if self.entries.remove(name).is_none() {
            return Ok(false);
        }
        self.write_atomically()?;
        Ok(true)
    }

    /// Pre-download entry point: materializes every spec through `fetcher` and
    /// registers the results in a single write.
    ///
    /// All-or-nothing by design: every spec is fetched and validated *before*
    /// the catalog file is touched, so a spec that fails leaves the catalog
    /// exactly as it was instead of half-registered. Specs already registered
    /// and ready are reported in [`PrefetchReport::skipped`] and left untouched
    /// (re-running a prefetch is idempotent).
    ///
    /// Offline hosts pass [`LocalOnlyFetcher`] and get metadata-only
    /// registration for directories that are already on disk.
    pub fn prefetch(
        &mut self,
        specs: &[ModelSpec],
        fetcher: &dyn ModelFetcher,
    ) -> Result<PrefetchReport, EmbeddingError> {
        let mut resolved: Vec<ModelEntry> = Vec::with_capacity(specs.len());
        let mut skipped: Vec<String> = Vec::new();
        for spec in specs {
            if let Some(existing) = self.entries.get(&spec.name) {
                if existing.is_ready() {
                    skipped.push(spec.name.clone());
                    continue;
                }
            }
            let dir = fetcher.fetch(spec)?;
            let mut entry =
                ModelEntry::new(spec.name.clone(), dir, spec.layout, spec.dimension)
                    .with_source(spec.source.clone());
            let missing = entry.missing_files();
            if !missing.is_empty() {
                return Err(EmbeddingError::Config(format!(
                    "prefetch of '{}' is incomplete: missing {}",
                    spec.name,
                    missing
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            entry.checksum = Some(entry.compute_checksum()?);
            validate(&entry)?;
            resolved.push(entry);
        }

        if resolved.is_empty() {
            return Ok(PrefetchReport {
                registered: Vec::new(),
                skipped,
            });
        }

        let _lock = FileLock::acquire(&self.path, LOCK_TIMEOUT)?;
        self.reload_locked()?;
        let mut registered = Vec::with_capacity(resolved.len());
        for entry in resolved {
            registered.push(entry.name.clone());
            self.entries.insert(entry.name.clone(), entry);
        }
        self.write_atomically()?;
        Ok(PrefetchReport {
            registered,
            skipped,
        })
    }

    /// Re-reads the catalog file, discarding in-memory state.
    pub fn reload(&mut self) -> Result<(), EmbeddingError> {
        let _lock = FileLock::acquire(&self.path, LOCK_TIMEOUT)?;
        self.reload_locked()
    }

    /// Persists the current in-memory state (atomic write under the lock).
    pub fn save(&self) -> Result<(), EmbeddingError> {
        let _lock = FileLock::acquire(&self.path, LOCK_TIMEOUT)?;
        self.write_atomically()
    }

    /// Re-reads the file assuming the caller already holds the lock.
    fn reload_locked(&mut self) -> Result<(), EmbeddingError> {
        self.entries = Self::read_from(&self.path)?;
        Ok(())
    }

    /// Reads and validates a catalog file; a missing file is an empty catalog.
    fn read_from(path: &Path) -> Result<BTreeMap<String, ModelEntry>, EmbeddingError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(e) => {
                return Err(EmbeddingError::Config(format!(
                    "failed to read model catalog '{}': {e}",
                    path.display()
                )));
            }
        };
        if bytes.is_empty() {
            return Ok(BTreeMap::new());
        }
        let file: CatalogFile = serde_json::from_slice(&bytes).map_err(|e| {
            EmbeddingError::Config(format!(
                "failed to parse model catalog '{}': {e}",
                path.display()
            ))
        })?;
        if file.version != CATALOG_FORMAT_VERSION {
            return Err(EmbeddingError::Config(format!(
                "model catalog '{}' declares format version {}, but this build understands \
                 version {CATALOG_FORMAT_VERSION}",
                path.display(),
                file.version
            )));
        }
        Ok(file.entries)
    }

    /// Writes the catalog through a temp file + rename (atomic on both
    /// Windows and Unix), so readers never see a partial document.
    fn write_atomically(&self) -> Result<(), EmbeddingError> {
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| {
                EmbeddingError::Config(format!(
                    "failed to create model catalog directory '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        let document = CatalogFile {
            version: CATALOG_FORMAT_VERSION,
            entries: self.entries.clone(),
        };
        let json = serde_json::to_vec_pretty(&document).map_err(|e| {
            EmbeddingError::Config(format!("failed to serialize model catalog: {e}"))
        })?;

        let seq = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = self
            .path
            .with_extension(format!("tmp-{}-{seq}", std::process::id()));
        fs::write(&tmp, &json).map_err(|e| {
            EmbeddingError::Config(format!(
                "failed to write model catalog temp file '{}': {e}",
                tmp.display()
            ))
        })?;
        // `fs::rename` replaces an existing destination on Unix (rename(2)) and
        // on Windows (MoveFileEx with MOVEFILE_REPLACE_EXISTING), which is what
        // makes the swap atomic for readers.
        fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = fs::remove_file(&tmp);
            EmbeddingError::Config(format!(
                "failed to commit model catalog '{}': {e}",
                self.path.display()
            ))
        })
    }
}

/// Rejects entries that would poison the catalog.
fn validate(entry: &ModelEntry) -> Result<(), EmbeddingError> {
    if entry.name.trim().is_empty() {
        return Err(EmbeddingError::Config(
            "model catalog entries need a non-empty name".to_string(),
        ));
    }
    if entry.dimension == 0 {
        return Err(EmbeddingError::Config(format!(
            "model catalog entry '{}' has dimension 0",
            entry.name
        )));
    }
    Ok(())
}

/// Exclusive lock over a catalog file, implemented with a `<file>.lock` sibling
/// created via `create_new` (the OS guarantees only one creator wins) and
/// removed on drop.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(target: &Path, timeout: Duration) -> Result<Self, EmbeddingError> {
        let path = target.with_extension("lock");
        let deadline = Instant::now() + timeout;
        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if Instant::now() >= deadline {
                        return Err(EmbeddingError::Config(format!(
                            "timed out waiting for the model catalog lock '{}' after {:?}",
                            path.display(),
                            timeout
                        )));
                    }
                    std::thread::sleep(LOCK_POLL);
                }
                Err(e) => {
                    return Err(EmbeddingError::Config(format!(
                        "failed to create model catalog lock '{}': {e}",
                        path.display()
                    )));
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Streaming FNV-1a (64-bit) hasher — the same function `BagOfWordsEmbeddings`
/// uses for token hashing, so the crate ships one checksum algorithm and no
/// extra dependency.
struct Fnv1a64(u64);

impl Fnv1a64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Checksums a file without loading it whole: `fnv1a64:<16 hex digits>`.
///
/// This is an **integrity** check (did the file change?), not a
/// cryptographic one — it detects truncation, a swapped model and bit rot, and
/// is fast enough to run at registration time.
pub fn file_checksum(path: impl AsRef<Path>) -> Result<String, EmbeddingError> {
    let path = path.as_ref();
    let mut file = fs::File::open(path).map_err(|e| {
        EmbeddingError::Config(format!(
            "failed to open '{}' for checksum: {e}",
            path.display()
        ))
    })?;
    let mut hasher = Fnv1a64::new();
    let mut buf = vec![0u8; CHECKSUM_BUFFER];
    loop {
        let read = file.read(&mut buf).map_err(|e| {
            EmbeddingError::Config(format!("failed to read '{}': {e}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("fnv1a64:{:016x}", hasher.finish()))
}

/// Unix timestamp in seconds (0 when the clock is before the epoch).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64 as TestCounter;

    static TEST_DIR_COUNTER: TestCounter = TestCounter::new(0);

    /// Self-cleaning scratch directory (no `tempfile` dependency).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let seq = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "lc-embeddings-catalog-{}-{tag}-{seq}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("scratch dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Creates `<root>/<name>` with every file the layout requires.
    fn ready_dir(root: &Path, name: &str, layout: ModelLayout) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(&dir).expect("model dir");
        for file in layout.required_files() {
            fs::write(dir.join(file), format!("{name}:{file}")).expect("model file");
        }
        dir
    }

    #[test]
    fn register_persist_reload_roundtrip() {
        let tmp = TempDir::new("roundtrip");
        let catalog_path = tmp.path().join("models.json");
        let dir = ready_dir(tmp.path(), "bge-small", ModelLayout::CandleBert);

        let mut catalog = ModelCatalog::open(&catalog_path).unwrap();
        assert!(catalog.is_empty());
        let entry = ModelEntry::new("bge-small", &dir, ModelLayout::CandleBert, 384)
            .with_source(ModelSource::HuggingFace {
                repo_id: "BAAI/bge-small-en-v1.5".to_string(),
            })
            .with_checksum("fnv1a64:deadbeefdeadbeef");
        catalog.register(entry.clone()).unwrap();
        assert_eq!(catalog.len(), 1);

        // A fresh handle sees byte-identical entries.
        let reopened = ModelCatalog::open(&catalog_path).unwrap();
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.get("bge-small"), Some(&entry));
        assert_eq!(
            reopened.resolve("bge-small").unwrap().dimension,
            384,
            "dimension survives the round-trip"
        );
        assert_eq!(
            reopened.resolve("bge-small").unwrap().model_path(),
            dir.join("model.safetensors")
        );
    }

    #[test]
    fn resolve_miss_is_explicit() {
        let tmp = TempDir::new("miss");
        let catalog = ModelCatalog::open(tmp.path().join("models.json")).unwrap();
        let err = catalog.resolve("nope").unwrap_err();
        assert!(matches!(err, EmbeddingError::ModelNotFound { name } if name == "nope"));
        // `resolve_ready` reports the same miss, not a readiness problem.
        assert!(matches!(
            catalog.resolve_ready("nope").unwrap_err(),
            EmbeddingError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn resolve_ready_lists_missing_files() {
        let tmp = TempDir::new("not-ready");
        let dir = ready_dir(tmp.path(), "half", ModelLayout::Onnx);
        fs::remove_file(dir.join("tokenizer.json")).unwrap();

        let mut catalog = ModelCatalog::open(tmp.path().join("models.json")).unwrap();
        catalog
            .register(ModelEntry::new("half", &dir, ModelLayout::Onnx, 768))
            .unwrap();

        // Registered → `resolve` succeeds (metadata is intact)…
        assert_eq!(catalog.resolve("half").unwrap().dimension, 768);
        // …but `resolve_ready` refuses to hand back an incomplete model.
        let err = catalog.resolve_ready("half").unwrap_err();
        match err {
            EmbeddingError::ModelNotReady { name, missing } => {
                assert_eq!(name, "half");
                assert_eq!(missing.len(), 1);
                assert!(missing[0].ends_with("tokenizer.json"), "got {missing:?}");
            }
            other => panic!("expected ModelNotReady, got {other:?}"),
        }
        assert!(!catalog.resolve("half").unwrap().is_ready());
    }

    #[test]
    fn malformed_catalog_is_an_error_not_an_empty_catalog() {
        let tmp = TempDir::new("malformed");
        let path = tmp.path().join("models.json");
        fs::write(&path, b"{not json").unwrap();
        let err = ModelCatalog::open(&path).unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));

        // A well-formed document from the future is rejected too.
        fs::write(&path, br#"{"version": 99, "entries": {}}"#).unwrap();
        let err = ModelCatalog::open(&path).unwrap_err();
        assert!(
            err.to_string().contains("format version 99"),
            "got {err}"
        );

        // An empty file is an empty catalog (a truncated write is impossible
        // thanks to the atomic rename, but a user-created placeholder is fine).
        fs::write(&path, b"").unwrap();
        assert!(ModelCatalog::open(&path).unwrap().is_empty());
    }

    #[test]
    fn empty_name_and_zero_dimension_are_rejected() {
        let tmp = TempDir::new("validate");
        let dir = ready_dir(tmp.path(), "m", ModelLayout::CandleBert);
        let mut catalog = ModelCatalog::open(tmp.path().join("models.json")).unwrap();

        let blank = ModelEntry::new("  ", &dir, ModelLayout::CandleBert, 384);
        assert!(matches!(
            catalog.register(blank).unwrap_err(),
            EmbeddingError::Config(_)
        ));
        let zero = ModelEntry::new("m", &dir, ModelLayout::CandleBert, 0);
        assert!(matches!(
            catalog.register(zero).unwrap_err(),
            EmbeddingError::Config(_)
        ));
        assert!(catalog.is_empty(), "nothing invalid was persisted");
    }

    #[test]
    fn unregister_removes_and_persists() {
        let tmp = TempDir::new("unregister");
        let path = tmp.path().join("models.json");
        let dir = ready_dir(tmp.path(), "m", ModelLayout::CandleBert);
        let mut catalog = ModelCatalog::open(&path).unwrap();
        catalog
            .register(ModelEntry::new("m", &dir, ModelLayout::CandleBert, 384))
            .unwrap();

        assert!(catalog.unregister("m").unwrap());
        assert!(!catalog.unregister("m").unwrap(), "second removal is a no-op");
        assert!(ModelCatalog::open(&path).unwrap().is_empty());
    }

    #[test]
    fn concurrent_registrations_do_not_corrupt_or_lose_entries() {
        const WRITERS: usize = 8;
        let tmp = TempDir::new("concurrent");
        let path = tmp.path().join("models.json");
        let mut dirs = Vec::new();
        for i in 0..WRITERS {
            dirs.push(ready_dir(tmp.path(), &format!("m{i}"), ModelLayout::Onnx));
        }
        // Materialize an empty catalog first so every writer starts from a file.
        ModelCatalog::open(&path).unwrap().save().unwrap();

        std::thread::scope(|scope| {
            for (i, dir) in dirs.iter().enumerate() {
                let path = path.clone();
                scope.spawn(move || {
                    let mut catalog = ModelCatalog::open(&path).unwrap();
                    catalog
                        .register(ModelEntry::new(
                            format!("m{i}"),
                            dir,
                            ModelLayout::Onnx,
                            768,
                        ))
                        .unwrap();
                });
            }
        });

        // The file still parses and holds every writer's entry.
        let catalog = ModelCatalog::open(&path).unwrap();
        assert_eq!(catalog.len(), WRITERS, "no lost updates");
        for i in 0..WRITERS {
            let entry = catalog.resolve(&format!("m{i}")).unwrap();
            assert_eq!(entry.dimension, 768);
        }
        // The lock file is released.
        assert!(!path.with_extension("lock").exists(), "lock removed on drop");
    }

    #[test]
    fn checksum_is_stable_and_content_sensitive() {
        let tmp = TempDir::new("checksum");
        let dir = ready_dir(tmp.path(), "m", ModelLayout::CandleBert);
        let entry = ModelEntry::new("m", &dir, ModelLayout::CandleBert, 384);

        let first = entry.compute_checksum().unwrap();
        assert!(first.starts_with("fnv1a64:"), "got {first}");
        assert_eq!(first, entry.compute_checksum().unwrap(), "stable");

        let stamped = entry.clone().with_checksum(first.clone());
        assert_eq!(stamped.verify_checksum().unwrap(), Some(true));
        assert_eq!(
            entry.verify_checksum().unwrap(),
            None,
            "no recorded checksum is reported, not assumed valid"
        );

        fs::write(entry.model_path(), b"tampered").unwrap();
        assert_eq!(stamped.verify_checksum().unwrap(), Some(false));
        assert_ne!(first, stamped.compute_checksum().unwrap());
    }

    #[test]
    fn checksum_of_missing_weights_errors() {
        let tmp = TempDir::new("checksum-missing");
        let entry = ModelEntry::new(
            "ghost",
            tmp.path().join("ghost"),
            ModelLayout::Onnx,
            384,
        );
        let err = entry.compute_checksum().unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
        assert!(err.to_string().contains("model.onnx"), "got {err}");
    }

    #[test]
    fn offline_prefetch_registers_metadata_and_is_idempotent() {
        let tmp = TempDir::new("prefetch");
        let path = tmp.path().join("models.json");
        let small = ready_dir(tmp.path(), "bge-small", ModelLayout::CandleBert);
        let onnx = ready_dir(tmp.path(), "mini", ModelLayout::Onnx);

        let specs = [
            ModelSpec::new("bge-small", &small, ModelLayout::CandleBert, 384).with_source(
                ModelSource::HuggingFace {
                    repo_id: "BAAI/bge-small-en-v1.5".to_string(),
                },
            ),
            ModelSpec::new("mini", &onnx, ModelLayout::Onnx, 384),
        ];

        let mut catalog = ModelCatalog::open(&path).unwrap();
        let report = catalog.prefetch(&specs, &LocalOnlyFetcher).unwrap();
        assert_eq!(report.registered, vec!["bge-small", "mini"]);
        assert!(report.skipped.is_empty());
        assert!(!report.is_empty());

        // Registration stamped a real checksum and recorded the origin.
        let entry = catalog.resolve_ready("bge-small").unwrap();
        assert_eq!(
            entry.source,
            ModelSource::HuggingFace {
                repo_id: "BAAI/bge-small-en-v1.5".to_string()
            }
        );
        assert_eq!(
            entry.verify_checksum().unwrap(),
            Some(true),
            "prefetch records a verifiable checksum"
        );

        // Re-running is a no-op that reports what it skipped.
        let report = catalog.prefetch(&specs, &LocalOnlyFetcher).unwrap();
        assert!(report.registered.is_empty());
        assert_eq!(report.skipped, vec!["bge-small", "mini"]);
        assert_eq!(ModelCatalog::open(&path).unwrap().len(), 2);
    }

    #[test]
    fn prefetch_failure_writes_nothing() {
        let tmp = TempDir::new("prefetch-fail");
        let path = tmp.path().join("models.json");
        let good = ready_dir(tmp.path(), "good", ModelLayout::CandleBert);
        let incomplete = ready_dir(tmp.path(), "incomplete", ModelLayout::Onnx);
        fs::remove_file(incomplete.join("model.onnx")).unwrap();

        let mut catalog = ModelCatalog::open(&path).unwrap();
        // 1. A directory that does not exist → the fetcher refuses.
        let missing = ModelSpec::new(
            "absent",
            tmp.path().join("nope"),
            ModelLayout::Onnx,
            384,
        );
        let err = catalog
            .prefetch(&[missing], &LocalOnlyFetcher)
            .unwrap_err();
        assert!(matches!(err, EmbeddingError::Config(_)));
        assert!(catalog.is_empty());

        // 2. A directory that exists but is incomplete → explicit error, and the
        //    earlier good spec in the same batch is *not* half-registered.
        let specs = [
            ModelSpec::new("good", &good, ModelLayout::CandleBert, 384),
            ModelSpec::new("incomplete", &incomplete, ModelLayout::Onnx, 384),
        ];
        let err = catalog
            .prefetch(&specs, &LocalOnlyFetcher)
            .unwrap_err();
        assert!(err.to_string().contains("incomplete"), "got {err}");
        assert!(
            catalog.is_empty(),
            "all-or-nothing: nothing was registered"
        );
        assert!(ModelCatalog::open(&path).unwrap().is_empty());
    }

    #[test]
    fn reload_picks_up_another_writers_entries() {
        let tmp = TempDir::new("reload");
        let path = tmp.path().join("models.json");
        let dir = ready_dir(tmp.path(), "m", ModelLayout::CandleBert);

        let mut first = ModelCatalog::open(&path).unwrap();
        let mut second = ModelCatalog::open(&path).unwrap();
        first
            .register(ModelEntry::new("m", &dir, ModelLayout::CandleBert, 384))
            .unwrap();

        assert!(second.is_empty(), "second handle has stale state");
        second.reload().unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second.resolve("m").unwrap().dimension, 384);
    }

    #[test]
    fn default_path_honors_env_override() {
        let _guard = crate::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(CATALOG_PATH_ENV);
        std::env::set_var(CATALOG_PATH_ENV, "D:/tmp/explicit-catalog.json");
        assert_eq!(
            ModelCatalog::default_path(),
            PathBuf::from("D:/tmp/explicit-catalog.json")
        );
        std::env::remove_var(CATALOG_PATH_ENV);
        let fallback = ModelCatalog::default_path();
        assert!(
            fallback.ends_with("catalog.json") || fallback.ends_with("lc-embeddings-models.json"),
            "unexpected default path {fallback:?}"
        );
        match previous {
            Some(v) => std::env::set_var(CATALOG_PATH_ENV, v),
            None => std::env::remove_var(CATALOG_PATH_ENV),
        }
    }

    #[test]
    fn entry_paths_follow_the_layout() {
        let candle = ModelEntry::new("c", "models/c", ModelLayout::CandleBert, 384);
        assert_eq!(
            candle.model_path().file_name().unwrap(),
            "model.safetensors"
        );
        assert_eq!(
            candle.tokenizer_path().file_name().unwrap(),
            "tokenizer.json"
        );
        assert_eq!(candle.config_path().unwrap().file_name().unwrap(), "config.json");
        assert_eq!(candle.model_path().parent().unwrap(), candle.dir);

        let onnx = ModelEntry::new("o", "models/o", ModelLayout::Onnx, 768);
        assert_eq!(onnx.model_path().file_name().unwrap(), "model.onnx");
        assert_eq!(onnx.config_path(), None);
    }
}
