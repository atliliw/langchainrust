// lc-memory/src/decay.rs
//! Memory decay & forgetting (C3, v0.22.1 §S4): TTL + importance-driven consolidation.
//!
//! Human memory fades; an agent's memory should too, or stale and weak memories
//! silently pollute every future prompt. C3 layers a lightweight decay policy on top of
//! [`super::file_memory::FileMemoryStore`]:
//!
//! - **Recency wins**: every access (remember/recall) stamps `last_access_at`. A memory
//!   that is read frequently stays alive; one that goes idle long enough is *forgotten*.
//! - **Importance floor**: each memory carries an `importance` in `[0, 1]`. Weak memories
//!   (`importance < min_importance`) that have gone quiet past `weak_grace` are pruned
//!   even before the full TTL — they are not earning their keep.
//! - **Explicit consolidation**: decay never happens on a hot read path. The caller runs
//!   [`ForgettingMemory::consolidate`] (e.g. between turns or on a schedule), which
//!   atomically drops expired/weak memories and persists the ledger.
//!
//! The time source is injected (`now: SystemTime`), making every rule pure and unit-testable
//! without sleeping. The metadata ledger is persisted as `.memory.json` inside the store
//! root, so importance and timestamps survive restarts.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use super::file_memory::{FileMemoryError, FileMemoryStore};

/// Ledger file name, kept inside the store root.
const LEDGER: &str = ".memory.json";

/// Decay policy configuration.
#[derive(Debug, Clone, Copy)]
pub struct ForgetConfig {
    /// A memory forgotten once idle this long (based on `last_access_at`).
    pub ttl: Duration,
    /// Contents importance in `[0, 1]`; below this floor counts as weak.
    pub min_importance: f64,
    /// A weak memory is pruned once it has been idle at least this long (even before TTL).
    pub weak_grace: Duration,
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30 * 24 * 3600), // 30 days
            min_importance: 0.5,
            weak_grace: Duration::from_secs(30 * 24 * 3600), // same as TTL by default
        }
    }
}

impl ForgetConfig {
    /// Explicit builder over the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the idle TTL after which a memory is forgotten.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Set the importance floor; below it a memory is weak.
    pub fn with_min_importance(mut self, min_importance: f64) -> Self {
        self.min_importance = min_importance;
        self
    }

    /// Set how long a weak memory may stay idle before consolidation prunes it.
    pub fn with_weak_grace(mut self, weak_grace: Duration) -> Self {
        self.weak_grace = weak_grace;
        self
    }
}

/// Per-memory decay metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryMeta {
    /// Importance in `[0, 1]`.
    importance: f64,
    /// Epoch milliseconds of first write.
    created_at: u64,
    /// Epoch milliseconds of most recent access (remember or recall).
    last_access_at: u64,
}

impl MemoryMeta {
    fn new(importance: f64, now: SystemTime) -> Self {
        Self {
            importance,
            created_at: epoch_ms(now),
            last_access_at: epoch_ms(now),
        }
    }
}

fn epoch_ms(t: SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn from_epoch_ms(ms: u64) -> SystemTime {
    std::time::UNIX_EPOCH + Duration::from_millis(ms)
}

/// A decaying, file-backed memory store.
pub struct ForgettingMemory {
    files: FileMemoryStore,
    config: ForgetConfig,
    meta: HashMap<String, MemoryMeta>,
}

impl std::fmt::Debug for ForgettingMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForgettingMemory")
            .field("root", &self.files.root())
            .field("config", &self.config)
            .field("live", &self.meta.len())
            .finish_non_exhaustive()
    }
}

impl ForgettingMemory {
    /// Opens the store root and loads any existing decay ledger.
    pub fn new(files: FileMemoryStore, config: ForgetConfig) -> Result<Self, FileMemoryError> {
        let meta = Self::load_ledger(files.root())?;
        Ok(Self {
            files,
            config,
            meta,
        })
    }

    /// Current decay policy.
    pub fn config(&self) -> ForgetConfig {
        self.config
    }

    fn ledger_path(root: &std::path::Path) -> PathBuf {
        root.join(LEDGER)
    }

    fn load_ledger(root: &std::path::Path) -> Result<HashMap<String, MemoryMeta>, FileMemoryError> {
        let path = Self::ledger_path(root);
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let raw = fs::read_to_string(&path)?;
        serde_json::from_str(&raw).map_err(|e| FileMemoryError::UnsafeName {
            name: LEDGER.to_string(),
            reason: format!("malformed ledger: {e}"),
        })
    }

    fn persist_ledger(&self) -> Result<(), FileMemoryError> {
        let path = Self::ledger_path(self.files.root());
        let raw =
            serde_json::to_string_pretty(&self.meta).map_err(|e| FileMemoryError::UnsafeName {
                name: LEDGER.to_string(),
                reason: format!("serialize ledger: {e}"),
            })?;
        fs::write(path, raw)?;
        Ok(())
    }

    /// Writes a new memory with the given importance. Re-remembering an existing name
    /// overwrites its content and refreshes its timestamps.
    pub fn remember(
        &mut self,
        name: &str,
        content: &str,
        importance: f64,
        now: SystemTime,
    ) -> Result<(), FileMemoryError> {
        if !(0.0..=1.0).contains(&importance) {
            return Err(FileMemoryError::UnsafeName {
                name: name.to_string(),
                reason: format!("importance {importance} outside [0, 1]"),
            });
        }
        self.files.write(name, content)?;
        self.meta
            .insert(name.to_string(), MemoryMeta::new(importance, now));
        self.persist_ledger()?;
        Ok(())
    }

    /// Reads a memory's content, refreshing its `last_access_at` (recency reward).
    pub fn recall(&mut self, name: &str, now: SystemTime) -> Result<String, FileMemoryError> {
        let content = self.files.view(name)?;
        if let Some(meta) = self.meta.get_mut(name) {
            meta.last_access_at = epoch_ms(now);
            self.persist_ledger()?;
        }
        Ok(content)
    }

    /// Importance of a memory, if tracked.
    pub fn importance(&self, name: &str) -> Option<f64> {
        self.meta.get(name).map(|m| m.importance)
    }

    /// Explicitly forgets one memory and its ledger entry.
    pub fn forget(&mut self, name: &str) -> Result<(), FileMemoryError> {
        self.files.delete(name)?;
        self.meta.remove(name);
        self.persist_ledger()?;
        Ok(())
    }

    /// Age of a memory's last access, else `None` if untracked / missing.
    fn idle(&self, name: &str, now: SystemTime) -> Option<Duration> {
        let meta = self.meta.get(name)?;
        let last = from_epoch_ms(meta.last_access_at);
        now.duration_since(last).ok()
    }

    /// Whether `name` should be pruned at `now`: either TTL-expired or weak-and-idle-grace.
    pub fn should_forget(&self, name: &str, now: SystemTime) -> bool {
        let Some(idle) = self.idle(name, now) else {
            return true; // untracked file is not a managed memory; consolidate prunes it
        };
        if idle >= self.config.ttl {
            return true;
        }
        let weak = self
            .meta
            .get(name)
            .is_none_or(|m| m.importance < self.config.min_importance);
        weak && idle >= self.config.weak_grace
    }

    /// Prunes every memory that should be forgotten at `now`. Returns the number pruned.
    ///
    /// Pure with respect to time (uses the injected `now`); the caller decides when to run it.
    pub fn consolidate(&mut self, now: SystemTime) -> Result<usize, FileMemoryError> {
        let doomed: Vec<String> = self
            .meta
            .keys()
            .filter(|name| self.should_forget(name, now))
            .cloned()
            .collect();
        let count = doomed.len();
        for name in doomed {
            let _ = self.files.delete(&name); // best-effort file removal
            self.meta.remove(&name);
        }
        if count > 0 {
            self.persist_ledger()?;
        }
        Ok(count)
    }

    /// Names of memories that currently survive `should_forget` at `now`.
    pub fn live_at(&self, now: SystemTime) -> Vec<String> {
        let mut live: Vec<String> = self
            .meta
            .keys()
            .filter(|name| !self.should_forget(name, now))
            .cloned()
            .collect();
        live.sort();
        live
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: std::time::SystemTime = std::time::UNIX_EPOCH;

    fn store_and_config(
        ttl_secs: u64,
        min_imp: f64,
        grace_secs: u64,
    ) -> (tempfile::TempDir, FileMemoryStore, ForgetConfig) {
        let dir = tempfile::tempdir().unwrap();
        let files = FileMemoryStore::new(dir.path()).unwrap();
        let config = ForgetConfig::new()
            .with_ttl(Duration::from_secs(ttl_secs))
            .with_min_importance(min_imp)
            .with_weak_grace(Duration::from_secs(grace_secs));
        (dir, files, config)
    }

    #[test]
    fn remember_recall_persists_content() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("site", "the docs live under /docs", 0.9, T0)
            .unwrap();
        assert_eq!(m.recall("site", T0).unwrap(), "the docs live under /docs");
        assert_eq!(m.importance("site"), Some(0.9));
    }

    #[test]
    fn active_memory_survives_before_ttl() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 0.9, T0).unwrap();
        let later = T0 + Duration::from_secs(90); // still < ttl
        assert!(!m.should_forget("a", later));
        assert_eq!(m.consolidate(later).unwrap(), 0);
    }

    #[test]
    fn idle_memory_forgotten_at_ttl() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 0.9, T0).unwrap();
        let past = T0 + Duration::from_secs(101);
        assert!(m.should_forget("a", past));
        assert_eq!(m.consolidate(past).unwrap(), 1);
        assert!(m.live_at(past).is_empty());
        // file removed too
        assert!(matches!(
            m.files.view("a"),
            Err(FileMemoryError::NotFound(_))
        ));
    }

    #[test]
    fn recall_refreshes_lifespan() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 0.9, T0).unwrap();
        // read again at t=80, pushing last_access forward
        let _ = m.recall("a", T0 + Duration::from_secs(80)).unwrap();
        // 150s after creation, but only 70s after the refresh -> survives
        let later = T0 + Duration::from_secs(150);
        assert!(!m.should_forget("a", later));
    }

    #[test]
    fn weak_memory_pruned_within_ttl_after_grace() {
        let (_d, files, cfg) = store_and_config(1000, 0.5, 50);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("weak", "trivia", 0.1, T0).unwrap();
        // within ttl, but past weak_grace and below importance floor -> pruned
        let later = T0 + Duration::from_secs(60);
        assert!(m.should_forget("weak", later));
        assert_eq!(m.consolidate(later).unwrap(), 1);
    }

    #[test]
    fn importance_out_of_range_rejected() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        assert!(m.remember("a", "x", 1.5, T0).is_err());
        assert!(m.remember("b", "x", -0.1, T0).is_err());
    }

    #[test]
    fn ledger_survives_restart() {
        let (dir, files, cfg) = store_and_config(1000, 0.5, 1000);
        {
            let mut m = ForgettingMemory::new(files, cfg).unwrap();
            m.remember("a", "content", 0.8, T0).unwrap();
        }
        // reopen on the same root: ledger is reloaded from disk
        let files2 = FileMemoryStore::new(dir.path()).unwrap();
        let m2 = ForgettingMemory::new(files2, cfg).unwrap();
        assert!(m2.live_at(T0).contains(&"a".to_string()));
        assert_eq!(m2.importance("a"), Some(0.8));
    }
}
