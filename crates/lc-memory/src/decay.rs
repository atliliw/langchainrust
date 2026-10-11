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
//! - **Spaced repetition** (T2): an optional [`ForgettingCurve`] replaces the TTL cliff
//!   with smooth Ebbinghaus decay — retention `0.5 ^ (idle / half_life)` — where every
//!   recall multiplies the half-life, so frequently retrieved memories become durable.
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

/// Smallest multiplier applied to a memory's initial half-life by importance:
/// an importance of `0.0` still gets half the base half-life, `1.0` gets 1.5×.
const IMPORTANCE_HALF_LIFE_FLOOR: f64 = 0.5;

/// Spaced-repetition forgetting curve (Ebbinghaus).
///
/// A TTL is a cliff: a memory is perfectly alive until the instant it is gone.
/// Human memory does not work that way — retention decays smoothly, and *recall
/// flattens the curve*. This models that directly:
///
/// - Each memory carries a **half-life**. Retention after `idle` time is
///   `0.5 ^ (idle / half_life)` — 1.0 at the moment of access, 0.5 one half-life
///   later, and so on.
/// - A memory is forgotten once retention falls below [`Self::retention_floor`].
/// - Every recall **multiplies the half-life by [`Self::reinforcement`]**, up to
///   [`Self::max_half_life`]. This is the spacing effect: material retrieved
///   repeatedly becomes durable, while material that is never touched decays on
///   its original schedule.
///
/// The initial half-life scales with importance (`0.5 + importance` ×
/// [`Self::base_half_life`]), so important memories start out more durable too.
///
/// [`ForgetConfig::ttl`] still applies as an absolute ceiling, and
/// [`ForgetConfig::min_importance`] / [`ForgetConfig::weak_grace`] still prune
/// weak memories independently of the curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForgettingCurve {
    /// Half-life of a brand-new memory with importance `0.5`.
    pub base_half_life: Duration,
    /// Factor the half-life is multiplied by on each recall.
    pub reinforcement: f64,
    /// Upper bound on the half-life, so a hot memory never becomes immortal.
    pub max_half_life: Duration,
    /// Retention in `[0, 1]` below which a memory is forgotten.
    pub retention_floor: f64,
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self {
            base_half_life: Duration::from_secs(24 * 3600), // 1 day
            reinforcement: 2.0,
            max_half_life: Duration::from_secs(90 * 24 * 3600), // 90 days
            retention_floor: 0.3,
        }
    }
}

impl ForgettingCurve {
    /// Explicit builder over the defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the half-life of a brand-new memory (importance `0.5`).
    pub fn with_base_half_life(mut self, base_half_life: Duration) -> Self {
        self.base_half_life = base_half_life;
        self
    }

    /// Sets the per-recall half-life multiplier (values `<= 1.0` disable the
    /// spacing effect; they are clamped to `1.0`).
    pub fn with_reinforcement(mut self, reinforcement: f64) -> Self {
        self.reinforcement = reinforcement.max(1.0);
        self
    }

    /// Sets the ceiling on the half-life.
    pub fn with_max_half_life(mut self, max_half_life: Duration) -> Self {
        self.max_half_life = max_half_life;
        self
    }

    /// Sets the retention floor below which a memory is forgotten.
    pub fn with_retention_floor(mut self, retention_floor: f64) -> Self {
        self.retention_floor = retention_floor.clamp(0.0, 1.0);
        self
    }

    /// Half-life for a newly written memory of the given importance.
    pub fn initial_half_life(&self, importance: f64) -> Duration {
        let factor = IMPORTANCE_HALF_LIFE_FLOOR + importance.clamp(0.0, 1.0);
        self.cap(self.base_half_life.mul_f64(factor))
    }

    /// Half-life after one more successful recall (the spacing effect).
    pub fn reinforced_half_life(&self, half_life: Duration) -> Duration {
        self.cap(half_life.mul_f64(self.reinforcement))
    }

    /// Retention in `[0, 1]` of a memory with `half_life` left idle for `idle`.
    ///
    /// `0.5 ^ (idle / half_life)` — 1.0 immediately after an access, 0.5 one
    /// half-life later, 0.25 after two.
    pub fn retention(&self, half_life: Duration, idle: Duration) -> f64 {
        let half = half_life.as_secs_f64();
        if half <= 0.0 {
            // Degenerate half-life: only a zero idle period is still retained.
            return if idle.is_zero() { 1.0 } else { 0.0 };
        }
        (-idle.as_secs_f64() / half * std::f64::consts::LN_2).exp()
    }

    fn cap(&self, half_life: Duration) -> Duration {
        half_life.min(self.max_half_life)
    }
}

/// Decay policy configuration.
#[derive(Debug, Clone, Copy)]
pub struct ForgetConfig {
    /// A memory forgotten once idle this long (based on `last_access_at`).
    pub ttl: Duration,
    /// Contents importance in `[0, 1]`; below this floor counts as weak.
    pub min_importance: f64,
    /// A weak memory is pruned once it has been idle at least this long (even before TTL).
    pub weak_grace: Duration,
    /// Optional spaced-repetition forgetting curve (T2). `None` (the default)
    /// keeps the original TTL-only policy.
    pub curve: Option<ForgettingCurve>,
}

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30 * 24 * 3600), // 30 days
            min_importance: 0.5,
            weak_grace: Duration::from_secs(30 * 24 * 3600), // same as TTL by default
            curve: None,
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

    /// Enables the spaced-repetition forgetting curve.
    pub fn with_curve(mut self, curve: ForgettingCurve) -> Self {
        self.curve = Some(curve);
        self
    }

    /// Disables the forgetting curve, returning to the TTL-only policy.
    pub fn without_curve(mut self) -> Self {
        self.curve = None;
        self
    }
}

/// Per-memory decay metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryMeta {
    /// Importance in `[0, 1]`.
    importance: f64,
    /// Epoch milliseconds of most recent access (remember or recall).
    last_access_at: u64,
    /// Current half-life in milliseconds, present only when a
    /// [`ForgettingCurve`] was configured at write time.
    ///
    /// `serde(default)` keeps ledgers written before the curve existed loadable;
    /// such entries simply have no curve information and fall back to the TTL
    /// rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    half_life_ms: Option<u64>,
}

impl MemoryMeta {
    fn new(importance: f64, now: SystemTime, half_life: Option<Duration>) -> Self {
        Self {
            importance,
            last_access_at: epoch_ms(now),
            half_life_ms: half_life.map(|d| d.as_millis() as u64),
        }
    }

    /// Current half-life, if this memory is curve-tracked.
    fn half_life(&self) -> Option<Duration> {
        self.half_life_ms.map(Duration::from_millis)
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
        // T2: with a curve configured, a new memory starts with a half-life
        // derived from its importance; a re-remembered name starts fresh.
        let half_life = self
            .config
            .curve
            .map(|curve| curve.initial_half_life(importance));
        self.meta.insert(
            name.to_string(),
            MemoryMeta::new(importance, now, half_life),
        );
        self.persist_ledger()?;
        Ok(())
    }

    /// Reads a memory's content, refreshing its `last_access_at` (recency reward).
    ///
    /// With a [`ForgettingCurve`] configured this is also the **reinforcement**
    /// step: the memory's half-life is multiplied by the curve's reinforcement
    /// factor, so repeatedly recalled memories decay more slowly (spacing effect).
    pub fn recall(&mut self, name: &str, now: SystemTime) -> Result<String, FileMemoryError> {
        let content = self.files.view(name)?;
        let curve = self.config.curve;
        if let Some(meta) = self.meta.get_mut(name) {
            meta.last_access_at = epoch_ms(now);
            if let (Some(curve), Some(half_life)) = (curve, meta.half_life()) {
                meta.half_life_ms = Some(curve.reinforced_half_life(half_life).as_millis() as u64);
            }
            self.persist_ledger()?;
        }
        Ok(content)
    }

    /// Importance of a memory, if tracked.
    pub fn importance(&self, name: &str) -> Option<f64> {
        self.meta.get(name).map(|m| m.importance)
    }

    /// Current half-life of a memory, if it is tracked by a [`ForgettingCurve`].
    ///
    /// `None` when the memory is untracked, or was written while no curve was
    /// configured (including entries loaded from a pre-curve ledger).
    pub fn half_life(&self, name: &str) -> Option<Duration> {
        self.meta.get(name).and_then(|m| m.half_life())
    }

    /// Current retention in `[0, 1]` of a memory at `now`, per the configured
    /// [`ForgettingCurve`].
    ///
    /// `None` when there is no curve information for the memory — either the
    /// store has no curve configured, or the entry predates it. Callers that
    /// rank by retention can treat `None` as fully retained (`unwrap_or(1.0)`).
    ///
    /// A clock rollback (`now` earlier than the recorded access) reports `1.0`
    /// rather than a decayed value, matching [`Self::should_forget`]'s L9 rule.
    pub fn retention(&self, name: &str, now: SystemTime) -> Option<f64> {
        let curve = self.config.curve?;
        let meta = self.meta.get(name)?;
        let half_life = meta.half_life()?;
        let idle = now
            .duration_since(from_epoch_ms(meta.last_access_at))
            .unwrap_or_default();
        Some(curve.retention(half_life, idle))
    }

    /// Explicitly forgets one memory and its ledger entry.
    pub fn forget(&mut self, name: &str) -> Result<(), FileMemoryError> {
        self.files.delete(name)?;
        self.meta.remove(name);
        self.persist_ledger()?;
        Ok(())
    }

    /// Whether `name` should be pruned at `now`.
    ///
    /// Three independent rules, any of which forgets the memory:
    ///
    /// 1. **TTL ceiling** — idle at least [`ForgetConfig::ttl`]. Applies whether or
    ///    not a curve is configured.
    /// 2. **Forgetting curve** (T2, only when [`ForgetConfig::curve`] is set and the
    ///    entry carries a half-life) — retention `0.5 ^ (idle / half_life)` has
    ///    fallen below the curve's `retention_floor`.
    /// 3. **Weak-and-idle** — importance below [`ForgetConfig::min_importance`] and
    ///    idle at least [`ForgetConfig::weak_grace`].
    ///
    /// Clock rollback (a `now` earlier than the recorded `last_access_at`) is **not** treated as
    /// forgettable — we cannot prove the memory has been idle, so we keep it alive (L9). Only a
    /// memory genuinely missing from the ledger is pruned as unmanaged.
    pub fn should_forget(&self, name: &str, now: SystemTime) -> bool {
        let Some(meta) = self.meta.get(name) else {
            return true; // untracked file is not a managed memory; consolidate prunes it
        };
        let last = from_epoch_ms(meta.last_access_at);
        // `duration_since` errors exactly when the clock went backwards (now < last); that must
        // not mark a recently-accessed memory as expiring.
        let Ok(idle) = now.duration_since(last) else {
            return false;
        };
        if idle >= self.config.ttl {
            return true;
        }
        if let Some(curve) = self.config.curve {
            if let Some(half_life) = meta.half_life() {
                if curve.retention(half_life, idle) < curve.retention_floor {
                    return true;
                }
            }
        }
        let weak = meta.importance < self.config.min_importance;
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

    /// L9: a system clock rollback (`now` < recorded `last_access_at`) must NOT make a
    /// memory look TTL-expired — `duration_since` errors backwards, and that error is now
    /// treated as "recently accessed", not "forgettable".
    #[test]
    fn clock_rollback_does_not_forget() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 0.9, T0).unwrap();
        // remember stamped `last_access_at = T0`; now queries at a time BEFORE T0.
        let rollback = T0 - Duration::from_secs(500);
        assert!(!m.should_forget("a", rollback));
        assert_eq!(m.consolidate(rollback).unwrap(), 0);
        assert!(m.live_at(rollback).contains(&"a".to_string()));
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

    // ---- T2: spaced-repetition forgetting curve ----

    /// TTL ceiling kept far away so the curve is the only thing deciding.
    fn curve_config(curve: ForgettingCurve) -> ForgetConfig {
        ForgetConfig::new()
            .with_ttl(Duration::from_secs(100_000))
            .with_min_importance(0.0) // disable the weak-and-idle rule
            .with_curve(curve)
    }

    #[test]
    fn curve_retention_decays_by_half_lives() {
        let curve = ForgettingCurve::new().with_base_half_life(Duration::from_secs(100));
        let half = Duration::from_secs(100);

        assert!((curve.retention(half, Duration::ZERO) - 1.0).abs() < 1e-9);
        assert!((curve.retention(half, half) - 0.5).abs() < 1e-9);
        assert!((curve.retention(half, half * 2) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn curve_forgets_once_retention_falls_below_floor() {
        // base 100s at importance 0.5 -> initial half-life 100s; floor 0.3 is
        // crossed between one and two half-lives (0.5 alive, 0.25 forgotten).
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(100))
            .with_retention_floor(0.3);
        let (_d, files, _) = store_and_config(100_000, 0.0, 100_000);
        let mut m = ForgettingMemory::new(files, curve_config(curve)).unwrap();
        m.remember("a", "x", 0.5, T0).unwrap();

        assert!(!m.should_forget("a", T0 + Duration::from_secs(100)));
        assert!(m.should_forget("a", T0 + Duration::from_secs(200)));
        assert_eq!(m.consolidate(T0 + Duration::from_secs(200)).unwrap(), 1);
    }

    /// The spacing effect: a recall doubles the half-life, so a memory that would
    /// have decayed away survives — the whole point of the curve.
    #[test]
    fn recall_reinforces_the_curve() {
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(100))
            .with_reinforcement(2.0)
            .with_retention_floor(0.3);
        let (_d, files, _) = store_and_config(100_000, 0.0, 100_000);
        let mut m = ForgettingMemory::new(files, curve_config(curve)).unwrap();
        m.remember("a", "x", 0.5, T0).unwrap();
        assert_eq!(m.half_life("a"), Some(Duration::from_secs(100)));

        // Recall at t=50 refreshes the clock AND doubles the half-life to 200s.
        m.recall("a", T0 + Duration::from_secs(50)).unwrap();
        assert_eq!(m.half_life("a"), Some(Duration::from_secs(200)));

        // 200s idle after the recall: retention 0.5, still above the floor. Without
        // the reinforcement the same idle span would be two half-lives (0.25).
        let later = T0 + Duration::from_secs(250);
        assert!(!m.should_forget("a", later));
        assert!(m.retention("a", later).unwrap() > 0.49);
    }

    #[test]
    fn reinforcement_is_capped_by_max_half_life() {
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(60))
            .with_reinforcement(10.0)
            .with_max_half_life(Duration::from_secs(300));
        let (_d, files, _) = store_and_config(100_000, 0.0, 100_000);
        let mut m = ForgettingMemory::new(files, curve_config(curve)).unwrap();
        m.remember("a", "x", 0.5, T0).unwrap();

        for i in 0..5 {
            m.recall("a", T0 + Duration::from_secs(i)).unwrap();
        }
        assert_eq!(
            m.half_life("a"),
            Some(Duration::from_secs(300)),
            "half-life must stop at the cap"
        );
    }

    /// Importance feeds the initial half-life, so a high-importance memory starts
    /// out more durable than a low-importance one.
    #[test]
    fn importance_scales_initial_half_life() {
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(100))
            .with_retention_floor(0.3);
        let (_d, files, _) = store_and_config(100_000, 0.0, 100_000);
        let mut m = ForgettingMemory::new(files, curve_config(curve)).unwrap();
        m.remember("low", "x", 0.0, T0).unwrap();
        m.remember("high", "x", 1.0, T0).unwrap();

        assert_eq!(m.half_life("low"), Some(Duration::from_secs(50)));
        assert_eq!(m.half_life("high"), Some(Duration::from_secs(150)));
    }

    /// The TTL stays an absolute ceiling even with a long half-life: a memory can
    /// never outlive `ttl`, however often it is recalled.
    #[test]
    fn curve_never_escapes_the_ttl_ceiling() {
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(10_000))
            .with_retention_floor(0.0); // the curve alone would never forget
        let cfg = ForgetConfig::new()
            .with_ttl(Duration::from_secs(100))
            .with_min_importance(0.0)
            .with_curve(curve);
        let (_d, files, _) = store_and_config(100, 0.0, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 1.0, T0).unwrap();

        assert!(!m.should_forget("a", T0 + Duration::from_secs(99)));
        assert!(m.should_forget("a", T0 + Duration::from_secs(100)));
    }

    /// Without a curve the store keeps its original behaviour: no half-life is
    /// recorded and `retention` has nothing to report.
    #[test]
    fn no_curve_means_no_retention_tracking() {
        let (_d, files, cfg) = store_and_config(100, 0.4, 100);
        let mut m = ForgettingMemory::new(files, cfg).unwrap();
        m.remember("a", "x", 0.9, T0).unwrap();

        assert_eq!(m.half_life("a"), None);
        assert_eq!(m.retention("a", T0), None);
        assert!(!m.should_forget("a", T0 + Duration::from_secs(90)));
    }

    /// A ledger written before the curve existed has no `half_life_ms`; it must
    /// still load, and those entries fall back to the TTL rule.
    #[test]
    fn pre_curve_ledger_loads_and_falls_back_to_ttl() {
        let dir = tempfile::tempdir().unwrap();
        let files = FileMemoryStore::new(dir.path()).unwrap();
        files.write("old", "content").unwrap();
        std::fs::write(
            dir.path().join(LEDGER),
            r#"{"old":{"importance":0.9,"last_access_at":0}}"#,
        )
        .unwrap();

        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(10))
            .with_retention_floor(0.9);
        let cfg = ForgetConfig::new()
            .with_ttl(Duration::from_secs(1000))
            .with_min_importance(0.0)
            .with_curve(curve);
        let m = ForgettingMemory::new(files, cfg).unwrap();

        // The curve would have forgotten it long ago, but there is no half-life to
        // decay, so only the TTL applies.
        assert_eq!(m.half_life("old"), None);
        assert!(!m.should_forget("old", T0 + Duration::from_secs(500)));
        assert!(m.should_forget("old", T0 + Duration::from_secs(1000)));
    }

    #[test]
    fn curve_half_life_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let curve = ForgettingCurve::new()
            .with_base_half_life(Duration::from_secs(100))
            .with_reinforcement(3.0);
        {
            let files = FileMemoryStore::new(dir.path()).unwrap();
            let mut m = ForgettingMemory::new(files, curve_config(curve)).unwrap();
            m.remember("a", "x", 0.5, T0).unwrap();
            m.recall("a", T0 + Duration::from_secs(1)).unwrap();
        }
        let files = FileMemoryStore::new(dir.path()).unwrap();
        let m2 = ForgettingMemory::new(files, curve_config(curve)).unwrap();
        assert_eq!(
            m2.half_life("a"),
            Some(Duration::from_secs(300)),
            "the reinforced half-life must persist across restarts"
        );
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
