//! Session checkpoint interface — durable execution (0.23.0 deliverable).
//!
//! Recovery semantics:
//! 1. `latest()` → the most recent snapshot `(up_to_id, projection)`;
//! 2. `EventStore::read(after = up_to_id)` → the tail to replay;
//! 3. idempotent `append` makes replaying the tail side-effect free;
//! 4. [`replay::merge_projections`](super::replay::merge_projections) folds the
//!    tail back into the snapshot.
//!
//! Implementations: [`NoopCheckpoint`] (full replay, always correct, the default)
//! and [`FileCheckpoint`] (durable, restart-surviving, no external service). The
//! trait is deliberately storage-agnostic — a Mongo/Redis-backed implementation
//! plugs in here with the same contract, and the recovery path in
//! [`EventSessionManager`](super::manager::EventSessionManager) does not change.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::replay::ProjectedSession;
use crate::store::SessionError;

/// Periodic materialization of a projected session, so recovery can skip a
/// full log replay. Persisted implementations own their storage.
#[async_trait]
pub trait SessionCheckpoint: Send + Sync {
    /// Materializes the projection up to (and including) `up_to_id`.
    async fn save(
        &self,
        session_id: &str,
        branch: &str,
        up_to_id: u64,
        projection: &ProjectedSession,
    ) -> Result<(), SessionError>;

    /// Returns the most recent snapshot for `(session_id, branch)`:
    /// `(up_to_id, projection)`, or `None` when none exists.
    async fn latest(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<Option<(u64, ProjectedSession)>, SessionError>;

    /// Drops the snapshot stored for `(session_id, branch)`.
    ///
    /// Called when a session is deleted: a snapshot left behind would otherwise
    /// resurrect the deleted session's history on the next recovery. The default
    /// implementation does nothing, which is correct for stateless
    /// implementations ([`NoopCheckpoint`]) and for stores whose snapshots share
    /// the session's own lifetime.
    async fn remove(&self, session_id: &str, branch: &str) -> Result<(), SessionError> {
        let _ = (session_id, branch);
        Ok(())
    }
}

/// No-op checkpoint: `latest()` always returns `None` (full replay), `save`
/// discards. Default wiring so callers can adopt the interface early.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopCheckpoint;

#[async_trait]
impl SessionCheckpoint for NoopCheckpoint {
    async fn save(
        &self,
        _session_id: &str,
        _branch: &str,
        _up_to_id: u64,
        _projection: &ProjectedSession,
    ) -> Result<(), SessionError> {
        Ok(())
    }

    async fn latest(
        &self,
        _session_id: &str,
        _branch: &str,
    ) -> Result<Option<(u64, ProjectedSession)>, SessionError> {
        Ok(None)
    }
}

/// On-disk checkpoint format version. Bumped when [`CheckpointRecord`] changes
/// shape; a record written by a *newer* version is refused rather than
/// misread.
const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Longest human-readable part kept from a session id / branch in a file name.
const MAX_STEM_CHARS: usize = 64;

/// The persisted form of one snapshot.
#[derive(Debug, Serialize, Deserialize)]
struct CheckpointRecord {
    /// Format version of this record (see [`CHECKPOINT_FORMAT_VERSION`]).
    version: u32,
    /// Highest event id folded into `projection`.
    up_to_id: u64,
    /// The materialized projection itself.
    projection: ProjectedSession,
}

/// Durable checkpoint backed by one JSON file per `(session_id, branch)`.
///
/// No external service: a restart re-reads the directory and recovery skips the
/// log replay. Writes are atomic — the record is written to a unique sibling
/// temp file and then `rename`d over the target, so a reader never observes a
/// half-written snapshot and a crash mid-write leaves only a stale `.tmp` file
/// behind (the previous snapshot stays valid).
///
/// The directory is created on first `save`; `latest` on a directory that does
/// not exist yet is simply `None`.
///
/// ```no_run
/// # use lc_sessions::events::checkpoint::{FileCheckpoint, SessionCheckpoint};
/// # async fn demo() -> Result<(), lc_sessions::store::SessionError> {
/// let cp = FileCheckpoint::new("/var/lib/myapp/session-checkpoints");
/// let saved = cp.latest("session-1", "main").await?;
/// # let _ = saved;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FileCheckpoint {
    dir: PathBuf,
}

impl FileCheckpoint {
    /// Creates a checkpoint rooted at `dir` (created lazily on first `save`).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory snapshots are stored in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file backing `(session_id, branch)`. Exposed so operators can inspect
    /// or back up a single snapshot.
    pub fn path_for(&self, session_id: &str, branch: &str) -> PathBuf {
        self.dir
            .join(format!("{}.json", file_stem(session_id, branch)))
    }

    /// A unique sibling temp path for `path`, so concurrent writers to the same
    /// key never share a scratch file (`rename` then decides the winner).
    fn tmp_path(path: &Path) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut name = path.file_name().unwrap_or_default().to_os_string();
        name.push(format!(".tmp-{}-{seq}", std::process::id()));
        path.with_file_name(name)
    }
}

#[async_trait]
impl SessionCheckpoint for FileCheckpoint {
    async fn save(
        &self,
        session_id: &str,
        branch: &str,
        up_to_id: u64,
        projection: &ProjectedSession,
    ) -> Result<(), SessionError> {
        let path = self.path_for(session_id, branch);
        let record = CheckpointRecord {
            version: CHECKPOINT_FORMAT_VERSION,
            up_to_id,
            projection: projection.clone(),
        };
        let bytes = serde_json::to_vec(&record).map_err(|e| {
            SessionError::StoreError(format!(
                "failed to encode checkpoint for {session_id}/{branch}: {e}"
            ))
        })?;
        fs::create_dir_all(&self.dir).map_err(|e| {
            SessionError::StoreError(format!(
                "failed to create checkpoint dir {}: {e}",
                self.dir.display()
            ))
        })?;
        let tmp = Self::tmp_path(&path);
        fs::write(&tmp, &bytes).map_err(|e| {
            SessionError::StoreError(format!("failed to write checkpoint {}: {e}", tmp.display()))
        })?;
        if let Err(e) = fs::rename(&tmp, &path) {
            // Best-effort cleanup: never leave scratch files behind on failure.
            let _ = fs::remove_file(&tmp);
            return Err(SessionError::StoreError(format!(
                "failed to commit checkpoint {}: {e}",
                path.display()
            )));
        }
        Ok(())
    }

    async fn latest(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<Option<(u64, ProjectedSession)>, SessionError> {
        let path = self.path_for(session_id, branch);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(SessionError::StoreError(format!(
                    "failed to read checkpoint {}: {e}",
                    path.display()
                )))
            }
        };
        let record: CheckpointRecord = serde_json::from_slice(&bytes).map_err(|e| {
            SessionError::StoreError(format!("corrupt checkpoint {}: {e}", path.display()))
        })?;
        if record.version > CHECKPOINT_FORMAT_VERSION {
            return Err(SessionError::StoreError(format!(
                "checkpoint {} uses format version {} but this build understands {}",
                path.display(),
                record.version,
                CHECKPOINT_FORMAT_VERSION
            )));
        }
        Ok(Some((record.up_to_id, record.projection)))
    }

    async fn remove(&self, session_id: &str, branch: &str) -> Result<(), SessionError> {
        let path = self.path_for(session_id, branch);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Idempotent: removing a snapshot that was never written is a no-op.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(SessionError::StoreError(format!(
                "failed to remove checkpoint {}: {e}",
                path.display()
            ))),
        }
    }
}

/// A filesystem-safe, collision-free stem for `(session_id, branch)`.
///
/// Session ids and branch names are caller-supplied, so they cannot be trusted
/// as path components: every character outside `[A-Za-z0-9._-]` is replaced by
/// `_`, which alone would let two distinct ids map to the same file. Appending a
/// hash of the *raw* value restores uniqueness, so sanitizing can never alias two
/// sessions onto one snapshot.
fn file_stem(session_id: &str, branch: &str) -> String {
    format!(
        "{}-{:016x}__{}-{:016x}",
        sanitize(session_id),
        hash64(session_id),
        sanitize(branch),
        hash64(branch)
    )
}

/// Keeps `[A-Za-z0-9._-]`, replaces everything else with `_`, truncates.
fn sanitize(raw: &str) -> String {
    let mut out: String = raw
        .chars()
        .take(MAX_STEM_CHARS)
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        // `.`/`..` would escape the directory; an empty name is unusable.
        out.push('_');
    }
    out
}

/// FNV-1a 64-bit — stable across runs (unlike `DefaultHasher`), so a snapshot
/// written by one process is found by the next.
fn hash64(raw: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    raw.as_bytes()
        .iter()
        .fold(OFFSET, |acc, b| (acc ^ u64::from(*b)).wrapping_mul(PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::model::{EventPayload, SessionEvent};
    use crate::events::replay::project;

    fn events(n: u64) -> Vec<SessionEvent> {
        let mut out = Vec::new();
        for turn in 0..n {
            out.push(SessionEvent::now(
                turn * 2 + 1,
                "s1",
                turn,
                "main",
                EventPayload::UserMessage {
                    content: format!("q{turn}"),
                },
            ));
            out.push(SessionEvent::now(
                turn * 2 + 2,
                "s1",
                turn,
                "main",
                EventPayload::AssistantMessage {
                    content: format!("a{turn}"),
                },
            ));
        }
        out
    }

    /// Unique scratch directory (no `tempfile` dependency in this crate).
    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lc-sessions-cp-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// NoopCheckpoint: save is a success, latest is always None → recovery
    /// degenerates to a full replay, which is always correct.
    #[tokio::test]
    async fn noop_checkpoint_semantics() {
        let cp = NoopCheckpoint;
        let events = vec![SessionEvent::now(
            1,
            "s1",
            0,
            "main",
            EventPayload::UserMessage {
                content: "hi".into(),
            },
        )];
        let p = crate::events::replay::project(&events).unwrap();
        cp.save("s1", "main", 1, &p).await.unwrap();
        assert!(cp.latest("s1", "main").await.unwrap().is_none());
    }

    /// T2: a snapshot survives a restart — a *second* `FileCheckpoint` over the
    /// same directory (nothing shared in memory) reads back exactly what the
    /// first one wrote.
    #[tokio::test]
    async fn file_checkpoint_survives_a_restart() {
        let dir = scratch();
        let p = project(&events(3)).unwrap();

        let writer = FileCheckpoint::new(&dir);
        writer.save("s1", "main", p.last_id, &p).await.unwrap();

        // Simulates a process restart: fresh handle, no shared state.
        let reader = FileCheckpoint::new(&dir);
        let (up_to_id, restored) = reader.latest("s1", "main").await.unwrap().unwrap();
        assert_eq!(up_to_id, p.last_id);
        assert_eq!(restored, p);

        fs::remove_dir_all(&dir).ok();
    }

    /// Recovery contract: the restored snapshot plus a replay of the tail is
    /// identical to projecting the whole log — which is what makes skipping the
    /// replay safe.
    #[tokio::test]
    async fn checkpoint_plus_tail_replay_equals_full_projection() {
        let dir = scratch();
        let all = events(4);
        let prefix = &all[..4]; // 2 turns
        let tail = &all[4..]; // 2 more turns

        let base = project(prefix).unwrap();
        FileCheckpoint::new(&dir)
            .save("s1", "main", base.last_id, &base)
            .await
            .unwrap();

        let (up_to_id, restored) = FileCheckpoint::new(&dir)
            .latest("s1", "main")
            .await
            .unwrap()
            .unwrap();
        let replayed_tail = project(&all[4..]).unwrap();
        assert_eq!(
            crate::events::replay::merge_projections(restored, replayed_tail),
            project(&all).unwrap(),
            "checkpoint({up_to_id}) + tail must equal the full projection"
        );

        // And the tail really was read back from disk, not from `base`.
        assert_eq!(up_to_id, base.last_id);
        assert!(!tail.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    /// Distinct keys never share a file — including keys that sanitize to the
    /// same readable prefix (the hash keeps them apart) and keys that would
    /// otherwise escape the directory.
    #[tokio::test]
    async fn distinct_keys_never_alias() {
        let cp = FileCheckpoint::new(scratch());
        let a = cp.path_for("s/1", "main");
        let b = cp.path_for("s?1", "main");
        assert_ne!(
            a, b,
            "keys sanitizing to the same prefix must stay distinct"
        );

        let traversal = cp.path_for("../../etc/passwd", "main");
        assert_eq!(
            traversal.parent(),
            Some(cp.dir()),
            "a traversal-shaped id must not escape the checkpoint dir"
        );

        // A branch is part of the key, not an alias of it.
        assert_ne!(cp.path_for("s1", "main"), cp.path_for("s1", "main2"));
    }

    /// A branch's snapshot is independent of the trunk's.
    #[tokio::test]
    async fn branches_are_isolated() {
        let dir = scratch();
        let cp = FileCheckpoint::new(&dir);
        let main = project(&events(1)).unwrap();
        let branch = project(&events(2)).unwrap();

        cp.save("s1", "main", main.last_id, &main).await.unwrap();
        cp.save("s1", "exp", branch.last_id, &branch).await.unwrap();

        let (main_id, main_p) = cp.latest("s1", "main").await.unwrap().unwrap();
        let (exp_id, exp_p) = cp.latest("s1", "exp").await.unwrap().unwrap();
        assert_eq!(main_p.turns.len(), 1);
        assert_eq!(exp_p.turns.len(), 2);
        assert!(exp_id > main_id);

        fs::remove_dir_all(&dir).ok();
    }

    /// Overwriting is atomic and leaves no scratch files behind: the temp file
    /// is renamed onto the target, so only one file per key exists.
    #[tokio::test]
    async fn save_overwrites_in_place_and_leaves_no_temp_files() {
        let dir = scratch();
        let cp = FileCheckpoint::new(&dir);
        cp.save("s1", "main", 1, &project(&events(1)).unwrap())
            .await
            .unwrap();
        cp.save("s1", "main", 2, &project(&events(2)).unwrap())
            .await
            .unwrap();

        let files: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            files.len(),
            1,
            "one file per key, no .tmp leftovers: {files:?}"
        );
        assert!(!files[0].contains(".tmp"));

        // The second write won.
        let (up_to_id, p) = cp.latest("s1", "main").await.unwrap().unwrap();
        assert_eq!(up_to_id, 2);
        assert_eq!(p.turns.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    /// An absent (or not-yet-created) directory is `None`, not an error.
    #[tokio::test]
    async fn missing_snapshot_is_none() {
        let root = scratch();
        let dir = root.join("never-created");
        let cp = FileCheckpoint::new(&dir);
        assert!(cp.latest("s1", "main").await.unwrap().is_none());
        // `save` creates the directory on demand.
        cp.save("s1", "main", 1, &project(&events(1)).unwrap())
            .await
            .unwrap();
        assert!(cp.latest("s1", "main").await.unwrap().is_some());
        fs::remove_dir_all(&root).ok();
    }

    /// A corrupt snapshot is an explicit error, not a silent full replay — a
    /// damaged checkpoint must be visible rather than quietly degrading.
    #[tokio::test]
    async fn corrupt_snapshot_is_an_error() {
        let dir = scratch();
        let cp = FileCheckpoint::new(&dir);
        fs::write(cp.path_for("s1", "main"), b"{not json").unwrap();
        let err = cp.latest("s1", "main").await.unwrap_err();
        assert!(
            err.to_string().contains("corrupt checkpoint"),
            "unexpected error: {err}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// A record from a newer format version is refused rather than misread.
    #[tokio::test]
    async fn future_format_version_is_refused() {
        let dir = scratch();
        let cp = FileCheckpoint::new(&dir);
        let p = project(&events(1)).unwrap();
        let record = serde_json::json!({
            "version": CHECKPOINT_FORMAT_VERSION + 1,
            "up_to_id": p.last_id,
            "projection": serde_json::to_value(&p).unwrap(),
        });
        fs::write(cp.path_for("s1", "main"), record.to_string()).unwrap();

        let err = cp.latest("s1", "main").await.unwrap_err();
        assert!(
            err.to_string().contains("format version"),
            "unexpected error: {err}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// `remove` drops the snapshot (so a deleted session cannot be resurrected
    /// from a stale checkpoint) and is idempotent.
    #[tokio::test]
    async fn remove_is_effective_and_idempotent() {
        let dir = scratch();
        let cp = FileCheckpoint::new(&dir);
        cp.save("s1", "main", 1, &project(&events(1)).unwrap())
            .await
            .unwrap();
        assert!(cp.latest("s1", "main").await.unwrap().is_some());

        cp.remove("s1", "main").await.unwrap();
        assert!(cp.latest("s1", "main").await.unwrap().is_none());
        // Removing again is a no-op, not an error.
        cp.remove("s1", "main").await.unwrap();

        fs::remove_dir_all(&dir).ok();
    }

    /// `hash64` is stable across calls (the property the file name relies on);
    /// `sanitize` never yields an empty or escaping component.
    #[test]
    fn stem_helpers_are_stable_and_safe() {
        assert_eq!(hash64("s1"), hash64("s1"));
        assert_ne!(hash64("s1"), hash64("s2"));
        assert_eq!(sanitize(""), "_");
        assert_eq!(sanitize(".."), ".."); // dots are kept, but the hash disambiguates
        assert_eq!(sanitize("a/b\\c:d"), "a_b_c_d");
        assert_eq!(sanitize(&"x".repeat(200)).len(), MAX_STEM_CHARS);
    }
}
