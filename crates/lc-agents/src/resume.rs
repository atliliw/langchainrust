// lc-agents/src/resume.rs
//! Cross-process resume (§4.2 approval/budget gate): suspend-state persistence + recovery.
//!
//! The approval gate's [`ApprovalHandler`](crate::approval::ApprovalHandler)
//! `approve` is purely async — suspending a future to wait for an approval
//! signal works naturally within one process, but **a process death loses the
//! suspension point**: if the process is killed while waiting for approval, the
//! signal never arrives and the agent loop cannot continue. This module
//! serializes "the tool call awaiting approval + the context needed to resume
//! the agent loop" to disk, so a restarted process resumes from the checkpoint
//! instead of replaying the whole conversation.
//!
//! Division of labor:
//!
//! - [`PendingApproval`]: the pending tool + a snapshot of inputs /
//!   intermediate steps / iteration / budget totals, `Serialize + Deserialize`.
//! - [`ResumeStore`]: checkpoint persistence interface.
//! - [`FileResumeStore`]: disk implementation (JSON + atomic write), for real
//!   cross-process recovery.
//! - [`MemoryResumeStore`]: in-memory implementation, for tests / single-process
//!   demos.
//!
//! Framework integration point (see `executor/agent_loop.rs::execute_tool_inner`):
//! before each tool call enters the approval gate to wait for approval, the
//! [`PendingApproval`] is written to the store; it is cleared once the approval
//! decision is finalized. On a crash the checkpoint stays on disk; a new process
//! inspects it via [`AgentExecutor::pending_approval`] and resumes via
//! [`AgentExecutor::resume`].
//!
//! Recovery (process B):
//!
//! ```rust,ignore
//! // Process B: rebuild the same executor as process A (same agent / tools / store dir).
//! let store = Arc::new(FileResumeStore::new("/var/checkpoints/app")?);
//! let executor = AgentExecutor::new(agent, tools)
//!     .with_resume_store(store)
//!     .with_approval(handler);
//!
//! if let Some(pending) = executor.pending_approval().await? {
//!     // Show the operator pending.tool_name / pending.arguments and collect the decision.
//!     let answer = executor.resume(decision).await?;
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::types::AgentStep;

/// Cross-process resume error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ResumeError {
    /// Filesystem I/O error (create dir / read / write / delete).
    #[error("resume store I/O error: {0}")]
    Io(String),
    /// Checkpoint serialization / deserialization error.
    #[error("resume store serialization error: {0}")]
    Serialize(String),
}

/// The tool call awaiting approval + the context snapshot needed to resume the agent loop.
///
/// The framework persists it inside `execute_tool`, **before** calling the
/// approval handler (at that point the sync hooks have finished and `tool_name` /
/// `arguments` are the final values the approval actually sees); it is cleared
/// once the approval decision is finalized. On a process crash the checkpoint
/// stays on disk; a new process loads it and re-enters the approval flow (or
/// resumes directly with the given decision) instead of replaying the completed
/// intermediate steps from scratch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    /// Tool name awaiting approval (final value after sync hooks).
    pub tool_name: String,
    /// Tool arguments awaiting approval (JSON; final value after sync hooks).
    pub arguments: serde_json::Value,
    /// Tool call id (function-calling style; empty string if none).
    pub tool_id: String,
    /// Input variables to resume the agent loop (prompt variables, including `input`).
    pub inputs: HashMap<String, String>,
    /// Completed intermediate steps (all prior tool actions + observations). Recovery continues from here without replaying.
    pub steps: Vec<AgentStep>,
    /// Iteration number at suspension. Recovery continues from this iteration so the iteration budget stays unbroken.
    pub iteration: usize,
    /// Budget: tool calls consumed so far (including the pending tool).
    pub tool_calls_consumed: usize,
    /// Budget: LLM tokens consumed (None if the agent does not report).
    pub tokens_consumed: Option<usize>,
    /// Original run trace_id (reused after recovery to keep tracing continuous).
    pub trace_id: Option<String>,
}

/// Checkpoint storage interface.
///
/// The framework calls [`save_pending`](Self::save_pending) /
/// [`clear_pending`](Self::clear_pending) around the approval handler; a new
/// process reads the checkpoint via [`load_pending`](Self::load_pending).
/// Implementations must be `Send + Sync` (shared across tasks / processes).
#[async_trait]
pub trait ResumeStore: Send + Sync {
    /// Persists a pending-approval checkpoint.
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError>;
    /// Reads the current checkpoint; returns `None` if absent.
    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError>;
    /// Clears the checkpoint (decision finalized / checkpoint claimed). Treats absence as success.
    async fn clear_pending(&self) -> Result<(), ResumeError>;
}

/// Disk checkpoint storage: JSON persistence + atomic write.
///
/// Atomic write: write `pending.json.tmp` first, then `rename`, so a crash never
/// leaves a partial checkpoint. The directory is chosen by the caller;
/// **concurrent executors must use separate directories** to avoid overwriting
/// each other's checkpoints.
pub struct FileResumeStore {
    dir: PathBuf,
}

impl FileResumeStore {
    /// Creates a checkpoint store; auto-creates the directory (and parents) if missing.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, ResumeError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| ResumeError::Io(format!("create dir {}: {}", dir.display(), e)))?;
        Ok(Self { dir })
    }

    fn pending_path(&self) -> PathBuf {
        self.dir.join("pending.json")
    }
}

#[async_trait]
impl ResumeStore for FileResumeStore {
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError> {
        let bytes = serde_json::to_vec_pretty(pending)
            .map_err(|e| ResumeError::Serialize(e.to_string()))?;
        let tmp = self.dir.join("pending.json.tmp");
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|e| ResumeError::Io(format!("write {}: {}", tmp.display(), e)))?;
        tokio::fs::rename(&tmp, self.pending_path())
            .await
            .map_err(|e| ResumeError::Io(format!("rename {}: {}", tmp.display(), e)))?;
        Ok(())
    }

    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError> {
        let path = self.pending_path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(ResumeError::Io(format!("read {}: {}", path.display(), e)));
            }
        };
        let pending = serde_json::from_slice(&bytes)
            .map_err(|e| ResumeError::Serialize(format!("parse {}: {}", path.display(), e)))?;
        Ok(Some(pending))
    }

    async fn clear_pending(&self) -> Result<(), ResumeError> {
        let path = self.pending_path();
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            // Checkpoint already absent: idempotent clear, treated as success.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(ResumeError::Io(format!("remove {}: {}", path.display(), e))),
        }
    }
}

/// In-memory checkpoint store (for tests / single-process demos; not persistent across process death).
#[derive(Default)]
pub struct MemoryResumeStore {
    pending: tokio::sync::Mutex<Option<PendingApproval>>,
}

impl MemoryResumeStore {
    /// Creates an empty in-memory checkpoint store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ResumeStore for MemoryResumeStore {
    async fn save_pending(&self, pending: &PendingApproval) -> Result<(), ResumeError> {
        *self.pending.lock().await = Some(pending.clone());
        Ok(())
    }

    async fn load_pending(&self) -> Result<Option<PendingApproval>, ResumeError> {
        Ok(self.pending.lock().await.clone())
    }

    async fn clear_pending(&self) -> Result<(), ResumeError> {
        *self.pending.lock().await = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentAction, ToolInput};

    fn sample_pending() -> PendingApproval {
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), "compute".to_string());
        PendingApproval {
            tool_name: "calculator".to_string(),
            arguments: serde_json::json!({"expression": "2 + 3"}),
            tool_id: "call_1".to_string(),
            inputs,
            steps: vec![AgentStep::new(
                AgentAction {
                    tool: "other".to_string(),
                    tool_input: ToolInput::String {
                        value: "x".to_string(),
                    },
                    log: String::new(),
                },
                "obs".to_string(),
            )],
            iteration: 3,
            tool_calls_consumed: 4,
            tokens_consumed: Some(128),
            trace_id: Some("trace-1".to_string()),
        }
    }

    #[tokio::test]
    async fn test_file_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();

        // Empty directory: no checkpoint.
        assert!(store.load_pending().await.unwrap().is_none());

        let pending = sample_pending();
        store.save_pending(&pending).await.unwrap();
        let loaded = store.load_pending().await.unwrap().unwrap();
        assert_eq!(loaded.tool_name, "calculator");
        assert_eq!(loaded.arguments, serde_json::json!({"expression": "2 + 3"}));
        assert_eq!(loaded.iteration, 3);
        assert_eq!(loaded.tool_calls_consumed, 4);
        assert_eq!(loaded.tokens_consumed, Some(128));
        assert_eq!(loaded.inputs.get("input").unwrap(), "compute");
        assert_eq!(loaded.steps.len(), 1);
        assert_eq!(loaded.trace_id.as_deref(), Some("trace-1"));

        store.clear_pending().await.unwrap();
        assert!(store.load_pending().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_file_store_clear_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();
        // clear with no checkpoint does not error (idempotent).
        store.clear_pending().await.unwrap();
        store.clear_pending().await.unwrap();
    }

    #[tokio::test]
    async fn test_file_store_atomic_no_tmp_left() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileResumeStore::new(dir.path()).unwrap();
        store.save_pending(&sample_pending()).await.unwrap();

        // Atomic write: no .tmp leftover after a successful save.
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !entries.iter().any(|n| n == "pending.json.tmp"),
            "tmp file should be renamed away, got {entries:?}"
        );
        assert!(entries.contains(&"pending.json".to_string()));
    }

    #[tokio::test]
    async fn test_memory_store_roundtrip() {
        let store = MemoryResumeStore::new();
        assert!(store.load_pending().await.unwrap().is_none());

        store.save_pending(&sample_pending()).await.unwrap();
        assert!(store.load_pending().await.unwrap().is_some());

        store.clear_pending().await.unwrap();
        assert!(store.load_pending().await.unwrap().is_none());
    }

    #[test]
    fn test_pending_approval_serde_roundtrip() {
        let pending = sample_pending();
        let bytes = serde_json::to_vec(&pending).unwrap();
        let back: PendingApproval = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.tool_name, pending.tool_name);
        assert_eq!(back.arguments, pending.arguments);
        assert_eq!(back.steps.len(), pending.steps.len());
        assert_eq!(back.trace_id, pending.trace_id);
    }

    #[test]
    fn test_resume_error_display() {
        let e = ResumeError::Io("disk full".to_string());
        assert!(e.to_string().contains("disk full"));
        let e = ResumeError::Serialize("bad json".to_string());
        assert!(e.to_string().contains("bad json"));
    }

    /// Convenience assertion: `FileResumeStore::new` auto-creates the directory.
    #[tokio::test]
    async fn test_file_store_creates_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b");
        let store = FileResumeStore::new(&nested).unwrap();
        assert!(nested.is_dir());
        assert!(store.load_pending().await.is_ok());
    }
}
