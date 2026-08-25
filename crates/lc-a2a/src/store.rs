//! Task persistence abstraction (P1-1).
//!
//! `A2AServer` talks to tasks exclusively through the [`TaskStore`] trait, so
//! the in-memory [`InMemoryTaskStore`] shipped here can be swapped for any
//! backend (database, Redis, file) without touching server logic.
//!
//! The trait intentionally returns owned snapshots: every read produces a
//! fresh [`StoredTask`] copy, so background workers and handlers never share
//! mutable references across `.await` points.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::protocol::{A2ATask, A2ATaskResult, TaskFilter};

/// A task snapshot stored by the server.
///
/// Wraps the protocol-visible [`A2ATask`] with server-only bookkeeping: the
/// terminal `result`/`error` payloads and `created_at`/`updated_at` timestamps
/// used for TTL expiry and LRU eviction (P1-2).
#[derive(Debug, Clone)]
pub struct StoredTask {
    /// The protocol-visible task.
    pub task: A2ATask,
    /// Result of the task (present when the task completed).
    pub result: Option<A2ATaskResult>,
    /// Error message (present when the task failed).
    pub error: Option<String>,
    /// W3C-style trace id carried on the request that created this task (P1-5).
    ///
    /// Server-only bookkeeping so a distributed trace can be correlated with a
    /// task after creation; the protocol-visible task itself does not expose it.
    pub trace_id: Option<String>,
    /// When the task was created.
    pub created_at: Instant,
    /// When the task was last modified.
    pub updated_at: Instant,
}

impl StoredTask {
    /// Wrap a task into a fresh stored snapshot.
    pub fn new(task: A2ATask) -> Self {
        let now = Instant::now();
        Self {
            task,
            result: None,
            error: None,
            trace_id: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Attach the trace id that created this task (P1-5).
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Mark the task as modified (bumps `updated_at`).
    pub fn touch(&mut self) {
        self.updated_at = Instant::now();
    }

    /// Age of this snapshot, measured from its last modification.
    pub fn age(&self) -> Duration {
        self.updated_at.elapsed()
    }
}

/// Error returned by a [`TaskStore`] backend.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    /// The backend is temporarily unavailable (e.g. connection loss).
    #[error("task store unavailable: {0}")]
    Unavailable(String),
    /// The store has reached its configured capacity.
    #[error("task store capacity exceeded: {0}")]
    CapacityExceeded(String),
}

/// Task persistence backend (P1-1).
///
/// The four operations mirror the A2A `tasks/*` surface: create/update
/// ([`upsert`](TaskStore::upsert)), read ([`get`](TaskStore::get)),
/// enumerate ([`list`](TaskStore::list)) and remove
/// ([`delete`](TaskStore::delete)). Implementations must be cheap under
/// concurrent access; the server does not hold the returned snapshot across
/// `.await` boundaries.
#[async_trait]
pub trait TaskStore: Send + Sync {
    /// Insert a new task or replace an existing one.
    async fn upsert(&self, stored: StoredTask) -> Result<(), StoreError>;

    /// Fetch a task snapshot by id, or `None` if absent.
    async fn get(&self, task_id: &str) -> Result<Option<StoredTask>, StoreError>;

    /// List task snapshots matching `filter`, ordered by creation (oldest first).
    async fn list(&self, filter: &TaskFilter) -> Result<Vec<StoredTask>, StoreError>;

    /// Delete a task by id. Returns `true` if a task was actually removed.
    async fn delete(&self, task_id: &str) -> Result<bool, StoreError>;
}

/// Default maximum number of tasks stored before LRU eviction.
pub const DEFAULT_MAX_TASKS: usize = 10_000;

/// In-memory [`TaskStore`] backed by a `RwLock<HashMap>`.
///
/// When at capacity and a *new* task id is inserted, the least recently
/// updated task is evicted (LRU). Re-inserting an existing id never evicts.
/// This is the default backend used by `A2AServer`.
#[derive(Debug, Clone)]
pub struct InMemoryTaskStore {
    inner: Arc<RwLock<HashMap<String, StoredTask>>>,
    max_tasks: usize,
}

impl InMemoryTaskStore {
    /// Create a store with the default capacity ([`DEFAULT_MAX_TASKS`]).
    pub fn new() -> Self {
        Self::with_max_tasks(DEFAULT_MAX_TASKS)
    }

    /// Create a store with an explicit capacity cap.
    pub fn with_max_tasks(max_tasks: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            max_tasks,
        }
    }
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn upsert(&self, stored: StoredTask) -> Result<(), StoreError> {
        // Atomic section: capacity check, LRU eviction and insert share one
        // write lock, so concurrent upserts cannot exceed `max_tasks` (the
        // previous check-then-act released the lock between the steps).
        let mut guard = self.inner.write().await;
        let inserting_new = !guard.contains_key(&stored.task.id);
        if inserting_new && self.max_tasks > 0 && guard.len() >= self.max_tasks {
            // Oldest-by-updated wins the LRU slot.
            let oldest_key = guard
                .iter()
                .min_by_key(|(_, t)| t.updated_at)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                guard.remove(&key);
            }
        }
        guard.insert(stored.task.id.clone(), stored);
        Ok(())
    }

    async fn get(&self, task_id: &str) -> Result<Option<StoredTask>, StoreError> {
        Ok(self.inner.read().await.get(task_id).cloned())
    }

    async fn list(&self, filter: &TaskFilter) -> Result<Vec<StoredTask>, StoreError> {
        let guard = self.inner.read().await;
        let mut out: Vec<StoredTask> = guard
            .values()
            .filter(|t| filter.matches(&t.task))
            .cloned()
            .collect();
        // Deterministic order: oldest created first.
        out.sort_by_key(|t| t.created_at);
        Ok(out)
    }

    async fn delete(&self, task_id: &str) -> Result<bool, StoreError> {
        Ok(self.inner.write().await.remove(task_id).is_some())
    }
}

/// Shared convenience: create a fresh in-memory store wrapped for trait use.
pub fn in_memory_store() -> Arc<dyn TaskStore> {
    Arc::new(InMemoryTaskStore::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{A2AMessage, TaskStatus};

    fn sample_task(id: &str, status: TaskStatus) -> A2ATask {
        A2ATask::new(id, A2AMessage::user("hi")).with_status(status)
    }

    #[tokio::test]
    async fn upsert_get_roundtrip() {
        let store = InMemoryTaskStore::new();
        let mut stored = StoredTask::new(sample_task("t1", TaskStatus::Working));
        stored.result = Some(A2ATaskResult::new("done"));
        store.upsert(stored).await.unwrap();

        let got = store.get("t1").await.unwrap().expect("task present");
        assert_eq!(got.task.id, "t1");
        assert_eq!(got.result.as_ref().unwrap().output, "done");
        assert_eq!(got.task.status, TaskStatus::Working);
        assert_eq!(got.created_at, got.updated_at);
    }

    #[tokio::test]
    async fn upsert_updates_existing_in_place() {
        let store = InMemoryTaskStore::new();
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Submitted)))
            .await
            .unwrap();
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Completed)))
            .await
            .unwrap();

        let got = store.get("t1").await.unwrap().unwrap();
        assert_eq!(got.task.status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let store = InMemoryTaskStore::new();
        assert!(store.get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_filters_by_owner_and_status() {
        let store = InMemoryTaskStore::new();
        store
            .upsert(StoredTask::new(
                sample_task("t1", TaskStatus::Working).with_owner("a"),
            ))
            .await
            .unwrap();
        store
            .upsert(StoredTask::new(
                sample_task("t2", TaskStatus::Completed).with_owner("a"),
            ))
            .await
            .unwrap();
        store
            .upsert(StoredTask::new(
                sample_task("t3", TaskStatus::Working).with_owner("b"),
            ))
            .await
            .unwrap();

        let all = store.list(&TaskFilter::new()).await.unwrap();
        assert_eq!(all.len(), 3);

        let only_a = store
            .list(&TaskFilter::new().with_owner("a"))
            .await
            .unwrap();
        assert_eq!(only_a.len(), 2);

        let a_working = store
            .list(
                &TaskFilter::new()
                    .with_owner("a")
                    .with_statuses(vec![TaskStatus::Working]),
            )
            .await
            .unwrap();
        assert_eq!(a_working.len(), 1);
        assert_eq!(a_working[0].task.id, "t1");
    }

    #[tokio::test]
    async fn delete_removes_and_reports() {
        let store = InMemoryTaskStore::new();
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Submitted)))
            .await
            .unwrap();

        assert!(store.delete("t1").await.unwrap());
        assert!(!store.delete("t1").await.unwrap());
        assert!(store.get("t1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn evicts_oldest_when_full() {
        let store = InMemoryTaskStore::with_max_tasks(2);
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Submitted)))
            .await
            .unwrap();
        store
            .upsert(StoredTask::new(sample_task("t2", TaskStatus::Submitted)))
            .await
            .unwrap();
        // t3 is new → evicts oldest (t1).
        store
            .upsert(StoredTask::new(sample_task("t3", TaskStatus::Submitted)))
            .await
            .unwrap();

        assert!(store.get("t1").await.unwrap().is_none());
        assert!(store.get("t2").await.unwrap().is_some());
        assert!(store.get("t3").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn touch_bumps_updated_at() {
        let store = InMemoryTaskStore::new();
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Submitted)))
            .await
            .unwrap();
        let mut stored = store.get("t1").await.unwrap().unwrap();
        stored.touch();
        assert!(stored.updated_at >= stored.created_at);
    }

    #[tokio::test]
    async fn store_is_clone_shareable() {
        let store = InMemoryTaskStore::new();
        let clone = store.clone();
        store
            .upsert(StoredTask::new(sample_task("t1", TaskStatus::Submitted)))
            .await
            .unwrap();
        assert!(clone.get("t1").await.unwrap().is_some());
    }
}
