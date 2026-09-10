//! Event store trait and the in-memory implementation (0.22.0 S4.2/S4.7).
//!
//! Idempotency contract: `append` with a `(session_id, branch, id)` key that
//! already exists is a **no-op success**, not an error — crash replays never
//! duplicate events.

use async_trait::async_trait;
use std::collections::BTreeMap;

use super::model::SessionEvent;
use crate::store::SessionError;

/// Append-only event storage. Keys are `(session_id, branch, id)`.
#[async_trait]
pub trait EventStore: Send + Sync {
    /// Appends one event. Idempotent on `(session_id, branch, id)`.
    async fn append(&self, event: &SessionEvent) -> Result<(), SessionError>;

    /// Appends a batch (atomicity per store implementation; the in-memory
    /// store is atomic under its lock).
    async fn append_batch(&self, events: &[SessionEvent]) -> Result<(), SessionError>;

    /// Reads events of one branch with `id > after`, id-ascending.
    /// `after = None` reads from the beginning.
    async fn read(
        &self,
        session_id: &str,
        branch: &str,
        after: Option<u64>,
    ) -> Result<Vec<SessionEvent>, SessionError>;

    /// Copies all events of `src_branch` with `id <= until_id` (None = all)
    /// into `dst_branch`, preserving ids and order. Returns the count copied.
    async fn fork(
        &self,
        session_id: &str,
        src_branch: &str,
        dst_branch: &str,
        until_id: Option<u64>,
    ) -> Result<usize, SessionError>;

    /// Highest event id on a branch (0 when the branch is empty) — callers
    /// derive the next id from it, which is what makes `append` idempotent.
    async fn latest_id(&self, session_id: &str, branch: &str) -> Result<u64, SessionError>;
}

/// In-memory [`EventStore`] — tests and single-process scenarios.
///
/// A `BTreeMap` keyed by `(session_id, branch, id)` gives ordered reads and
/// O(log n) idempotency checks. A single `tokio::Mutex` keeps
/// `append_batch`/`fork` atomic under concurrency.
pub struct MemoryEventStore {
    events: tokio::sync::Mutex<BTreeMap<(String, String, u64), SessionEvent>>,
}

impl MemoryEventStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            events: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    /// Number of events across all sessions/branches.
    pub async fn len(&self) -> usize {
        self.events.lock().await.len()
    }

    /// Whether the store holds no events.
    pub async fn is_empty(&self) -> bool {
        self.events.lock().await.is_empty()
    }
}

impl Default for MemoryEventStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryEventStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryEventStore").finish_non_exhaustive()
    }
}

fn key_of(e: &SessionEvent) -> (String, String, u64) {
    (e.session_id.clone(), e.branch.clone(), e.id)
}

#[async_trait]
impl EventStore for MemoryEventStore {
    async fn append(&self, event: &SessionEvent) -> Result<(), SessionError> {
        let mut events = self.events.lock().await;
        // Idempotency: existing (session, branch, id) is a no-op success so a
        // crash between "persist" and "ack" can be replayed safely.
        events.entry(key_of(event)).or_insert_with(|| event.clone());
        Ok(())
    }

    async fn append_batch(&self, batch: &[SessionEvent]) -> Result<(), SessionError> {
        let mut events = self.events.lock().await;
        for event in batch {
            events.entry(key_of(event)).or_insert_with(|| event.clone());
        }
        Ok(())
    }

    async fn read(
        &self,
        session_id: &str,
        branch: &str,
        after: Option<u64>,
    ) -> Result<Vec<SessionEvent>, SessionError> {
        let events = self.events.lock().await;
        Ok(events
            .range(
                (
                    session_id.to_string(),
                    branch.to_string(),
                    after.unwrap_or(0) + 1,
                )..=(session_id.to_string(), branch.to_string(), u64::MAX),
            )
            // A (session, branch) prefix ends where the session/branch name
            // stops matching; the upper bound trick above would also catch
            // other branches of the same session with lexicographically
            // larger names, so filter explicitly.
            .filter(|((s, b, _), _)| s == session_id && b == branch)
            .map(|(_, e)| e.clone())
            .collect())
    }

    async fn fork(
        &self,
        session_id: &str,
        src_branch: &str,
        dst_branch: &str,
        until_id: Option<u64>,
    ) -> Result<usize, SessionError> {
        let mut events = self.events.lock().await;
        let src: Vec<SessionEvent> = events
            .range(
                (session_id.to_string(), src_branch.to_string(), 0)
                    ..=(session_id.to_string(), src_branch.to_string(), u64::MAX),
            )
            .filter(|((s, b, _), _)| s == session_id && b == src_branch)
            .map(|(_, e)| e.clone())
            .collect();
        let mut copied = 0usize;
        for mut e in src {
            if let Some(until) = until_id {
                if e.id > until {
                    break;
                }
            }
            e.branch = dst_branch.to_string();
            let k = key_of(&e);
            if events.insert(k, e.clone()).is_none() {
                copied += 1;
            }
        }
        Ok(copied)
    }

    async fn latest_id(&self, session_id: &str, branch: &str) -> Result<u64, SessionError> {
        let events = self.events.lock().await;
        Ok(events
            .range(
                (session_id.to_string(), branch.to_string(), 0)
                    ..=(session_id.to_string(), branch.to_string(), u64::MAX),
            )
            .filter(|((s, b, _), _)| s == session_id && b == branch)
            .map(|((_, _, id), _)| *id)
            .next_back()
            .unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::model::EventPayload;

    fn ev(id: u64, turn: u64, payload: EventPayload) -> SessionEvent {
        SessionEvent::now(id, "s1", turn, "main", payload)
    }

    /// T1 append_read_roundtrip: read returns events id-ascending, equal content.
    #[tokio::test]
    async fn t1_append_read_roundtrip() {
        let store = MemoryEventStore::new();
        store
            .append(&ev(
                1,
                0,
                EventPayload::UserMessage {
                    content: "q".into(),
                },
            ))
            .await
            .unwrap();
        store
            .append(&ev(
                2,
                0,
                EventPayload::AssistantMessage {
                    content: "a".into(),
                },
            ))
            .await
            .unwrap();

        let read = store.read("s1", "main", None).await.unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].id, 1);
        assert_eq!(read[1].id, 2);
        assert_eq!(
            read[0].payload,
            EventPayload::UserMessage {
                content: "q".into()
            }
        );
    }

    /// T2 idempotent_append: same (session, branch, id) is a no-op success.
    #[tokio::test]
    async fn t2_idempotent_append() {
        let store = MemoryEventStore::new();
        let e = ev(
            1,
            0,
            EventPayload::UserMessage {
                content: "q".into(),
            },
        );
        store.append(&e).await.unwrap();
        store.append(&e).await.unwrap();
        assert_eq!(store.len().await, 1, "duplicate id must not duplicate");
        assert_eq!(store.latest_id("s1", "main").await.unwrap(), 1);
    }

    /// T7 crash_replay: a batch "half-written" (persisted before ack) can be
    /// replayed in full — duplicates are absorbed, ordering is intact.
    #[tokio::test]
    async fn t7_crash_replay() {
        let store = MemoryEventStore::new();
        let batch: Vec<SessionEvent> = (1..=5)
            .map(|i| {
                ev(
                    i,
                    0,
                    EventPayload::UserMessage {
                        content: format!("m{i}"),
                    },
                )
            })
            .collect();

        // Simulate: batch persisted up to id 3, then the process crashed
        // before acking; on restart the caller replays the whole batch.
        store.append_batch(&batch[..3]).await.unwrap();
        store.append_batch(&batch).await.unwrap();

        let read = store.read("s1", "main", None).await.unwrap();
        assert_eq!(read.len(), 5, "replay must not duplicate");
        let ids: Vec<u64> = read.iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5], "order preserved");
        assert_eq!(store.latest_id("s1", "main").await.unwrap(), 5);
    }

    /// Read `after` returns only strictly-later events (snapshot-then-tail).
    #[tokio::test]
    async fn read_after_excludes_prefix() {
        let store = MemoryEventStore::new();
        for i in 1..=4 {
            store
                .append(&ev(
                    i,
                    0,
                    EventPayload::UserMessage {
                        content: format!("m{i}"),
                    },
                ))
                .await
                .unwrap();
        }
        let tail = store.read("s1", "main", Some(2)).await.unwrap();
        let ids: Vec<u64> = tail.iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![3, 4]);
    }

    /// Branches of the same session are isolated (read/fork/latest per branch).
    #[tokio::test]
    async fn branches_are_isolated() {
        let store = MemoryEventStore::new();
        store
            .append(&ev(
                1,
                0,
                EventPayload::UserMessage {
                    content: "main".into(),
                },
            ))
            .await
            .unwrap();
        let mut forked = ev(
            1,
            0,
            EventPayload::UserMessage {
                content: "main".into(),
            },
        );
        forked.branch = "b1".into();
        store.append(&forked).await.unwrap();

        assert_eq!(store.read("s1", "main", None).await.unwrap().len(), 1);
        assert_eq!(store.read("s1", "b1", None).await.unwrap().len(), 1);
        assert_eq!(store.latest_id("s1", "b1").await.unwrap(), 1);
    }

    /// Empty branch: latest_id is 0, read is empty.
    #[tokio::test]
    async fn empty_branch_defaults() {
        let store = MemoryEventStore::new();
        assert_eq!(store.latest_id("s1", "main").await.unwrap(), 0);
        assert!(store.read("s1", "main", None).await.unwrap().is_empty());
    }

    /// T5 fork_copies_prefix: dst gets src's events up to until_id; further
    /// appends on either branch do not affect the other.
    #[tokio::test]
    async fn t5_fork_copies_prefix() {
        let store = MemoryEventStore::new();
        for i in 1..=4 {
            store
                .append(&ev(
                    i,
                    0,
                    EventPayload::UserMessage {
                        content: format!("m{i}"),
                    },
                ))
                .await
                .unwrap();
        }
        let copied = store.fork("s1", "main", "b1", Some(2)).await.unwrap();
        assert_eq!(copied, 2);

        let src = store.read("s1", "main", None).await.unwrap();
        let dst = store.read("s1", "b1", None).await.unwrap();
        assert_eq!(dst.len(), 2);
        assert_eq!(dst[0].id, src[0].id, "fork preserves ids");
        assert_eq!(dst[1].id, src[1].id);
        assert_eq!(dst[0].branch, "b1");

        // Post-fork appends diverge independently.
        store
            .append(&SessionEvent::now(
                5,
                "s1",
                1,
                "b1",
                EventPayload::UserMessage {
                    content: "fork-only".into(),
                },
            ))
            .await
            .unwrap();
        assert_eq!(store.read("s1", "main", None).await.unwrap().len(), 4);
        assert_eq!(store.read("s1", "b1", None).await.unwrap().len(), 3);
    }

    /// fork with until_id = None copies everything; forking into an existing
    /// branch is idempotent (no double copy).
    #[tokio::test]
    async fn fork_all_and_idempotent_refork() {
        let store = MemoryEventStore::new();
        for i in 1..=3 {
            store
                .append(&ev(
                    i,
                    0,
                    EventPayload::UserMessage {
                        content: format!("m{i}"),
                    },
                ))
                .await
                .unwrap();
        }
        assert_eq!(store.fork("s1", "main", "b1", None).await.unwrap(), 3);
        assert_eq!(store.fork("s1", "main", "b1", None).await.unwrap(), 0);
        assert_eq!(store.read("s1", "b1", None).await.unwrap().len(), 3);
    }
}
