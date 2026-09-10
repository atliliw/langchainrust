//! Session checkpoint interface — a *placeholder* for durable execution
//! (0.23.0 deliverable). 0.22.0 only fixes the interface and ships a no-op
//! default so callers can wire the dependency without behavior change.
//!
//! Recovery semantics (for the 0.23.0 implementation):
//! 1. `latest()` → the most recent snapshot `(up_to_id, projection)`;
//! 2. `EventStore::read(after = up_to_id)` → the tail to replay;
//! 3. idempotent `append` makes replaying the tail side-effect free.

use async_trait::async_trait;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::model::{EventPayload, SessionEvent};

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
}
