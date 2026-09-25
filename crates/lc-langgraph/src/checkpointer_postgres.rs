//! Postgres-backed checkpointer (B2, 0.22.4, `checkpoint-postgres` feature).
//!
//! Durable, multi-process persistence: many application instances can share
//! one Postgres database, each scoped to its own `thread_id`. Optimistic
//! concurrency is enforced storage-side by a conditional `UPDATE ... WHERE
//! version = $n`, so concurrent `update_state` calls against the same base
//! version cannot silently clobber each other even from different machines.
//!
//! Storage layout:
//!
//! ```sql
//! CREATE TABLE lc_checkpoints (
//!   seq             BIGSERIAL,                       -- global save order
//!   thread_id       TEXT NOT NULL,
//!   id              TEXT NOT NULL,                   -- UUID
//!   version         BIGINT NOT NULL,                 -- update_state OCC
//!   ts              BIGINT NOT NULL,                 -- unix seconds
//!   recursion_count BIGINT NOT NULL,
//!   state           TEXT NOT NULL,                   -- JSON of `S`
//!   PRIMARY KEY (thread_id, id)
//! );
//! ```
//!
//! The state blob is stored as `TEXT` (rather than `JSONB`) deliberately: it
//! keeps the checkpointer generic over any [`StateSchema`] without coupling
//! tokio-postgres's serde_json feature to a concrete JSON representation, and
//! matches the SQLite/Redis backends' opaque-blob storage.
//!
//! TLS is intentionally not pulled in. [`PostgresCheckpointer::connect`] uses
//! plain TCP; production deployments terminating TLS can build their own
//! configured `tokio_postgres::Client` and pass it to
//! [`PostgresCheckpointer::with_client`].

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls};
use uuid::Uuid;

use crate::checkpointer::{CheckpointInfo, Checkpointer};
use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS lc_checkpoints (
    seq             BIGSERIAL,
    thread_id       TEXT NOT NULL,
    id              TEXT NOT NULL,
    version         BIGINT NOT NULL,
    ts              BIGINT NOT NULL,
    recursion_count BIGINT NOT NULL,
    state           TEXT NOT NULL,
    parent_id       TEXT,
    PRIMARY KEY (thread_id, id)
);
CREATE INDEX IF NOT EXISTS idx_lc_checkpoints_thread
    ON lc_checkpoints(thread_id, ts, seq);
-- Idempotently add the fork-lineage column to databases created by older
-- versions (`CREATE TABLE IF NOT EXISTS` does not alter existing tables).
ALTER TABLE lc_checkpoints ADD COLUMN IF NOT EXISTS parent_id TEXT;
"#;

/// Checkpointer persisting checkpoints to a Postgres database.
pub struct PostgresCheckpointer<S: StateSchema> {
    // `Client::transaction` requires `&mut Client`; the mutex also serializes
    // the read-modify-write window of `update_state` within this process.
    // Cross-process serialization is enforced storage-side by the conditional
    // UPDATE's `WHERE version = $n` predicate.
    client: Arc<Mutex<Client>>,
    thread_id: String,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> PostgresCheckpointer<S> {
    /// Connects to Postgres using a tokio-postgres connection string
    /// (e.g. `host=localhost user=postgres dbname=lc` or a URI), creates the
    /// checkpoint table if needed, and returns a checkpointer scoped to
    /// `thread_id`.
    ///
    /// Uses plain TCP without TLS. For TLS, construct the `Client` yourself
    /// and use [`Self::with_client`].
    pub async fn connect(config: &str, thread_id: impl Into<String>) -> GraphResult<Self> {
        let (client, connection) = tokio_postgres::connect(config, NoTls)
            .await
            .map_err(|e| GraphError::CheckpointError(format!("postgres connect error: {e}")))?;
        // The connection future drives the protocol; it must run to completion.
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("postgres checkpointer connection error: {e}");
            }
        });
        Self::with_client(client, thread_id).await
    }

    /// Wraps an existing (possibly TLS-enabled) `tokio_postgres::Client`,
    /// creating the checkpoint table if needed.
    pub async fn with_client(client: Client, thread_id: impl Into<String>) -> GraphResult<Self> {
        let thread_id = thread_id.into();
        if thread_id.is_empty() {
            return Err(GraphError::CheckpointError(
                "thread_id must not be empty".to_string(),
            ));
        }
        client.batch_execute(SCHEMA).await.map_err(pg_err)?;
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            thread_id,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Thread (workflow/conversation) this checkpointer is scoped to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// A thread-bound backend only serves its own thread; the threaded trait
    /// methods assert the requested thread matches before delegating.
    fn assert_thread(&self, thread: &str) -> GraphResult<()> {
        if thread != self.thread_id {
            return Err(GraphError::CheckpointError(format!(
                "this checkpointer is scoped to thread '{}', not '{thread}'",
                self.thread_id
            )));
        }
        Ok(())
    }

    /// Shared insert path; `parent_id` records the checkpoint this one was
    /// forked from (fork lineage). Bound to the checkpointer's own thread.
    ///
    /// An inherent (not trait) helper so the trait impl can call it: it is a
    /// private detail of this backend, not a [`Checkpointer`] member.
    async fn insert_internal(
        &self,
        parent_id: Option<&str>,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        let id = Uuid::new_v4().to_string();
        let ts = chrono::Utc::now().timestamp();
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;
        let client = self.client.lock().await;
        client
            .execute(
                "INSERT INTO lc_checkpoints \
                 (thread_id, id, version, ts, recursion_count, state, parent_id) \
                 VALUES ($1, $2, 1, $3, $4, $5, $6)",
                &[
                    &self.thread_id,
                    &id,
                    &ts,
                    &(recursion_count as i64),
                    &state_json,
                    &parent_id,
                ],
            )
            .await
            .map_err(pg_err)?;
        Ok(id)
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for PostgresCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        self.insert_internal(None, state, recursion_count).await
    }

    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        self.assert_thread(thread)?;
        self.insert_internal(None, state, recursion_count).await
    }

    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        self.assert_thread(thread)?;
        self.insert_internal(parent_id, state, recursion_count)
            .await
    }

    async fn load_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<S> {
        self.assert_thread(thread)?;
        self.load(checkpoint_id).await
    }

    async fn list_threaded(&self, thread: &str) -> GraphResult<Vec<String>> {
        self.assert_thread(thread)?;
        self.list().await
    }

    async fn delete_threaded(&self, thread: &str, checkpoint_id: &str) -> GraphResult<()> {
        self.assert_thread(thread)?;
        self.delete(checkpoint_id).await
    }

    async fn last_threaded(&self, thread: &str) -> GraphResult<Option<(S, usize)>> {
        self.assert_thread(thread)?;
        self.last().await
    }

    async fn snapshots_threaded(&self, thread: &str) -> GraphResult<Vec<CheckpointInfo<S>>> {
        self.assert_thread(thread)?;
        self.snapshots().await
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT state FROM lc_checkpoints WHERE thread_id = $1 AND id = $2",
                &[&self.thread_id, &checkpoint_id],
            )
            .await
            .map_err(pg_err)?
            .ok_or_else(|| {
                GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
            })?;
        let state_json: String = row.get(0);
        serde_json::from_str(&state_json)
            .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "SELECT id FROM lc_checkpoints WHERE thread_id = $1 \
                 ORDER BY ts ASC, seq ASC",
                &[&self.thread_id],
            )
            .await
            .map_err(pg_err)?;
        Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
    }

    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "SELECT id, ts, seq, recursion_count, state, parent_id FROM lc_checkpoints \
                 WHERE thread_id = $1 ORDER BY ts ASC, seq ASC",
                &[&self.thread_id],
            )
            .await
            .map_err(pg_err)?;
        let mut snaps = Vec::with_capacity(rows.len());
        for row in &rows {
            let state_json: String = row.get(4);
            let state: S = serde_json::from_str(&state_json)
                .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))?;
            snaps.push(CheckpointInfo {
                id: row.get(0),
                timestamp: row.get(1),
                seq: row.get::<_, i64>(2) as u64,
                recursion_count: row.get::<_, i64>(3) as usize,
                state,
                parent: row.get(5),
            });
        }
        Ok(snaps)
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "DELETE FROM lc_checkpoints WHERE thread_id = $1 AND id = $2",
                &[&self.thread_id, &checkpoint_id],
            )
            .await
            .map_err(pg_err)?;
        Ok(())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT state, recursion_count FROM lc_checkpoints \
                 WHERE thread_id = $1 ORDER BY ts DESC, seq DESC LIMIT 1",
                &[&self.thread_id],
            )
            .await
            .map_err(pg_err)?;
        match row {
            Some(row) => {
                let state_json: String = row.get(0);
                let recursion_count: i64 = row.get(1);
                let state: S = serde_json::from_str(&state_json)
                    .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))?;
                Ok(Some((state, recursion_count as usize)))
            }
            None => Ok(None),
        }
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;

        let mut client = self.client.lock().await;
        let tx = client.transaction().await.map_err(pg_err)?;
        // B4: do NOT refresh `ts` — `last()` (`ORDER BY ts DESC`) must keep
        // reflecting save order, not edit order.
        let updated = tx
            .execute(
                "UPDATE lc_checkpoints SET state = $1, version = version + 1 \
                 WHERE thread_id = $2 AND id = $3 AND version = $4",
                &[
                    &state_json,
                    &self.thread_id,
                    &checkpoint_id,
                    &(expected_version as i64),
                ],
            )
            .await
            .map_err(pg_err)?;

        if updated == 0 {
            // Distinguish a stale version from a missing checkpoint.
            let current: Option<i64> = tx
                .query_opt(
                    "SELECT version FROM lc_checkpoints WHERE thread_id = $1 AND id = $2",
                    &[&self.thread_id, &checkpoint_id],
                )
                .await
                .map_err(pg_err)?
                .map(|row| row.get(0));
            tx.rollback().await.map_err(pg_err)?;
            return match current {
                Some(version) => Err(GraphError::CheckpointVersionConflict {
                    checkpoint_id: checkpoint_id.to_string(),
                    expected: expected_version,
                    actual: version as u64,
                }),
                None => Err(GraphError::CheckpointError(format!(
                    "Checkpoint '{checkpoint_id}' not found"
                ))),
            };
        }

        let new_version: i64 = tx
            .query_one(
                "SELECT version FROM lc_checkpoints WHERE thread_id = $1 AND id = $2",
                &[&self.thread_id, &checkpoint_id],
            )
            .await
            .map_err(pg_err)?
            .get(0);
        tx.commit().await.map_err(pg_err)?;
        Ok(new_version as u64)
    }
}

fn pg_err(e: tokio_postgres::Error) -> GraphError {
    GraphError::CheckpointError(format!("postgres error: {e}"))
}

#[cfg(test)]
mod tests {
    //! Live-server tests, ignored by default. Run with a reachable Postgres:
    //!
    //! ```text
    //! LANGCHAINRUST_TEST_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/lc_test \
    //! cargo test -p lc-langgraph --features checkpoint-postgres -- --ignored
    //! ```

    use super::*;
    use crate::state::AgentState;

    async fn test_client(thread: &str) -> Option<PostgresCheckpointer<AgentState>> {
        let Ok(url) = std::env::var("LANGCHAINRUST_TEST_POSTGRES_URL") else {
            return None;
        };
        // Unique thread per test run so repeated executions start clean.
        let thread = format!("{thread}-{}", Uuid::new_v4());
        Some(
            PostgresCheckpointer::<AgentState>::connect(&url, thread)
                .await
                .expect("connect"),
        )
    }

    #[tokio::test]
    #[ignore = "requires LANGCHAINRUST_TEST_POSTGRES_URL"]
    async fn postgres_roundtrip_and_thread_isolation() {
        let Some(cp) = test_client("rt").await else {
            return;
        };
        assert!(cp.list().await.unwrap().is_empty());
        let id1 = cp.save(&AgentState::new("one"), 1).await.unwrap();
        let id2 = cp.save(&AgentState::new("two"), 2).await.unwrap();
        assert_eq!(cp.load(&id1).await.unwrap().input, "one");
        assert_eq!(cp.list().await.unwrap(), vec![id1.clone(), id2.clone()]);
        let (last, recursion) = cp.last().await.unwrap().unwrap();
        assert_eq!(last.input, "two");
        assert_eq!(recursion, 2);
        cp.delete(&id1).await.unwrap();
        assert_eq!(cp.list().await.unwrap(), vec![id2]);
        assert!(cp.load(&id1).await.is_err());
    }

    #[tokio::test]
    #[ignore = "requires LANGCHAINRUST_TEST_POSTGRES_URL"]
    async fn postgres_update_state_occ() {
        let Some(cp) = test_client("occ").await else {
            return;
        };
        let id = cp.save(&AgentState::new("v1"), 0).await.unwrap();
        assert_eq!(
            cp.update_state(&id, &AgentState::new("v2"), 1)
                .await
                .unwrap(),
            2
        );
        let err = cp
            .update_state(&id, &AgentState::new("stale"), 1)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            GraphError::CheckpointVersionConflict {
                expected: 1,
                actual: 2,
                ..
            }
        ));
        assert!(cp
            .update_state("missing", &AgentState::new("x"), 1)
            .await
            .is_err());
    }
}
