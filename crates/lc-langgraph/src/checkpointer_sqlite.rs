//! SQLite-backed checkpointer (B2, 0.22.4, `checkpoint-sqlite` feature).
//!
//! Durable, embeddable persistence for graph state: the same SQLite file can
//! be reopened by another process (`WAL` journal + busy timeout), so a graph
//! run interrupted by a process exit can resume elsewhere. Checkpoints are
//! scoped to a `thread_id`, so one database file can host many independent
//! conversations/workflows.
//!
//! Storage layout (single shared table):
//!
//! ```sql
//! CREATE TABLE lc_checkpoints (
//!   seq             INTEGER PRIMARY KEY AUTOINCREMENT, -- global save order
//!   thread_id       TEXT NOT NULL,
//!   id              TEXT NOT NULL,                     -- UUID
//!   version         INTEGER NOT NULL,                  -- update_state OCC
//!   ts              INTEGER NOT NULL,                  -- unix seconds
//!   recursion_count INTEGER NOT NULL,
//!   state           TEXT NOT NULL,                     -- JSON of `S`
//!   UNIQUE(thread_id, id)
//! );
//! ```
//!
//! Blocking rusqlite calls run inside [`tokio::task::spawn_blocking`]; the
//! connection lives behind a process-local mutex (rusqlite `Connection` is
//! `Send` but not `Sync`).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use crate::checkpointer::Checkpointer;
use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;

/// Milliseconds SQLite waits on a locked database before erroring (covers the
/// brief WAL writer hand-off between separate processes).
const BUSY_TIMEOUT_MS: u32 = 5_000;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS lc_checkpoints (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       TEXT NOT NULL,
    id              TEXT NOT NULL,
    version         INTEGER NOT NULL,
    ts              INTEGER NOT NULL,
    recursion_count INTEGER NOT NULL,
    state           TEXT NOT NULL,
    UNIQUE(thread_id, id)
);
CREATE INDEX IF NOT EXISTS idx_lc_checkpoints_thread
    ON lc_checkpoints(thread_id, ts, seq);
"#;

/// Checkpointer persisting checkpoints to a SQLite database file.
pub struct SqliteCheckpointer<S: StateSchema> {
    conn: Arc<Mutex<Connection>>,
    thread_id: String,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> SqliteCheckpointer<S> {
    /// Opens (creating if needed) the SQLite database at `path` and returns a
    /// checkpointer scoped to `thread_id`.
    pub fn new(path: impl Into<PathBuf>, thread_id: impl Into<String>) -> GraphResult<Self> {
        let path = path.into();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                GraphError::CheckpointError(format!(
                    "failed to create checkpoint directory '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        let conn = Connection::open(&path).map_err(|e| {
            GraphError::CheckpointError(format!(
                "failed to open SQLite database '{}': {e}",
                path.display()
            ))
        })?;
        Self::init(conn, thread_id.into())
    }

    /// Creates an ephemeral in-memory checkpointer (each instance owns a
    /// private database; useful for tests).
    pub fn in_memory(thread_id: impl Into<String>) -> GraphResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| GraphError::CheckpointError(format!("failed to open :memory:: {e}")))?;
        Self::init(conn, thread_id.into())
    }

    fn init(conn: Connection, thread_id: String) -> GraphResult<Self> {
        if thread_id.is_empty() {
            return Err(GraphError::CheckpointError(
                "thread_id must not be empty".to_string(),
            ));
        }
        conn.busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS as u64))
            .map_err(sql_err)?;
        // WAL lets multiple processes (one writer at a time) share the file;
        // NORMAL sync is durable enough for checkpoints under WAL.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(sql_err)?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(sql_err)?;
        conn.execute_batch(SCHEMA).map_err(sql_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            thread_id,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Thread (workflow/conversation) this checkpointer is scoped to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// Runs one blocking rusqlite closure on the shared connection.
    async fn with_conn<T, F>(&self, f: F) -> GraphResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| {
                rusqlite::Error::ToSqlConversionFailure("sqlite connection mutex poisoned".into())
            })?;
            f(&guard)
        })
        .await
        .map_err(|e| GraphError::CheckpointError(format!("sqlite task join error: {e}")))?
        .map_err(sql_err)
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for SqliteCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        let id = Uuid::new_v4().to_string();
        let ts = chrono::Utc::now().timestamp();
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;
        let thread_id = self.thread_id.clone();
        let id_clone = id.clone();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO lc_checkpoints \
                 (thread_id, id, version, ts, recursion_count, state) \
                 VALUES (?1, ?2, 1, ?3, ?4, ?5)",
                params![thread_id, id_clone, ts, recursion_count as i64, state_json],
            )?;
            Ok(())
        })
        .await?;
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let thread_id = self.thread_id.clone();
        let id = checkpoint_id.to_string();
        let state_json: Option<String> = self
            .with_conn(move |conn| {
                conn.query_row(
                    "SELECT state FROM lc_checkpoints WHERE thread_id = ?1 AND id = ?2",
                    params![thread_id, id],
                    |row| row.get(0),
                )
                .optional()
            })
            .await?;
        decode_state(checkpoint_id, state_json)
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        let thread_id = self.thread_id.clone();
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id FROM lc_checkpoints WHERE thread_id = ?1 \
                 ORDER BY ts ASC, seq ASC",
            )?;
            let rows = stmt.query_map(params![thread_id], |row| row.get::<_, String>(0))?;
            rows.collect()
        })
        .await
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        let thread_id = self.thread_id.clone();
        let id = checkpoint_id.to_string();
        self.with_conn(move |conn| {
            conn.execute(
                "DELETE FROM lc_checkpoints WHERE thread_id = ?1 AND id = ?2",
                params![thread_id, id],
            )
        })
        .await?;
        Ok(())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let thread_id = self.thread_id.clone();
        let row: Option<(String, i64)> = self
            .with_conn(move |conn| {
                conn.query_row(
                    "SELECT state, recursion_count FROM lc_checkpoints \
                     WHERE thread_id = ?1 ORDER BY ts DESC, seq DESC LIMIT 1",
                    params![thread_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
            })
            .await?;
        match row {
            Some((state_json, recursion_count)) => {
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
        let thread_id = self.thread_id.clone();
        let id = checkpoint_id.to_string();
        let ts = chrono::Utc::now().timestamp();
        let outcome = self
            .with_conn(move |conn| {
                let tx = conn.unchecked_transaction()?;
                let changed = tx.execute(
                    "UPDATE lc_checkpoints SET state = ?1, version = version + 1, ts = ?2 \
                     WHERE thread_id = ?3 AND id = ?4 AND version = ?5",
                    params![state_json, ts, thread_id, id, expected_version as i64],
                )?;
                if changed == 0 {
                    // Distinguish a stale version from a missing checkpoint.
                    let current: Option<i64> = tx
                        .query_row(
                            "SELECT version FROM lc_checkpoints \
                             WHERE thread_id = ?1 AND id = ?2",
                            params![thread_id, id],
                            |row| row.get(0),
                        )
                        .optional()?;
                    return Ok(match current {
                        Some(version) => UpdateOutcome::Conflict(version as u64),
                        None => UpdateOutcome::Missing,
                    });
                }
                let new_version: i64 = tx.query_row(
                    "SELECT version FROM lc_checkpoints WHERE thread_id = ?1 AND id = ?2",
                    params![thread_id, id],
                    |row| row.get(0),
                )?;
                tx.commit()?;
                Ok(UpdateOutcome::Updated(new_version as u64))
            })
            .await?;
        match outcome {
            UpdateOutcome::Updated(version) => Ok(version),
            UpdateOutcome::Missing => Err(GraphError::CheckpointError(format!(
                "Checkpoint '{checkpoint_id}' not found"
            ))),
            UpdateOutcome::Conflict(actual) => Err(GraphError::CheckpointVersionConflict {
                checkpoint_id: checkpoint_id.to_string(),
                expected: expected_version,
                actual,
            }),
        }
    }
}

/// Storage-side result of a compare-and-set update.
enum UpdateOutcome {
    Updated(u64),
    /// No row matched (thread, checkpoint id).
    Missing,
    /// Row exists but its version differs.
    Conflict(u64),
}

fn decode_state<S: StateSchema>(checkpoint_id: &str, state_json: Option<String>) -> GraphResult<S> {
    let state_json = state_json.ok_or_else(|| {
        GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
    })?;
    serde_json::from_str(&state_json)
        .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))
}

fn sql_err(e: rusqlite::Error) -> GraphError {
    GraphError::CheckpointError(format!("sqlite error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AgentState;

    #[tokio::test]
    async fn save_load_list_last_delete_roundtrip() {
        let cp = SqliteCheckpointer::<AgentState>::in_memory("thread-a").unwrap();
        assert!(cp.list().await.unwrap().is_empty());
        assert!(cp.last().await.unwrap().is_none());

        let id1 = cp
            .save(&AgentState::new("one".to_string()), 1)
            .await
            .unwrap();
        let id2 = cp
            .save(&AgentState::new("two".to_string()), 2)
            .await
            .unwrap();

        assert_eq!(cp.load(&id1).await.unwrap().input, "one");
        let list = cp.list().await.unwrap();
        assert_eq!(list, vec![id1.clone(), id2.clone()]);
        let (last_state, recursion) = cp.last().await.unwrap().unwrap();
        assert_eq!(last_state.input, "two");
        assert_eq!(recursion, 2);

        cp.delete(&id1).await.unwrap();
        assert_eq!(cp.list().await.unwrap(), vec![id2]);
        assert!(cp.load(&id1).await.is_err());
    }

    #[tokio::test]
    async fn threads_are_isolated() {
        let cp_a = SqliteCheckpointer::<AgentState>::in_memory("a").unwrap();
        // Separate connections against the same *file* are exercised in the
        // cross-process integration test; here check constructor validation.
        assert!(SqliteCheckpointer::<AgentState>::in_memory("").is_err());
        let id = cp_a
            .save(&AgentState::new("for-a".to_string()), 0)
            .await
            .unwrap();
        assert_eq!(cp_a.load(&id).await.unwrap().input, "for-a");
    }

    #[tokio::test]
    async fn update_state_occ_conflict() {
        let cp = SqliteCheckpointer::<AgentState>::in_memory("t").unwrap();
        let id = cp
            .save(&AgentState::new("v1".to_string()), 0)
            .await
            .unwrap();
        let v = cp
            .update_state(&id, &AgentState::new("v2".to_string()), 1)
            .await
            .unwrap();
        assert_eq!(v, 2);
        let conflict = cp
            .update_state(&id, &AgentState::new("stale".to_string()), 1)
            .await
            .unwrap_err();
        assert!(
            matches!(
                conflict,
                GraphError::CheckpointVersionConflict {
                    expected: 1,
                    actual: 2,
                    ..
                }
            ),
            "expected version conflict, got {conflict:?}"
        );
        assert_eq!(cp.load(&id).await.unwrap().input, "v2");
    }

    /// B2 gate: a checkpointer reopened against the same database file (the
    /// in-process analogue of a different process after a crash/restart) sees
    /// the previous writer's checkpoints and recursion budget, and an
    /// `update_state` made by one writer causes a version conflict for the
    /// other.
    #[tokio::test]
    async fn cross_process_resume_state_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("graph.db");

        let id;
        {
            // "Process" 1: save a checkpoint and exit (drop the connection).
            let cp = SqliteCheckpointer::<AgentState>::new(&db_path, "session-42").unwrap();
            id = cp
                .save(&AgentState::new("mid-run".to_string()), 3)
                .await
                .unwrap();
            cp.update_state(&id, &AgentState::new("edited".to_string()), 1)
                .await
                .unwrap();
        }

        // "Process" 2: reopen the file, scoped to the same thread.
        let resumed = SqliteCheckpointer::<AgentState>::new(&db_path, "session-42").unwrap();
        let list = resumed.list().await.unwrap();
        assert_eq!(list, vec![id.clone()]);
        let loaded = resumed.load(&id).await.unwrap();
        assert_eq!(loaded.input, "edited");
        let (last_state, recursion_count) = resumed.last().await.unwrap().unwrap();
        assert_eq!(last_state.input, "edited");
        assert_eq!(recursion_count, 3, "recursion budget survives the restart");

        // The state was edited once before the restart, so its version is 2:
        // a stale edit based on version 1 must conflict (storage-side CAS).
        let err = resumed
            .update_state(&id, &AgentState::new("stale".to_string()), 1)
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
        let version = resumed
            .update_state(&id, &AgentState::new("post-restart".to_string()), 2)
            .await
            .unwrap();
        assert_eq!(version, 3);

        // A different thread sees an empty history even over the same file.
        let other_thread =
            SqliteCheckpointer::<AgentState>::new(&db_path, "other-session").unwrap();
        assert!(other_thread.list().await.unwrap().is_empty());
        assert!(other_thread.last().await.unwrap().is_none());
        assert!(other_thread.load(&id).await.is_err());
    }
}
