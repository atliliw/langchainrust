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

use crate::checkpointer::{CheckpointInfo, Checkpointer};
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
    parent_id       TEXT,
    UNIQUE(thread_id, id)
);
CREATE INDEX IF NOT EXISTS idx_lc_checkpoints_thread
    ON lc_checkpoints(thread_id, ts, seq);
"#;

/// Idempotently add the `parent_id` (fork-lineage) column to a table written by
/// an older version. `CREATE TABLE IF NOT EXISTS` keeps new databases correct,
/// but existing files are not altered by it, so existing checkpoints must be
/// migrated in place. Runs on every open.
fn migrate_parent_column(conn: &Connection) -> GraphResult<()> {
    let has_parent: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('lc_checkpoints') \
             WHERE name = 'parent_id'",
            [],
            |row| row.get(0),
        )
        .map_err(sql_err)?;
    if has_parent == 0 {
        conn.execute_batch("ALTER TABLE lc_checkpoints ADD COLUMN parent_id TEXT")
            .map_err(sql_err)?;
    }
    Ok(())
}

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
        migrate_parent_column(&conn)?;
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

    /// Shared insert path; `parent_id` records the checkpoint this one was
    /// forked from (fork lineage).
    ///
    /// An inherent (not trait) helper so the trait impl can call it: it is a
    /// private detail of this backend, not a [`Checkpointer`] member.
    async fn insert_internal(
        &self,
        thread_id: String,
        parent_id: Option<String>,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        let id = Uuid::new_v4().to_string();
        let ts = chrono::Utc::now().timestamp();
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;
        let id_clone = id.clone();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO lc_checkpoints \
                 (thread_id, id, version, ts, recursion_count, state, parent_id) \
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)",
                params![
                    thread_id,
                    id_clone,
                    ts,
                    recursion_count as i64,
                    state_json,
                    parent_id,
                ],
            )?;
            Ok(())
        })
        .await?;
        Ok(id)
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for SqliteCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        self.insert_internal(self.thread_id.clone(), None, state, recursion_count)
            .await
    }

    async fn save_threaded(
        &self,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        self.assert_thread(thread)?;
        self.insert_internal(self.thread_id.clone(), None, state, recursion_count)
            .await
    }

    async fn save_fork_threaded(
        &self,
        parent_id: Option<&str>,
        thread: &str,
        state: &S,
        recursion_count: usize,
    ) -> GraphResult<String> {
        self.assert_thread(thread)?;
        self.insert_internal(
            self.thread_id.clone(),
            parent_id.map(ToOwned::to_owned),
            state,
            recursion_count,
        )
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

    async fn snapshots(&self) -> GraphResult<Vec<CheckpointInfo<S>>> {
        let thread_id = self.thread_id.clone();
        let rows: Vec<(String, i64, i64, i64, String, Option<String>)> = self
            .with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, ts, seq, recursion_count, state, parent_id FROM lc_checkpoints \
                     WHERE thread_id = ?1 ORDER BY ts ASC, seq ASC",
                )?;
                let rows = stmt.query_map(params![thread_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<String>>(5)?,
                    ))
                })?;
                rows.collect::<Result<Vec<_>, _>>()
            })
            .await?;
        let mut snaps = Vec::with_capacity(rows.len());
        for (id, ts, seq, recursion_count, state_json, parent) in rows {
            let state: S = serde_json::from_str(&state_json)
                .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))?;
            snaps.push(CheckpointInfo {
                id,
                timestamp: ts,
                seq: seq as u64,
                recursion_count: recursion_count as usize,
                state,
                parent,
            });
        }
        Ok(snaps)
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
        let outcome = self
            .with_conn(move |conn| {
                let tx = conn.unchecked_transaction()?;
                // B4: do NOT refresh `ts` — `last()` (`ORDER BY ts DESC`) must keep
                // reflecting save order, not edit order, so a state edit cannot
                // move a checkpoint ahead of later saves.
                let changed = tx.execute(
                    "UPDATE lc_checkpoints SET state = ?1, version = version + 1 \
                     WHERE thread_id = ?2 AND id = ?3 AND version = ?4",
                    params![state_json, thread_id, id, expected_version as i64],
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
    async fn snapshots_carry_real_ordering_keys_and_budget() {
        // #3: the durable backend must report true timestamp/seq/recursion_count
        // (not the trait fallback's zeros) so history ordering and fork budgets
        // are correct on sqlite, not just in memory.
        let cp = SqliteCheckpointer::<AgentState>::in_memory("thread-hist").unwrap();
        let id1 = cp
            .save(&AgentState::new("one".to_string()), 1)
            .await
            .unwrap();
        let id2 = cp
            .save(&AgentState::new("two".to_string()), 2)
            .await
            .unwrap();

        let snaps = cp.snapshots().await.unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].id, id1);
        assert_eq!(snaps[0].state.input, "one");
        assert_eq!(snaps[0].recursion_count, 1);
        assert_eq!(snaps[1].id, id2);
        assert_eq!(snaps[1].recursion_count, 2);
        // seq strictly increases (sqlite AUTOINCREMENT primary key).
        assert!(snaps[0].seq < snaps[1].seq);
        assert!(snaps[0].timestamp <= snaps[1].timestamp);
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

    #[tokio::test]
    async fn save_fork_records_parent_lineage() {
        let cp = SqliteCheckpointer::<AgentState>::in_memory("fork-lin").unwrap();
        let root = cp
            .save(&AgentState::new("root".to_string()), 1)
            .await
            .unwrap();
        let fork = cp
            .save_fork_threaded(
                Some(&root),
                "fork-lin",
                &AgentState::new("forked".to_string()),
                3,
            )
            .await
            .unwrap();

        let snaps = cp.snapshots().await.unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].id, root);
        assert_eq!(snaps[0].parent, None);
        assert_eq!(snaps[1].id, fork);
        assert_eq!(snaps[1].parent.as_deref(), Some(root.as_str()));

        // Threaded asserts: a mismatched thread is rejected.
        assert!(cp
            .save_threaded("other", &AgentState::new("x".to_string()), 0)
            .await
            .is_err());
        assert!(cp.load_threaded("other", &fork).await.is_err());
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
