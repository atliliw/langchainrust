//! Redis-backed checkpointer (B2, 0.22.4, `checkpoint-redis` feature).
//!
//! Fast durable (when Redis persistence is enabled) checkpoint storage shared
//! by many processes. Keys belonging to one thread are pinned to one Redis
//! Cluster hash slot with the `{thread_id}` hashtag, so the layout works on
//! both standalone and clustered deployments.
//!
//! Key layout:
//!
//! ```text
//! lc:cp:{thread}:seq         STRING  monotonic per-thread save counter (INCR)
//! lc:cp:{thread}:index       ZSET    checkpoint id scored by save sequence
//! lc:cp:{thread}:cp:{id}     HASH    state / version / ts / recursion_count
//! ```
//!
//! The `update_state` compare-and-set runs as a single server-side Lua script
//! (`EVAL` is atomic in Redis): the version is checked and bumped without a
//! WATCH/MULTI round trip, so two concurrent editors on different machines
//! cannot both succeed against the same base version.

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Client};
use uuid::Uuid;

use crate::checkpointer::Checkpointer;
use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;

/// Atomic compare-and-set edit.
///
/// Returns `{1, new_version}` on success, `{0, current_version}` on a version
/// mismatch, `{-1, 0}` when the checkpoint hash does not exist.
const UPDATE_SCRIPT: &str = r#"
local key = KEYS[1]
if redis.call('EXISTS', key) == 0 then
    return {-1, 0}
end
local current = tonumber(redis.call('HGET', key, 'version'))
if current ~= tonumber(ARGV[1]) then
    return {0, current}
end
redis.call('HSET', key, 'state', ARGV[2], 'ts', ARGV[3])
local new_version = redis.call('HINCRBY', key, 'version', 1)
return {1, new_version}
"#;

/// Checkpointer persisting checkpoints to Redis.
pub struct RedisCheckpointer<S: StateSchema> {
    conn: MultiplexedConnection,
    thread_id: String,
    update_script: redis::Script,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateSchema> RedisCheckpointer<S> {
    /// Opens a multiplexed connection to `url`
    /// (e.g. `redis://127.0.0.1:6379/`) and returns a checkpointer scoped to
    /// `thread_id`.
    pub async fn connect(url: &str, thread_id: impl Into<String>) -> GraphResult<Self> {
        let client = Client::open(url)
            .map_err(|e| GraphError::CheckpointError(format!("invalid redis url: {e}")))?;
        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(redis_err)?;
        Self::with_connection(conn, thread_id).await
    }

    /// Wraps an existing multiplexed connection (e.g. one built with custom
    /// TLS or sentinel configuration).
    pub async fn with_connection(
        conn: MultiplexedConnection,
        thread_id: impl Into<String>,
    ) -> GraphResult<Self> {
        let thread_id = thread_id.into();
        if thread_id.is_empty() {
            return Err(GraphError::CheckpointError(
                "thread_id must not be empty".to_string(),
            ));
        }
        Ok(Self {
            conn,
            thread_id,
            update_script: redis::Script::new(UPDATE_SCRIPT),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Thread (workflow/conversation) this checkpointer is scoped to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    fn seq_key(&self) -> String {
        format!("lc:cp:{{{}}}:seq", self.thread_id)
    }

    fn index_key(&self) -> String {
        format!("lc:cp:{{{}}}:index", self.thread_id)
    }

    fn checkpoint_key(&self, id: &str) -> String {
        format!("lc:cp:{{{}}}:cp:{}", self.thread_id, id)
    }
}

#[async_trait]
impl<S: StateSchema> Checkpointer<S> for RedisCheckpointer<S> {
    async fn save(&self, state: &S, recursion_count: usize) -> GraphResult<String> {
        let id = Uuid::new_v4().to_string();
        let ts = chrono::Utc::now().timestamp();
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;

        let mut conn = self.conn.clone();
        let seq: i64 = conn.incr(self.seq_key(), 1_i64).await.map_err(redis_err)?;

        let key = self.checkpoint_key(&id);
        redis::pipe()
            .atomic()
            .hset(&key, "state", state_json)
            .hset(&key, "version", 1_i64)
            .hset(&key, "ts", ts)
            .hset(&key, "recursion_count", recursion_count as i64)
            .zadd(self.index_key(), seq, id.clone())
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(redis_err)?;
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> GraphResult<S> {
        let mut conn = self.conn.clone();
        let state_json: Option<String> = conn
            .hget(self.checkpoint_key(checkpoint_id), "state")
            .await
            .map_err(redis_err)?;
        let state_json = state_json.ok_or_else(|| {
            GraphError::CheckpointError(format!("Checkpoint '{checkpoint_id}' not found"))
        })?;
        serde_json::from_str(&state_json)
            .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))
    }

    async fn list(&self) -> GraphResult<Vec<String>> {
        let mut conn = self.conn.clone();
        conn.zrange(self.index_key(), 0, -1)
            .await
            .map_err(redis_err)
    }

    async fn delete(&self, checkpoint_id: &str) -> GraphResult<()> {
        let mut conn = self.conn.clone();
        redis::pipe()
            .atomic()
            .del(self.checkpoint_key(checkpoint_id))
            .zrem(self.index_key(), checkpoint_id)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(redis_err)?;
        Ok(())
    }

    async fn last(&self) -> GraphResult<Option<(S, usize)>> {
        let mut conn = self.conn.clone();
        let ids: Vec<String> = conn
            .zrange::<_, Vec<String>>(self.index_key(), -1, -1)
            .await
            .map_err(redis_err)?;
        let Some(id) = ids.into_iter().next() else {
            return Ok(None);
        };
        let key = self.checkpoint_key(&id);
        let (state_json, recursion_count): (String, i64) = redis::pipe()
            .hget(&key, "state")
            .hget(&key, "recursion_count")
            .query_async(&mut conn)
            .await
            .map_err(redis_err)?;
        let state: S = serde_json::from_str(&state_json)
            .map_err(|e| GraphError::CheckpointError(format!("deserialize error: {e}")))?;
        Ok(Some((state, recursion_count as usize)))
    }

    async fn update_state(
        &self,
        checkpoint_id: &str,
        state: &S,
        expected_version: u64,
    ) -> GraphResult<u64> {
        let state_json = serde_json::to_string(state)
            .map_err(|e| GraphError::CheckpointError(format!("serialize error: {e}")))?;
        let ts = chrono::Utc::now().timestamp().to_string();
        let mut conn = self.conn.clone();
        let (status, value): (i64, i64) = self
            .update_script
            .key(self.checkpoint_key(checkpoint_id))
            .arg(expected_version as i64)
            .arg(state_json)
            .arg(ts)
            .invoke_async(&mut conn)
            .await
            .map_err(redis_err)?;
        match status {
            1 => Ok(value as u64),
            0 => Err(GraphError::CheckpointVersionConflict {
                checkpoint_id: checkpoint_id.to_string(),
                expected: expected_version,
                actual: value as u64,
            }),
            _ => Err(GraphError::CheckpointError(format!(
                "Checkpoint '{checkpoint_id}' not found"
            ))),
        }
    }
}

fn redis_err(e: redis::RedisError) -> GraphError {
    GraphError::CheckpointError(format!("redis error: {e}"))
}

#[cfg(test)]
mod tests {
    //! Live-server tests, ignored by default. Run with a reachable Redis:
    //!
    //! ```text
    //! LANGCHAINRUST_TEST_REDIS_URL=redis://127.0.0.1:6379/ \
    //! cargo test -p lc-langgraph --features checkpoint-redis -- --ignored
    //! ```

    use super::*;
    use crate::state::AgentState;

    async fn test_client(thread: &str) -> Option<RedisCheckpointer<AgentState>> {
        let Ok(url) = std::env::var("LANGCHAINRUST_TEST_REDIS_URL") else {
            return None;
        };
        // Unique thread per run: also gives a fresh Redis Cluster hash tag.
        let thread = format!("{thread}-{}", Uuid::new_v4());
        Some(
            RedisCheckpointer::<AgentState>::connect(&url, thread)
                .await
                .expect("connect"),
        )
    }

    #[tokio::test]
    #[ignore = "requires LANGCHAINRUST_TEST_REDIS_URL"]
    async fn redis_roundtrip_and_ordering() {
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
    #[ignore = "requires LANGCHAINRUST_TEST_REDIS_URL"]
    async fn redis_update_state_occ() {
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
