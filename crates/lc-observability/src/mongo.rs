// lc-observability/src/mongo.rs
//! MongoDB sink (feature `mongodb`).

use lc_core::observability::{MetricsSink, ObsError, ObsEvent};
use mongodb::bson::Document;
use mongodb::options::IndexOptions;
use mongodb::IndexModel;
use std::time::Duration;

/// Inserts each event as one document into the given collection. Records keep the
/// serde tag (`"kind"`), so a single collection can hold token-usage and agent
/// metrics side by side.
///
/// # TTL / index support (T2)
///
/// Unbounded growth was the default: nothing pruned a caller-named collection and
/// no index aided the `trace_id` join. [`MongoSink::with_expire_after_seconds`]
/// enables a TTL: every insert is stamped with a BSON `_recorded_at` date and
/// [`MongoSink::ensure_indexes`] adds a matching TTL index plus a sparse `trace_id`
/// index (join key across a run's cost/agent/error events). `ensure_indexes` is
/// idempotent and caller-invoked so it never blocks the first insert.
#[derive(Clone)]
pub struct MongoSink {
    collection: mongodb::Collection<Document>,
    /// Optional expiry (in seconds) for the TTL index on `_recorded_at`.
    ttl_seconds: Option<i64>,
}

impl MongoSink {
    /// Builds a sink from a MongoDB connection string.
    pub async fn new(uri: &str, database: &str, collection: &str) -> Result<Self, ObsError> {
        let client = mongodb::Client::with_uri_str(uri)
            .await
            .map_err(|e| ObsError::Transport(format!("mongo: connect {uri}: {e}")))?;
        Ok(Self::with_client(&client, database, collection))
    }

    /// Builds a sink from an existing client (shares its connection pool).
    pub fn with_client(client: &mongodb::Client, database: &str, collection: &str) -> Self {
        Self {
            collection: client.database(database).collection::<Document>(collection),
            ttl_seconds: None,
        }
    }

    /// Enables a TTL on this collection: every insert is stamped `_recorded_at`
    /// with the current UTC time and, once [`MongoSink::ensure_indexes`] runs, a
    /// TTL index expires documents older than `expire_after_seconds`. Clamps to
    /// a positive value — a non-positive expiry would be a no-op/error for MongoDB.
    pub fn with_expire_after_seconds(mut self, expire_after_seconds: i64) -> Self {
        self.ttl_seconds = Some(clamp_ttl_seconds(expire_after_seconds));
        self
    }

    /// Applies the backing indexes idempotently:
    ///
    /// - a **sparse** index on `trace_id` (the run-join key shared by the cost,
    ///   agent-metrics, and error docs — `None`-trace records are skipped),
    /// - a **TTL** index on `_recorded_at` when [`MongoSink::with_expire_after_seconds`]
    ///   was configured.
    ///
    /// Call it once after constructing the sink (it is a no-op for a bare sink
    /// with no TTL configured, since the sparse trace index still helps queries).
    pub async fn ensure_indexes(&self) -> Result<(), ObsError> {
        let mut models = vec![IndexModel::builder()
            .keys(mongodb::bson::doc! { "trace_id": 1 })
            .options(
                IndexOptions::builder()
                    .name("idx_trace_id".to_string())
                    .sparse(true)
                    .build(),
            )
            .build()];
        if let Some(secs) = self.ttl_seconds {
            models.push(
                IndexModel::builder()
                    .keys(mongodb::bson::doc! { "_recorded_at": 1 })
                    .options(
                        IndexOptions::builder()
                            .name("ttl_recorded_at".to_string())
                            .expire_after(Some(Duration::from_secs(secs as u64)))
                            .build(),
                    )
                    .build(),
            );
        }
        self.collection
            .create_indexes(models, None)
            .await
            .map_err(|e| ObsError::Transport(format!("mongo: create_indexes: {e}")))?;
        Ok(())
    }
}

/// Clamps a TTL expiry to a positive number of seconds (MongoDB rejects a
/// non-positive TTL; 0/negative is treated as a 1-second floor).
fn clamp_ttl_seconds(seconds: i64) -> i64 {
    seconds.max(1)
}

#[async_trait::async_trait]
impl MetricsSink for MongoSink {
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
        let mut doc =
            mongodb::bson::to_document(event).map_err(|e| ObsError::Encode(e.to_string()))?;
        if self.ttl_seconds.is_some() {
            doc.insert("_recorded_at", mongodb::bson::DateTime::now());
        }
        self.collection
            .insert_one(doc, None)
            .await
            .map_err(|e| ObsError::Transport(format!("mongo: insert: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;

    #[test]
    fn ttl_seconds_are_clamped_to_positive() {
        assert_eq!(clamp_ttl_seconds(0), 1);
        assert_eq!(clamp_ttl_seconds(-5), 1);
        assert_eq!(clamp_ttl_seconds(3600), 3600);
    }

    #[tokio::test]
    #[ignore = "requires a running MongoDB (set MONGODB_URI)"]
    async fn mongo_inserts_document() {
        let uri =
            std::env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());
        let sink = MongoSink::new(&uri, "lc_obs_test", "events").await.unwrap();
        sink.export(&ObsEvent::TokenUsage(TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        ..TokenUsage::default()}))
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires a running MongoDB (set MONGODB_URI)"]
    async fn mongo_ttl_sink_stamps_and_indexes() {
        let uri =
            std::env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());
        let sink = MongoSink::new(&uri, "lc_obs_test", "events_ttl")
            .await
            .unwrap()
            .with_expire_after_seconds(3600);
        sink.ensure_indexes().await.unwrap();
        sink.export(&ObsEvent::TokenUsage(TokenUsage {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        ..TokenUsage::default()}))
        .await
        .unwrap();
    }
}
