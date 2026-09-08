// lc-observability/src/mongo.rs
//! MongoDB sink (feature `mongodb`).

use lc_core::observability::{MetricsSink, ObsError, ObsEvent};

/// Inserts each event as one document into the given collection. Records keep the
/// serde tag (`"kind"`), so a single collection can hold token-usage and agent
/// metrics side by side.
#[derive(Clone)]
pub struct MongoSink {
    collection: mongodb::Collection<serde_json::Value>,
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
            collection: client
                .database(database)
                .collection::<serde_json::Value>(collection),
        }
    }
}

#[async_trait::async_trait]
impl MetricsSink for MongoSink {
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
        let doc = serde_json::to_value(event).map_err(|e| ObsError::Encode(e.to_string()))?;
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
        }))
        .await
        .unwrap();
    }
}
