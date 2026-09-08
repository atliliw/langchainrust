// lc-observability/src/jsonl.rs
//! JSON-lines file sink.

use lc_core::observability::{MetricsSink, ObsError, ObsEvent};
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Appends each event as a single JSON line to a file (one record per line —
/// tail-able, `jq`-parseable, replayable). The file is opened once in append mode
/// at construction, so concurrent exports serialize through the mutex.
pub struct JsonLinesSink {
    file: tokio::sync::Mutex<tokio::fs::File>,
}

impl JsonLinesSink {
    /// Opens (creating if needed) the output file in append mode.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self, ObsError> {
        let path = path.as_ref();
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|e| ObsError::Transport(format!("jsonl: open {}: {e}", path.display())))?;
        Ok(Self {
            file: tokio::sync::Mutex::new(file),
        })
    }
}

#[async_trait::async_trait]
impl MetricsSink for JsonLinesSink {
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
        let line = serde_json::to_string(event).map_err(|e| ObsError::Encode(e.to_string()))?;
        let mut file = self.file.lock().await;
        file.write_all(line.as_bytes())
            .await
            .map_err(|e| ObsError::Transport(format!("jsonl: write: {e}")))?;
        file.write_all(b"\n")
            .await
            .map_err(|e| ObsError::Transport(format!("jsonl: write: {e}")))?;
        file.flush()
            .await
            .map_err(|e| ObsError::Transport(format!("jsonl: flush: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;
    use lc_core::observability::AgentMetrics;
    use std::time::Duration;

    fn token_event() -> ObsEvent {
        ObsEvent::TokenUsage(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        })
    }

    #[tokio::test]
    async fn jsonl_appends_one_record_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obs.jsonl");
        let sink = JsonLinesSink::new(&path).await.unwrap();

        sink.export(&token_event()).await.unwrap();
        sink.export(&ObsEvent::AgentMetrics(AgentMetrics {
            trace_id: None,
            llm_calls: 2,
            cache_hits: 0,
            tool_calls: 1,
            compactions: 0,
            total_tokens: Some(15),
            duration: Duration::from_millis(10),
        }))
        .await
        .unwrap();

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "one JSON line per event");
        let v0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v0["kind"], "token_usage");
        assert_eq!(v0["total_tokens"], 15);
        let v1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v1["kind"], "agent_metrics");
        assert_eq!(v1["tool_calls"], 1);
    }

    #[tokio::test]
    async fn jsonl_append_preserves_existing_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("obs.jsonl");
        tokio::fs::write(&path, "existing\n").await.unwrap();

        let sink = JsonLinesSink::new(&path).await.unwrap();
        sink.export(&token_event()).await.unwrap();

        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content.lines().count(), 2, "must not truncate the file");
    }
}
