//! Audit persistence sink (P1-7)
//!
//! On every Guardrail violation, `GuardrailRunner` writes the violation record to the configured
//! audit sink, for post-hoc auditing, alerting, or compliance analysis.

use std::path::PathBuf;

use async_trait::async_trait;

use crate::runner::GuardrailViolation;

/// Audit sink: asynchronously persists violation records.
///
/// Implementations must handle errors self-contained: a failed record only writes a log and never
/// throws to the caller — auditing must not block or break the main flow.
#[async_trait]
pub trait AuditSink: Send + Sync {
    /// The audit sink's name.
    fn name(&self) -> &str;

    /// Records a violation.
    async fn record(&self, violation: &GuardrailViolation);
}

/// Append-only JSON Lines audit sink: one JSON line per violation, easy to parse offline.
pub struct FileAuditSink {
    path: PathBuf,
}

impl FileAuditSink {
    /// Creates a JSON Lines audit sink writing to the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl AuditSink for FileAuditSink {
    fn name(&self) -> &str {
        "file_audit_sink"
    }

    async fn record(&self, violation: &GuardrailViolation) {
        use tokio::io::AsyncWriteExt;

        let line = match serde_json::to_string(violation) {
            Ok(json) => format!("{}\n", json),
            Err(e) => {
                log::error!("[FileAuditSink] serialize violation failed: {}", e);
                return;
            }
        };

        let result = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
        {
            Ok(mut f) => f.write_all(line.as_bytes()).await,
            Err(e) => Err(e),
        };

        if let Err(e) = result {
            log::error!(
                "[FileAuditSink] append violation to {:?} failed: {}",
                self.path,
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::io::AsyncReadExt;

    fn violation() -> GuardrailViolation {
        GuardrailViolation {
            guardrail_name: "test".to_string(),
            stage: "input".to_string(),
            reason: "blocked".to_string(),
        }
    }

    #[tokio::test]
    async fn test_file_audit_sink_appends_json_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("guardrails_audit_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let sink = Arc::new(FileAuditSink::new(path.clone()));
        sink.record(&violation()).await;
        sink.record(&violation()).await;

        let mut content = String::new();
        let mut file = tokio::fs::File::open(&path).await.unwrap();
        file.read_to_string(&mut content).await.unwrap();
        assert_eq!(content.lines().count(), 2);
        assert!(content.contains("guardrail_name"));

        let _ = std::fs::remove_file(&path);
    }
}
