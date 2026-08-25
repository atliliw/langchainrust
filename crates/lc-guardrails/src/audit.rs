//! 审计持久化 sink(P1-7)
//!
//! 每次 Guardrail 违规发生时,`GuardrailRunner` 把违规记录写入配置的审计 sink,
//! 供事后审计、告警或合规分析。

use std::path::PathBuf;

use async_trait::async_trait;

use crate::runner::GuardrailViolation;

/// 审计 sink:异步持久化违规记录。
///
/// 实现必须自包含错误处理:记录失败仅写日志,绝不向调用方抛错——
/// 审计不应阻塞或破坏主流程。
#[async_trait]
pub trait AuditSink: Send + Sync {
    /// 审计 sink 的名称。
    fn name(&self) -> &str;

    /// 记录一条违规。
    async fn record(&self, violation: &GuardrailViolation);
}

/// 追加式 JSON Lines 审计 sink:每条违规一行 JSON,便于离线解析。
pub struct FileAuditSink {
    path: PathBuf,
}

impl FileAuditSink {
    /// 创建一个写入指定路径的 JSON Lines 审计 sink。
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
