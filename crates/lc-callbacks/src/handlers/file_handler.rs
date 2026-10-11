use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::Mutex;

use crate::{CallbackHandler, RunTree};
use lc_schema::Message;

/// Log output format.
pub enum LogFormat {
    /// Plain text output.
    Plain,
    /// Pretty-printed JSON output.
    Json,
    /// Newline-delimited JSON output.
    JsonLines,
}

/// A callback handler that writes run events to a log file.
pub struct FileCallbackHandler {
    file: Mutex<tokio::fs::File>,
    path: PathBuf,
    format: LogFormat,
    /// Current byte length of the valid JSON array written under `LogFormat::Json`,
    /// or 0 when no Json array is open yet (or a non-Json format last wrote). Atomic so
    /// `write_log(&self)` can maintain it despite the trait's `&self` receiver; all access
    /// is under `file`'s mutex, so `Ordering::Relaxed` suffices.
    json_len: AtomicU64,
}

impl FileCallbackHandler {
    /// Creates a new handler that appends logs to the file at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = path.into();
        // Open synchronously for construction, then convert to tokio::fs::File.
        // We use read+write (not append) so the `Json` format can rewrite its trailing
        // `]` terminator to keep the file a single valid JSON array (§3.2). `Plain` /
        // `JsonLines` seek to EOF before each write to preserve append semantics.
        let std_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            // We never truncate: `Plain`/`JsonLines` seek to EOF to append, and `Json`
            // maintains a streaming array by rewriting its trailing `]`.
            .truncate(false)
            .open(&path)?;
        let file = tokio::fs::File::from_std(std_file);
        Ok(Self {
            file: Mutex::new(file),
            path,
            format: LogFormat::JsonLines,
            json_len: AtomicU64::new(0),
        })
    }

    /// Sets the log output format.
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    async fn write_log(&self, event: &str, run: &RunTree, data: serde_json::Value) {
        let mut file = self.file.lock().await;
        match self.format {
            LogFormat::JsonLines => {
                let entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "event": event,
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_type": run.run_type.as_str(),
                    "data": data
                });
                // Non-Json formats append at EOF (the handle is read+write now), and reset
                // the tracked Json array length so a later Json write starts a fresh array.
                self.json_len.store(0, Ordering::Relaxed);
                if let Err(e) = file.seek(SeekFrom::End(0)).await {
                    log::warn!("FileCallbackHandler seek failed: {}", e);
                }
                // M10 fix: log write errors instead of silently discarding
                if let Err(e) = file.write_all(format!("{}\n", entry).as_bytes()).await {
                    log::warn!("FileCallbackHandler write failed: {}", e);
                }
                if let Err(e) = file.flush().await {
                    log::warn!("FileCallbackHandler flush failed: {}", e);
                }
            }
            LogFormat::Json => {
                let entry = serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "event": event,
                    "run_id": run.id,
                    "run_name": run.name,
                    "run_type": run.run_type.as_str(),
                    "data": data
                });
                let pretty = serde_json::to_string_pretty(&entry).unwrap_or_default();
                // §3.2: `Json` must produce a single *valid* JSON array document, not
                // concatenated bare objects that `serde_json::from_str` rejects. We hold
                // the buffer exclusively, so we maintain a streaming array by rewriting a
                // trailing `]` terminator, keeping the file a valid `[...]` array at every
                // point:
                //   first write → `[ <entry> ]`
                //   next        → overwrite the `]` with `, <entry> ]`
                // `json_len` records the current valid-array byte length (0 = array open).
                if self.json_len.load(Ordering::Relaxed) == 0 {
                    let doc = format!("[\n{pretty}\n]");
                    self.json_len.store(doc.len() as u64, Ordering::Relaxed);
                    if let Err(e) = file.write_all(doc.as_bytes()).await {
                        self.json_len.store(0, Ordering::Relaxed);
                        log::warn!("FileCallbackHandler write failed: {}", e);
                    }
                } else {
                    // Move the cursor over the trailing `]` and replace it with the next,
                    // comma-separated element followed by a fresh closing `]`.
                    let tail = format!(", {pretty} ]");
                    if let Err(e) = file
                        .seek(SeekFrom::Start(
                            self.json_len.load(Ordering::Relaxed).saturating_sub(1),
                        ))
                        .await
                    {
                        log::warn!("FileCallbackHandler seek failed: {}", e);
                    } else if file.write_all(tail.as_bytes()).await.is_ok() {
                        // Overwrote the 1-byte `]` with `tail`; net growth = tail.len() - 1.
                        self.json_len.store(
                            self.json_len.load(Ordering::Relaxed) - 1 + tail.len() as u64,
                            Ordering::Relaxed,
                        );
                    } else {
                        log::warn!("FileCallbackHandler write failed");
                    }
                }
                if let Err(e) = file.flush().await {
                    log::warn!("FileCallbackHandler flush failed: {}", e);
                }
            }
            LogFormat::Plain => {
                let line = format!(
                    "[{}] {} - {} ({})\n",
                    chrono::Utc::now().to_rfc3339(),
                    event,
                    run.name,
                    run.run_type.as_str()
                );
                self.json_len.store(0, Ordering::Relaxed);
                if let Err(e) = file.seek(SeekFrom::End(0)).await {
                    log::warn!("FileCallbackHandler seek failed: {}", e);
                }
                if let Err(e) = file.write_all(line.as_bytes()).await {
                    log::warn!("FileCallbackHandler write failed: {}", e);
                }
                if let Err(e) = file.flush().await {
                    log::warn!("FileCallbackHandler flush failed: {}", e);
                }
            }
        }
    }
}

#[async_trait]
impl CallbackHandler for FileCallbackHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.write_log("run_start", run, serde_json::json!({"inputs": run.inputs}))
            .await;
    }

    async fn on_run_end(&self, run: &RunTree) {
        self.write_log(
            "run_end",
            run,
            serde_json::json!({"outputs": run.outputs, "duration_ms": run.duration_ms()}),
        )
        .await;
    }

    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.write_log("run_error", run, serde_json::json!({"error": error}))
            .await;
    }

    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]) {
        self.write_log(
            "llm_start",
            run,
            serde_json::json!({
                "messages_count": messages.len(),
                "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>()
            }),
        )
        .await;
    }

    async fn on_llm_end(&self, run: &RunTree, response: &str) {
        self.write_log(
            "llm_end",
            run,
            serde_json::json!({"response_length": response.len()}),
        )
        .await;
    }

    async fn on_llm_new_token(&self, run: &RunTree, token: &str) {
        self.write_log("llm_new_token", run, serde_json::json!({"token": token}))
            .await;
    }

    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.write_log("llm_error", run, serde_json::json!({"error": error}))
            .await;
    }

    async fn on_chain_start(&self, run: &RunTree, inputs: &serde_json::Value) {
        self.write_log("chain_start", run, serde_json::json!({"inputs": inputs}))
            .await;
    }

    async fn on_chain_end(&self, run: &RunTree, outputs: &serde_json::Value) {
        self.write_log("chain_end", run, serde_json::json!({"outputs": outputs}))
            .await;
    }

    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.write_log("chain_error", run, serde_json::json!({"error": error}))
            .await;
    }

    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        self.write_log(
            "tool_start",
            run,
            serde_json::json!({"tool_name": tool_name, "input": input}),
        )
        .await;
    }

    async fn on_tool_end(&self, run: &RunTree, output: &str) {
        self.write_log("tool_end", run, serde_json::json!({"output": output}))
            .await;
    }

    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.write_log("tool_error", run, serde_json::json!({"error": error}))
            .await;
    }

    async fn on_retriever_start(&self, run: &RunTree, query: &str) {
        self.write_log("retriever_start", run, serde_json::json!({"query": query}))
            .await;
    }

    async fn on_retriever_end(&self, run: &RunTree, documents: &[serde_json::Value]) {
        self.write_log(
            "retriever_end",
            run,
            serde_json::json!({"documents_count": documents.len()}),
        )
        .await;
    }

    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.write_log("retriever_error", run, serde_json::json!({"error": error}))
            .await;
    }
}

impl std::fmt::Debug for FileCallbackHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCallbackHandler")
            .field("path", &self.path)
            .field(
                "format",
                &match self.format {
                    LogFormat::Plain => "Plain",
                    LogFormat::Json => "Json",
                    LogFormat::JsonLines => "JsonLines",
                },
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_file_handler_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        assert!(handler.path.exists());
    }

    #[tokio::test]
    async fn test_write_log_json_lines() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();

        let run = RunTree::new(
            "test_run",
            crate::RunType::Llm,
            serde_json::json!({"input": "test"}),
        );
        handler
            .write_log("test_event", &run, serde_json::json!({"data": "value"}))
            .await;

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_event"));
        assert!(content.contains("\"run_name\":\"test_run\""));
    }

    #[tokio::test]
    async fn test_write_log_json_produces_valid_array() {
        // §3.2: `LogFormat::Json` must yield a single document `serde_json::from_str`
        // accepts, not concatenated bare objects.
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path())
            .unwrap()
            .with_format(LogFormat::Json);

        let run = RunTree::new(
            "test_run",
            crate::RunType::Llm,
            serde_json::json!({"input": "test"}),
        );
        handler
            .write_log("event_a", &run, serde_json::json!({"data": "a"}))
            .await;
        handler
            .write_log("event_b", &run, serde_json::json!({"data": "b"}))
            .await;

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        let arr: serde_json::Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Json log must parse as one document: {e}\n{content}"));
        let arr = arr.as_array().expect("Json log must be a JSON array");
        assert_eq!(arr.len(), 2, "both events must be array elements");
        assert_eq!(arr[0]["event"], "event_a");
        assert_eq!(arr[1]["event"], "event_b");
    }

    #[tokio::test]
    async fn test_write_log_plain_format() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path())
            .unwrap()
            .with_format(LogFormat::Plain);

        let run = RunTree::new(
            "test_run",
            crate::RunType::Llm,
            serde_json::json!({"input": "test"}),
        );
        handler
            .write_log("test_event", &run, serde_json::json!({}))
            .await;

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_event"));
        assert!(content.contains("test_run"));
    }
}
