use async_trait::async_trait;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::callbacks::{CallbackHandler, RunTree};
use crate::schema::Message;

pub enum LogFormat {
    Plain,
    Json,
    JsonLines,
}

pub struct FileCallbackHandler {
    file: Mutex<File>,
    path: PathBuf,
    format: LogFormat,
}

impl FileCallbackHandler {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            file: Mutex::new(file),
            path,
            format: LogFormat::JsonLines,
        })
    }

    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    fn write_log(&self, event: &str, run: &RunTree, data: serde_json::Value) {
        if let Ok(mut file) = self.file.lock() {
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
                    let _ = writeln!(file, "{}", entry);
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
                    let _ = writeln!(file, "{}", serde_json::to_string_pretty(&entry).unwrap_or_default());
                }
                LogFormat::Plain => {
                    let _ = writeln!(
                        file,
                        "[{}] {} - {} ({})",
                        chrono::Utc::now().to_rfc3339(),
                        event,
                        run.name,
                        run.run_type.as_str()
                    );
                }
            }
        }
    }
}

#[async_trait]
impl CallbackHandler for FileCallbackHandler {
    async fn on_run_start(&self, run: &RunTree) {
        self.write_log("run_start", run, serde_json::json!({"inputs": run.inputs}));
    }

    async fn on_run_end(&self, run: &RunTree) {
        self.write_log("run_end", run, serde_json::json!({"outputs": run.outputs, "duration_ms": run.duration_ms()}));
    }

    async fn on_run_error(&self, run: &RunTree, error: &str) {
        self.write_log("run_error", run, serde_json::json!({"error": error}));
    }

    async fn on_llm_start(&self, run: &RunTree, messages: &[Message]) {
        self.write_log("llm_start", run, serde_json::json!({
            "messages_count": messages.len(),
            "messages": messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>()
        }));
    }

    async fn on_llm_end(&self, run: &RunTree, response: &str) {
        self.write_log("llm_end", run, serde_json::json!({"response_length": response.len()}));
    }

    async fn on_llm_new_token(&self, run: &RunTree, token: &str) {
        self.write_log("llm_new_token", run, serde_json::json!({"token": token}));
    }

    async fn on_llm_error(&self, run: &RunTree, error: &str) {
        self.write_log("llm_error", run, serde_json::json!({"error": error}));
    }

    async fn on_chain_start(&self, run: &RunTree, inputs: &serde_json::Value) {
        self.write_log("chain_start", run, serde_json::json!({"inputs": inputs}));
    }

    async fn on_chain_end(&self, run: &RunTree, outputs: &serde_json::Value) {
        self.write_log("chain_end", run, serde_json::json!({"outputs": outputs}));
    }

    async fn on_chain_error(&self, run: &RunTree, error: &str) {
        self.write_log("chain_error", run, serde_json::json!({"error": error}));
    }

    async fn on_tool_start(&self, run: &RunTree, tool_name: &str, input: &str) {
        self.write_log("tool_start", run, serde_json::json!({"tool_name": tool_name, "input": input}));
    }

    async fn on_tool_end(&self, run: &RunTree, output: &str) {
        self.write_log("tool_end", run, serde_json::json!({"output": output}));
    }

    async fn on_tool_error(&self, run: &RunTree, error: &str) {
        self.write_log("tool_error", run, serde_json::json!({"error": error}));
    }

    async fn on_retriever_start(&self, run: &RunTree, query: &str) {
        self.write_log("retriever_start", run, serde_json::json!({"query": query}));
    }

    async fn on_retriever_end(&self, run: &RunTree, documents: &[serde_json::Value]) {
        self.write_log("retriever_end", run, serde_json::json!({"documents_count": documents.len()}));
    }

    async fn on_retriever_error(&self, run: &RunTree, error: &str) {
        self.write_log("retriever_error", run, serde_json::json!({"error": error}));
    }
}

impl std::fmt::Debug for FileCallbackHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCallbackHandler")
            .field("path", &self.path)
            .field("format", &match self.format {
                LogFormat::Plain => "Plain",
                LogFormat::Json => "Json",
                LogFormat::JsonLines => "JsonLines",
            })
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_handler_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        assert!(handler.path.exists());
    }

    #[tokio::test]
    async fn test_write_log_json_lines() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap();
        
        let run = RunTree::new("test_run", crate::callbacks::RunType::Llm, serde_json::json!({"input": "test"}));
        handler.write_log("test_event", &run, serde_json::json!({"data": "value"}));
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_event"));
        assert!(content.contains("\"run_name\":\"test_run\""));
    }

    #[tokio::test]
    async fn test_write_log_plain_format() {
        let temp_file = NamedTempFile::new().unwrap();
        let handler = FileCallbackHandler::new(temp_file.path()).unwrap().with_format(LogFormat::Plain);
        
        let run = RunTree::new("test_run", crate::callbacks::RunType::Llm, serde_json::json!({"input": "test"}));
        handler.write_log("test_event", &run, serde_json::json!({}));
        
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("test_event"));
        assert!(content.contains("test_run"));
    }
}