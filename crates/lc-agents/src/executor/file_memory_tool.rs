// lc-agents/src/executor/file_memory_tool.rs
//! C1 — a file-memory tool mount (v0.22.1 §S8).
//!
//! Mechanical introspection alone cannot persist an agent's cross-session knowledge; the
//! file-backed memory from lc-memory (`FileMemoryStore`) is normally prompt-injected via
//! `with_memory`. This module exposes the *same* store as an ordinary `Arc<dyn BaseTool>`, so
//! an agent can explicitly write, read, append and list named memories itself.
//!
//! ## Mounting (default off)
//!
//! It is **not** auto-attached — callers opt in explicitly:
//! `AgentExecutor::with_memory_tool(store)` pushes it into the executor's tool set, mirroring
//! the Rule-of-Two discipline: every safety feature here defaults to off and is opted into.
//!
//! ## Input contract
//!
//! Expects a JSON object in the tool input:
//! `{"op":"view|create|write|append|delete|list","name":"<memory>","content":"<text>"}`.
//! `op` is required; `name` is required except for `list`; `content` is required for
//! `create`/`write`/`append`.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use lc_core::tools::{BaseTool, ToolError};
use lc_memory::file_memory::{FileMemoryError, FileMemoryStore};

/// Name under which the memory tool registers itself.
pub const FILE_MEMORY_TOOL_NAME: &str = "file_memory";

/// The memory tool described in the tool description so the LLM knows when to use it.
const DESCRIPTION: &str = "\
Persistent file-backed agent memory. Ops:
- view: read a memory's text (name)
- create: write a new named memory (name, content)
- write: overwrite an existing memory (name, content)
- append: append text to a memory (name, content)
- delete: remove a memory (name)
- list: list existing memory names
Input JSON: {\"op\": \"view\", \"name\": \"user_profile\"}";

/// An `Arc<dyn BaseTool>` adapter over a [`FileMemoryStore`], exposing the store's ops as a
/// single string-in/string-out tool for the agent loop.
pub struct FileMemoryTool {
    store: FileMemoryStore,
}

impl FileMemoryTool {
    /// Builds the tool over a store rooted at `root` (the directory is created if absent).
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, FileMemoryError> {
        Ok(Self {
            store: FileMemoryStore::new(root)?,
        })
    }
}

#[async_trait]
impl BaseTool for FileMemoryTool {
    fn name(&self) -> &str {
        FILE_MEMORY_TOOL_NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: Value = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("memory input not JSON: {e}")))?;
        let op = parsed
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing required field 'op'".into()))?;
        let name = parsed.get("name").and_then(Value::as_str);
        let content = parsed.get("content").and_then(Value::as_str).unwrap_or("");

        match op {
            "view" => {
                let n = require_name(name)?;
                self.store
                    .view(n)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
            }
            "list" => {
                let entries = self
                    .store
                    .list()
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                if entries.is_empty() {
                    Ok("(no memories)".to_string())
                } else {
                    let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
                    Ok(format!("memories: {}", names.join(", ")))
                }
            }
            "create" => {
                let n = require_name(name)?;
                self.store
                    .create(n, content)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(format!("created memory '{n}'"))
            }
            "write" => {
                let n = require_name(name)?;
                self.store
                    .write(n, content)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(format!("wrote memory '{n}'"))
            }
            "append" => {
                let n = require_name(name)?;
                self.store
                    .append(n, content)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(format!("appended to memory '{n}'"))
            }
            "delete" => {
                let n = require_name(name)?;
                self.store
                    .delete(n)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(format!("deleted memory '{n}'"))
            }
            other => Err(ToolError::InvalidInput(format!(
                "unknown op '{other}' (expected view|create|write|append|delete|list)"
            ))),
        }
    }
}

fn require_name(name: Option<&str>) -> Result<&str, ToolError> {
    name.filter(|s| !s.is_empty())
        .ok_or_else(|| ToolError::InvalidInput("missing required field 'name'".into()))
}

impl std::fmt::Debug for FileMemoryTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMemoryTool").finish()
    }
}

/// Convenience: builds and mounts as a boxed `BaseTool` (`Err` when the root dir can't be set up).
pub fn mount(root: impl Into<PathBuf>) -> Result<Arc<dyn BaseTool>, FileMemoryError> {
    Ok(Arc::new(FileMemoryTool::new(root)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_tool() -> (tempfile::TempDir, FileMemoryTool) {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileMemoryTool::new(dir.path()).unwrap();
        (dir, tool)
    }

    async fn run(tool: &FileMemoryTool, s: &str) -> Result<String, ToolError> {
        tool.run(s.to_string()).await
    }

    #[tokio::test]
    async fn create_view_write_roundtrip() {
        let (_d, tool) = temp_tool();
        let out = run(
            &tool,
            r#"{"op":"create","name":"profile","content":"Alice"}"#,
        )
        .await
        .unwrap();
        assert!(out.contains("created memory 'profile'"));
        let out = run(&tool, r#"{"op":"view","name":"profile"}"#)
            .await
            .unwrap();
        assert_eq!(out, "Alice");
        let out = run(
            &tool,
            r#"{"op":"append","name":"profile","content":" / engineer"}"#,
        )
        .await
        .unwrap();
        assert!(out.contains("appended"));
        let view = run(&tool, r#"{"op":"view","name":"profile"}"#)
            .await
            .unwrap();
        assert!(view.contains("engineer"));
        let out = run(&tool, r#"{"op":"write","name":"profile","content":"Bob"}"#)
            .await
            .unwrap();
        assert!(out.contains("wrote memory 'profile'"));
        let view = run(&tool, r#"{"op":"view","name":"profile"}"#)
            .await
            .unwrap();
        assert_eq!(view, "Bob");
    }

    #[tokio::test]
    async fn list_and_delete() {
        let (_d, tool) = temp_tool();
        run(&tool, r#"{"op":"create","name":"a","content":"1"}"#)
            .await
            .unwrap();
        run(&tool, r#"{"op":"create","name":"b","content":"2"}"#)
            .await
            .unwrap();
        let list = run(&tool, r#"{"op":"list"}"#).await.unwrap();
        assert!(list.contains("a") && list.contains("b"));
        run(&tool, r#"{"op":"delete","name":"a"}"#).await.unwrap();
        let list = run(&tool, r#"{"op":"list"}"#).await.unwrap();
        assert!(!list.contains("a"));
    }

    #[tokio::test]
    async fn missing_fields_error() {
        let (_d, tool) = temp_tool();
        assert!(matches!(
            tool.run(r#"{}"#.to_string()).await.unwrap_err(),
            ToolError::InvalidInput(_)
        ));
        let err = tool.run(r#"{"op":"view"}"#.to_string()).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        let err = tool
            .run(r#"{"op":"bogus","name":"x"}"#.to_string())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn view_missing_memory_is_execution_error_not_invalid_input() {
        let (_d, tool) = temp_tool();
        // a memory that doesn't exist is an execution outcome, not a malformed input
        let err = tool
            .run(r#"{"op":"view","name":"nope"}"#.to_string())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn mount_returns_arc_base_tool() {
        let dir = tempfile::tempdir().unwrap();
        let arc = mount(dir.path()).unwrap();
        assert_eq!(arc.name(), FILE_MEMORY_TOOL_NAME);
    }
}
