//! File tool (sandbox-safe)

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// File read/write tool (confined to the base_path sandbox, extension whitelist, size limit)
pub struct FileTool {
    base_path: PathBuf,
    allowed_extensions: Vec<String>,
    max_size: usize,
}

impl FileTool {
    /// Creates a file tool (sandbox root is `base_path`).
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            allowed_extensions: vec![
                "txt".to_string(),
                "md".to_string(),
                "json".to_string(),
                "csv".to_string(),
            ],
            max_size: 10 * 1024 * 1024,
        }
    }

    /// Sets the extension whitelist (builder style).
    pub fn with_allowed_extensions(mut self, exts: Vec<String>) -> Self {
        self.allowed_extensions = exts;
        self
    }

    /// Sets the maximum file size (bytes, builder style).
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    fn safe_path(&self, relative: &str) -> Result<PathBuf, ToolError> {
        let base = self
            .base_path
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("invalid base_path: {}", e)))?;
        let target = base.join(relative);

        // Reject files without an extension (bypasses whitelist)
        if target.extension().is_none() {
            return Err(ToolError::InvalidInput(
                "file must have an extension (files without an extension are not in the whitelist)"
                    .to_string(),
            ));
        }

        // Extension check
        if let Some(ext) = target.extension().and_then(|e| e.to_str()) {
            if !self.allowed_extensions.iter().any(|a| a == ext) {
                return Err(ToolError::InvalidInput(format!(
                    "extension not allowed: {} (allowed: {:?})",
                    ext, self.allowed_extensions
                )));
            }
        }

        // Path escape check
        let canon = target
            .canonicalize()
            .or_else(|_| {
                let file_name = target.file_name().ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty file name")
                })?;
                target
                    .parent()
                    .and_then(|p| p.canonicalize().ok())
                    .map(|p| p.join(file_name))
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "parent"))
            })
            .map_err(|e| ToolError::InvalidInput(format!("invalid path: {}", e)))?;

        if !canon.starts_with(&base) {
            return Err(ToolError::InvalidInput(
                "path escapes base_path sandbox".to_string(),
            ));
        }
        Ok(canon)
    }

    /// Reads a file inside the sandbox (bounded by path-escape and size checks).
    pub async fn read(&self, path: &str) -> Result<String, ToolError> {
        let p = self.safe_path(path)?;
        let metadata = fs::metadata(&p)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        if metadata.len() as usize > self.max_size {
            return Err(ToolError::InvalidInput(format!(
                "file exceeds {} bytes",
                self.max_size
            )));
        }
        fs::read_to_string(p)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    /// Writes content to a file inside the sandbox (auto-creates parent directories).
    pub async fn write(&self, path: &str, content: &str) -> Result<(), ToolError> {
        if content.len() > self.max_size {
            return Err(ToolError::InvalidInput(format!(
                "content exceeds {} bytes",
                self.max_size
            )));
        }
        let p = self.safe_path(path)?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        }
        let mut f = fs::File::create(&p)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        f.write_all(content.as_bytes())
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(())
    }

    /// Lists the entries under a directory inside the sandbox.
    pub async fn list(&self, dir: &str) -> Result<Vec<String>, ToolError> {
        let base = self
            .base_path
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("invalid base_path: {}", e)))?;
        let target = base.join(dir);
        let canon = target
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("invalid path: {}", e)))?;
        if !canon.starts_with(&base) {
            return Err(ToolError::InvalidInput(
                "path escapes base_path".to_string(),
            ));
        }
        let mut entries = fs::read_dir(&canon)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        let mut result = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
        {
            result.push(entry.file_name().to_string_lossy().to_string());
        }
        Ok(result)
    }
}

#[async_trait]
impl BaseTool for FileTool {
    fn name(&self) -> &str {
        "file_operation"
    }

    fn description(&self) -> &str {
        "文件操作。输入 JSON: {\"op\": \"read|write|list\", \"path\": \"...\", \"content\": \"...\"}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let v: Value =
            serde_json::from_str(&input).map_err(|e| ToolError::InvalidInput(e.to_string()))?;
        let op = v
            .get("op")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing op".to_string()))?;
        let path = v
            .get("path")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing path".to_string()))?;
        match op {
            "read" => self.read(path).await,
            "write" => {
                let content = v.get("content").and_then(|x| x.as_str()).unwrap_or("");
                self.write(path, content).await?;
                Ok("写入成功".to_string())
            }
            "list" => {
                let list = self.list(path).await?;
                serde_json::to_string(&list).map_err(|e| ToolError::ExecutionFailed(e.to_string()))
            }
            other => Err(ToolError::InvalidInput(format!("unknown op: {}", other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tool() -> (FileTool, TempDir) {
        let dir = TempDir::new().unwrap();
        let tool = FileTool::new(dir.path().to_path_buf());
        (tool, dir)
    }

    #[tokio::test]
    async fn test_write_and_read() {
        let (tool, _dir) = tool();
        tool.write("test.txt", "hello").await.unwrap();
        let content = tool.read("test.txt").await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_extension_not_allowed() {
        let (tool, _dir) = tool();
        assert!(tool.write("file.exe", "x").await.is_err());
    }

    #[tokio::test]
    async fn test_path_traversal_blocked() {
        let (tool, _dir) = tool();
        assert!(tool.read("../../../../etc/passwd").await.is_err());
    }

    #[tokio::test]
    async fn test_list() {
        let (tool, _dir) = tool();
        tool.write("a.txt", "a").await.unwrap();
        tool.write("b.txt", "b").await.unwrap();
        let list = tool.list(".").await.unwrap();
        assert!(list.len() >= 2);
    }

    #[tokio::test]
    async fn test_max_size_exceeded() {
        let (tool, _dir) = tool();
        let tool = tool.with_max_size(5);
        assert!(tool.write("big.txt", "12345678").await.is_err());
    }

    #[tokio::test]
    async fn test_run_write_read_via_base_tool() {
        let (tool, _dir) = tool();
        let write_input = r#"{"op":"write","path":"x.txt","content":"hi"}"#;
        assert!(tool.run(write_input.to_string()).await.is_ok());
        let read_input = r#"{"op":"read","path":"x.txt"}"#;
        let result = tool.run(read_input.to_string()).await.unwrap();
        assert_eq!(result, "hi");
    }
}
