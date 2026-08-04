//! File 工具(沙箱安全)

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// 文件读写工具(限制在 base_path 沙箱内,扩展名白名单,大小限制)
pub struct FileTool {
    base_path: PathBuf,
    allowed_extensions: Vec<String>,
    max_size: usize,
}

impl FileTool {
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

    pub fn with_allowed_extensions(mut self, exts: Vec<String>) -> Self {
        self.allowed_extensions = exts;
        self
    }

    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    fn safe_path(&self, relative: &str) -> Result<PathBuf, ToolError> {
        let base = self
            .base_path
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("base_path 无效: {}", e)))?;
        let target = base.join(relative);

        // Reject files without an extension (bypasses whitelist)
        if target.extension().is_none() {
            return Err(ToolError::InvalidInput(
                "文件必须包含扩展名（无扩展名文件不在白名单中）".to_string(),
            ));
        }

        // 扩展名检查
        if let Some(ext) = target.extension().and_then(|e| e.to_str()) {
            if !self.allowed_extensions.iter().any(|a| a == ext) {
                return Err(ToolError::InvalidInput(format!(
                    "扩展名不允许: {}(允许: {:?})",
                    ext, self.allowed_extensions
                )));
            }
        }

        // 路径越界检查
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
            .map_err(|e| ToolError::InvalidInput(format!("路径无效: {}", e)))?;

        if !canon.starts_with(&base) {
            return Err(ToolError::InvalidInput(
                "路径越界(base_path 沙箱)".to_string(),
            ));
        }
        Ok(canon)
    }

    pub async fn read(&self, path: &str) -> Result<String, ToolError> {
        let p = self.safe_path(path)?;
        let metadata = fs::metadata(&p)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        if metadata.len() as usize > self.max_size {
            return Err(ToolError::InvalidInput(format!(
                "文件超过 {} 字节",
                self.max_size
            )));
        }
        fs::read_to_string(p)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))
    }

    pub async fn write(&self, path: &str, content: &str) -> Result<(), ToolError> {
        if content.len() > self.max_size {
            return Err(ToolError::InvalidInput(format!(
                "内容超过 {} 字节",
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

    pub async fn list(&self, dir: &str) -> Result<Vec<String>, ToolError> {
        let base = self
            .base_path
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("base_path 无效: {}", e)))?;
        let target = base.join(dir);
        let canon = target
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("路径无效: {}", e)))?;
        if !canon.starts_with(&base) {
            return Err(ToolError::InvalidInput("路径越界".to_string()));
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
            .ok_or_else(|| ToolError::InvalidInput("缺 op".to_string()))?;
        let path = v
            .get("path")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ToolError::InvalidInput("缺 path".to_string()))?;
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
            other => Err(ToolError::InvalidInput(format!("未知 op: {}", other))),
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
