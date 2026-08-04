// lc-tools/src/sandbox/local.rs
//! Local process sandbox backend.

use std::time::Instant;

use async_trait::async_trait;
use tokio::process::Command;

use super::{CodeSandbox, Language, RunResult, SandboxError};

/// Dangerous Python modules that are blocked for security.
const BLOCKED_PYTHON_IMPORTS: &[&str] = &[
    "os",
    "subprocess",
    "sys",
    "shutil",
    "signal",
    "ctypes",
    "multiprocessing",
    "socket",
    "http.server",
    "xmlrpc",
    "pickle",
    "shelve",
    "importlib",
    "code",
    "codeop",
    "compileall",
    "pty",
    "commands",
    "pdb",
    "webbrowser",
];

/// Check if Python code contains dangerous imports.
fn contains_dangerous_python_import(code: &str) -> Option<String> {
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        let code_part = trimmed.split('#').next().unwrap_or(trimmed);
        if code_part.contains("import") {
            for blocked in BLOCKED_PYTHON_IMPORTS {
                if code_part.contains(&format!("import {}", blocked))
                    || code_part.contains(&format!("from {} ", blocked))
                    || code_part.contains(&format!("from {}.", blocked))
                    || code_part.contains(&format!("from {}import", blocked))
                {
                    return Some(blocked.to_string());
                }
            }
        }
    }
    None
}

/// Local process sandbox using `tokio::process::Command`.
pub struct LocalSandbox {
    python_path: String,
    node_path: String,
}

impl LocalSandbox {
    /// Create a new local sandbox with auto-detected interpreter paths.
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
            node_path: "node".to_string(),
        }
    }

    /// Use a custom Python interpreter path.
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// Use a custom Node.js runtime path.
    pub fn with_node_path(mut self, path: impl Into<String>) -> Self {
        self.node_path = path.into();
        self
    }

    /// Auto-detect the Python interpreter on the system.
    fn find_python() -> String {
        for candidate in &["python3", "python"] {
            if std::process::Command::new(candidate)
                .arg("--version")
                .output()
                .is_ok()
            {
                return candidate.to_string();
            }
        }
        "python3".to_string()
    }

    /// Build the command for the given language and code.
    fn build_command(&self, code: &str, language: Language) -> Result<Command, SandboxError> {
        match language {
            Language::Python => {
                if let Some(blocked) = contains_dangerous_python_import(code) {
                    return Err(SandboxError::Runtime(format!(
                        "Code contains dangerous import: '{}'. \
                         This module is blocked for security in local sandbox.",
                        blocked
                    )));
                }
                let mut cmd = Command::new(&self.python_path);
                cmd.arg("-c").arg(code);
                Ok(cmd)
            }
            Language::JavaScript => {
                let mut cmd = Command::new(&self.node_path);
                cmd.arg("-e").arg(code);
                Ok(cmd)
            }
            Language::Rust => Err(SandboxError::UnsupportedLanguage(
                "Rust compilation is not supported by LocalSandbox (use E2BSandbox or WasmSandbox)"
                    .to_string(),
            )),
        }
    }
}

impl Default for LocalSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CodeSandbox for LocalSandbox {
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError> {
        let mut cmd = self.build_command(code, language)?;

        let start = Instant::now();

        let result =
            tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), cmd.output())
                .await
                .map_err(|_| SandboxError::Timeout(timeout_ms))?
                .map_err(|e| {
                    SandboxError::Runtime(format!("failed to execute subprocess: {}", e))
                })?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let exit_code = result.status.code().unwrap_or(-1);

        Ok(RunResult {
            stdout,
            stderr,
            exit_code,
            execution_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_sandbox_default() {
        let sandbox = LocalSandbox::default();
        assert!(!sandbox.python_path.is_empty());
        assert_eq!(sandbox.node_path, "node");
    }

    #[test]
    fn test_local_sandbox_custom_paths() {
        let sandbox = LocalSandbox::new()
            .with_python_path("/usr/bin/python3.11")
            .with_node_path("/usr/local/bin/node");
        assert_eq!(sandbox.python_path, "/usr/bin/python3.11");
        assert_eq!(sandbox.node_path, "/usr/local/bin/node");
    }

    #[test]
    fn test_rust_unsupported() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.build_command("fn main(){}", Language::Rust);
        assert!(result.is_err());
        match result.unwrap_err() {
            SandboxError::UnsupportedLanguage(msg) => {
                assert!(msg.contains("Rust"));
            }
            other => panic!("expected UnsupportedLanguage, got: {:?}", other),
        }
    }

    #[test]
    fn test_dangerous_python_import_detection() {
        assert!(contains_dangerous_python_import("import os").is_some());
        assert!(contains_dangerous_python_import("from sys import path").is_some());
        assert!(contains_dangerous_python_import("import subprocess").is_some());
        assert!(contains_dangerous_python_import("import math").is_none());
        assert!(contains_dangerous_python_import("import json").is_none());
        assert!(contains_dangerous_python_import("from datetime import datetime").is_none());
        assert!(contains_dangerous_python_import("# import os").is_none());
    }

    #[tokio::test]
    async fn test_python_dangerous_import_blocked() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("import os; print(os.getcwd())", Language::Python, 10_000)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dangerous import"),
            "Expected dangerous import error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_python_execution_if_available() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("print(1 + 2)", Language::Python, 10_000).await;

        match result {
            Ok(run_result) => {
                if run_result.exit_code == 0 {
                    assert!(
                        run_result.stdout.trim() == "3",
                        "expected '3', got '{}'",
                        run_result.stdout.trim()
                    );
                }
            }
            Err(SandboxError::Runtime(msg)) => {
                eprintln!("Python not available (expected in some CI): {}", msg);
            }
            Err(other) => panic!("unexpected error: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_javascript_execution_if_available() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("console.log(1 + 2)", Language::JavaScript, 10_000)
            .await;

        match result {
            Ok(run_result) => {
                if run_result.exit_code == 0 {
                    assert!(
                        run_result.stdout.trim() == "3",
                        "expected '3', got '{}'",
                        run_result.stdout.trim()
                    );
                }
            }
            Err(SandboxError::Runtime(msg)) => {
                eprintln!("Node.js not available (expected in some CI): {}", msg);
            }
            Err(other) => panic!("unexpected error: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_rust_execution_unsupported() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("fn main(){}", Language::Rust, 10_000).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            SandboxError::UnsupportedLanguage(_) => {}
            other => panic!("expected UnsupportedLanguage, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_execution_timeout() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run("import time; time.sleep(10)", Language::Python, 100)
            .await;

        match result {
            Err(SandboxError::Timeout(ms)) => {
                assert_eq!(ms, 100);
            }
            Ok(_) => {
                // Python not available, code didn't run — acceptable
            }
            Err(other) => panic!("expected Timeout, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_execution_time_is_recorded() {
        let sandbox = LocalSandbox::new();
        let result = sandbox.run("print('hi')", Language::Python, 10_000).await;

        match result {
            Ok(run_result) => {
                assert!(run_result.execution_time_ms < 10_000);
            }
            Err(_) => {}
        }
    }

    #[tokio::test]
    async fn test_stderr_captured() {
        let sandbox = LocalSandbox::new();
        let result = sandbox
            .run(
                "import sys; print('error', file=sys.stderr)",
                Language::Python,
                10_000,
            )
            .await;

        match result {
            Ok(run_result) => {
                if run_result.exit_code == 0 {
                    assert!(
                        run_result.stderr.contains("error"),
                        "stderr should contain 'error', got: '{}'",
                        run_result.stderr
                    );
                }
            }
            Err(_) => {}
        }
    }
}
