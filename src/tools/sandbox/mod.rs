// src/tools/sandbox/mod.rs
//! Code Interpreter Sandbox for secure code execution.
//!
//! Provides a pluggable sandbox architecture for executing code in isolated environments.
//! The [`CodeSandbox`] trait defines the interface for sandbox backends, and [`SandboxTool`]
//! wraps any sandbox implementation as a [`BaseTool`] usable by agents.
//!
//! # Backends
//!
//! - **[`LocalSandbox`]**: Default backend using subprocess with timeout. Works without
//!   external dependencies. Suitable for development and testing.
//! - **`E2BSandbox`**: Cloud sandbox via E2B (behind `sandbox-e2b` feature gate).
//! - **`WasmSandbox`**: WASM-based sandbox (behind `sandbox-wasm` feature gate).
//!
//! # Example
//!
//! ```ignore
//! use langchainrust::tools::sandbox::{LocalSandbox, SandboxTool, Language};
//!
//! let sandbox = LocalSandbox::new();
//! let tool = SandboxTool::new(sandbox, Language::Python)
//!     .with_timeout(10_000);
//!
//! let result = tool.run(r#"{"code": "print(1 + 2)"}"#.to_string()).await?;
//! ```

mod local;

#[cfg(feature = "sandbox-e2b")]
mod e2b;

#[cfg(feature = "sandbox-wasm")]
mod wasm;

pub use local::LocalSandbox;

#[cfg(feature = "sandbox-e2b")]
pub use e2b::E2BSandbox;

#[cfg(feature = "sandbox-wasm")]
pub use wasm::WasmSandbox;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::core::tools::{BaseTool, ToolError};

/// Supported programming languages for sandbox execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Python,
    JavaScript,
    Rust,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Python => write!(f, "python"),
            Language::JavaScript => write!(f, "javascript"),
            Language::Rust => write!(f, "rust"),
        }
    }
}

/// Result of a sandboxed code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Standard output captured from the execution.
    pub stdout: String,
    /// Standard error captured from the execution.
    pub stderr: String,
    /// Process exit code (0 = success, non-zero = failure).
    pub exit_code: i32,
    /// Wall-clock execution time in milliseconds.
    pub execution_time_ms: u64,
}

/// Errors that can occur during sandbox execution.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// Execution exceeded the configured time limit.
    #[error("execution timeout after {0}ms")]
    Timeout(u64),

    /// Runtime error during code execution.
    #[error("sandbox error: {0}")]
    Runtime(String),

    /// The requested language is not supported by this sandbox backend.
    #[error("language not supported: {0}")]
    UnsupportedLanguage(String),
}

/// Trait for sandbox backends that execute code in an isolated environment.
///
/// Implementors provide the actual execution mechanism (subprocess, cloud API,
/// WASM runtime, etc.). The trait is object-safe so backends can be used
/// behind `dyn CodeSandbox`.
#[async_trait]
pub trait CodeSandbox: Send + Sync {
    /// Execute the given code in the specified language.
    ///
    /// # Arguments
    /// * `code` - Source code to execute.
    /// * `language` - Programming language of the code.
    /// * `timeout_ms` - Maximum execution time in milliseconds.
    ///
    /// # Returns
    /// A [`RunResult`] on success or a [`SandboxError`] on failure.
    async fn run(&self, code: &str, language: Language, timeout_ms: u64) -> Result<RunResult, SandboxError>;
}

/// Input JSON schema for [`SandboxTool`].
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SandboxInput {
    /// Source code to execute.
    code: String,
}

/// A [`BaseTool`] that executes code in a sandboxed environment.
///
/// Wraps any [`CodeSandbox`] backend and exposes it as a tool usable by agents.
/// The tool name is `"code_interpreter"`.
///
/// # Example
///
/// ```ignore
/// let tool = SandboxTool::new(LocalSandbox::new(), Language::Python)
///     .with_timeout(5_000);
/// ```
pub struct SandboxTool<S: CodeSandbox> {
    sandbox: S,
    language: Language,
    timeout_ms: u64,
}

impl<S: CodeSandbox> SandboxTool<S> {
    /// Create a new sandbox tool with the given backend and default language.
    pub fn new(sandbox: S, language: Language) -> Self {
        Self {
            sandbox,
            language,
            timeout_ms: 30_000, // 30s default
        }
    }

    /// Set the execution timeout in milliseconds.
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

#[async_trait]
impl<S: CodeSandbox + 'static> BaseTool for SandboxTool<S> {
    fn name(&self) -> &str {
        "code_interpreter"
    }

    fn description(&self) -> &str {
        "Execute code in a sandboxed environment. \
         Input JSON: {\"code\": \"...\"}. \
         Returns stdout, stderr, exit_code, and execution_time_ms. \
         Supported languages depend on the sandbox backend."
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: SandboxInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;

        if parsed.code.trim().is_empty() {
            return Err(ToolError::InvalidInput("code must not be empty".to_string()));
        }

        let result = self
            .sandbox
            .run(&parsed.code, self.language, self.timeout_ms)
            .await
            .map_err(|e| match e {
                SandboxError::Timeout(ms) => ToolError::Timeout(ms / 1000),
                SandboxError::Runtime(msg) => ToolError::ExecutionFailed(msg),
                SandboxError::UnsupportedLanguage(lang) => {
                    ToolError::InvalidInput(format!("unsupported language: {}", lang))
                }
            })?;

        serde_json::to_string_pretty(&result)
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to serialize result: {}", e)))
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(SandboxInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_display() {
        assert_eq!(Language::Python.to_string(), "python");
        assert_eq!(Language::JavaScript.to_string(), "javascript");
        assert_eq!(Language::Rust.to_string(), "rust");
    }

    #[test]
    fn test_language_serde() {
        let json = serde_json::to_string(&Language::Python).unwrap();
        assert_eq!(json, "\"python\"");

        let lang: Language = serde_json::from_str("\"javascript\"").unwrap();
        assert_eq!(lang, Language::JavaScript);
    }

    #[test]
    fn test_run_result_serialization() {
        let result = RunResult {
            stdout: "hello\n".to_string(),
            stderr: String::new(),
            exit_code: 0,
            execution_time_ms: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: RunResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.stdout, "hello\n");
        assert_eq!(parsed.exit_code, 0);
        assert_eq!(parsed.execution_time_ms, 42);
    }

    #[test]
    fn test_sandbox_error_display() {
        let err = SandboxError::Timeout(5000);
        assert!(err.to_string().contains("5000ms"));

        let err = SandboxError::Runtime("crashed".to_string());
        assert!(err.to_string().contains("crashed"));

        let err = SandboxError::UnsupportedLanguage("brainfuck".to_string());
        assert!(err.to_string().contains("brainfuck"));
    }

    /// A minimal mock sandbox for testing SandboxTool without a real backend.
    struct MockSandbox;

    #[async_trait]
    impl CodeSandbox for MockSandbox {
        async fn run(&self, code: &str, _language: Language, _timeout_ms: u64) -> Result<RunResult, SandboxError> {
            Ok(RunResult {
                stdout: format!("executed: {}", code),
                stderr: String::new(),
                exit_code: 0,
                execution_time_ms: 1,
            })
        }
    }

    #[tokio::test]
    async fn test_sandbox_tool_name_and_description() {
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        assert_eq!(tool.name(), "code_interpreter");
        assert!(tool.description().contains("sandbox"));
    }

    #[tokio::test]
    async fn test_sandbox_tool_args_schema() {
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        let schema = tool.args_schema();
        assert!(schema.is_some());
        let schema = schema.unwrap();
        assert!(schema["properties"]["code"].is_object());
    }

    #[tokio::test]
    async fn test_sandbox_tool_run_success() {
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        let result = tool.run(r#"{"code": "print(1+1)"}"#.to_string()).await;
        assert!(result.is_ok());
        let output: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(output["exit_code"], 0);
        assert!(output["stdout"].as_str().unwrap().contains("executed"));
    }

    #[tokio::test]
    async fn test_sandbox_tool_run_empty_code() {
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        let result = tool.run(r#"{"code": "  "}"#.to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("empty"));
    }

    #[tokio::test]
    async fn test_sandbox_tool_run_invalid_json() {
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        let result = tool.run("not json".to_string()).await;
        assert!(result.is_err());
    }

    /// Mock sandbox that always times out.
    struct TimeoutSandbox;

    #[async_trait]
    impl CodeSandbox for TimeoutSandbox {
        async fn run(&self, _code: &str, _language: Language, timeout_ms: u64) -> Result<RunResult, SandboxError> {
            Err(SandboxError::Timeout(timeout_ms))
        }
    }

    #[tokio::test]
    async fn test_sandbox_tool_timeout_maps_to_tool_error() {
        let tool = SandboxTool::new(TimeoutSandbox, Language::Python).with_timeout(5000);
        let result = tool.run(r#"{"code": "while True: pass"}"#.to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            ToolError::Timeout(secs) => assert_eq!(secs, 5),
            other => panic!("expected Timeout error, got: {:?}", other),
        }
    }

    /// Mock sandbox that returns unsupported language error.
    struct UnsupportedSandbox;

    #[async_trait]
    impl CodeSandbox for UnsupportedSandbox {
        async fn run(&self, _code: &str, language: Language, _timeout_ms: u64) -> Result<RunResult, SandboxError> {
            Err(SandboxError::UnsupportedLanguage(language.to_string()))
        }
    }

    #[tokio::test]
    async fn test_sandbox_tool_unsupported_language() {
        let tool = SandboxTool::new(UnsupportedSandbox, Language::Rust);
        let result = tool.run(r#"{"code": "fn main(){}"}"#.to_string()).await;
        assert!(result.is_err());
    }
}
