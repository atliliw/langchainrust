// lc-tools/src/sandbox/mod.rs
//! Code Interpreter Sandbox for pluggable code execution.
//!
//! Provides a pluggable architecture for executing code. The [`CodeSandbox`] trait
//! defines the interface for sandbox backends, and [`SandboxTool`] wraps any sandbox
//! implementation as a [`BaseTool`] usable by agents.
//!
//! # Security
//!
//! [`SandboxTool`] is **disabled by default**: code does not run until
//! [`SandboxTool::with_dangerously_allow`] is called. This is deliberate — the bundled
//! [`LocalSandbox`] backend is a plain subprocess + timeout with **no OS-level
//! isolation**. Treat it as *convenience*, not a security boundary: untrusted code
//! must run in a real sandbox (container / VM / WASM).
//!
//! # Backends
//!
//! - **[`LocalSandbox`]**: the current only backend, a subprocess + timeout.
//!
//! > The former `WasmSandbox` / `E2BSandbox` were hollow shells with a complete interface but
//! > a permanent "not implemented" body (review Q2). They were removed together with the
//! > `sandbox-wasm` / `sandbox-e2b` features — a backend that promises but cannot deliver
//! > damages trust the most; they will come back once actually implemented.

mod local;

pub use local::LocalSandbox;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use lc_core::tools::{BaseTool, ToolError};

/// Supported programming languages for sandbox execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Python.
    Python,
    /// JavaScript.
    JavaScript,
    /// Rust.
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
#[non_exhaustive]
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
#[async_trait]
pub trait CodeSandbox: Send + Sync {
    /// Execute the given code in the specified language.
    async fn run(
        &self,
        code: &str,
        language: Language,
        timeout_ms: u64,
    ) -> Result<RunResult, SandboxError>;
}

/// Input JSON schema for [`SandboxTool`].
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SandboxInput {
    /// Source code to execute.
    code: String,
}

/// A [`BaseTool`] that executes code in a sandboxed environment.
///
/// **Disabled by default.** Call [`SandboxTool::with_dangerously_allow`] to enable
/// execution. See the [module docs](self) for the security model.
pub struct SandboxTool<S: CodeSandbox> {
    sandbox: S,
    language: Language,
    timeout_ms: u64,
    /// Whether code execution is allowed (default false, must explicitly opt in).
    dangerously_allow: bool,
}

impl<S: CodeSandbox> SandboxTool<S> {
    /// Create a new sandbox tool with the given backend and default language.
    ///
    /// Execution is **disabled** until
    /// [`with_dangerously_allow`](Self::with_dangerously_allow) is called.
    pub fn new(sandbox: S, language: Language) -> Self {
        Self {
            sandbox,
            language,
            timeout_ms: 30_000,
            dangerously_allow: false,
        }
    }

    /// Set the execution timeout in milliseconds.
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Explicitly enables code execution (disabled by default).
    pub fn with_dangerously_allow(mut self, allow: bool) -> Self {
        self.dangerously_allow = allow;
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
         Disabled by default for security; call .with_dangerously_allow(true) to enable. \
         Input JSON: {\"code\": \"...\"}. \
         Returns stdout, stderr, exit_code, and execution_time_ms. \
         Supported languages depend on the sandbox backend."
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: SandboxInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;

        if parsed.code.trim().is_empty() {
            return Err(ToolError::InvalidInput(
                "code must not be empty".to_string(),
            ));
        }

        if !self.dangerously_allow {
            return Err(ToolError::ExecutionFailed(
                "SandboxTool is disabled by default for security. \
                 Call .with_dangerously_allow(true) to enable execution."
                    .to_string(),
            ));
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

    struct MockSandbox;

    #[async_trait]
    impl CodeSandbox for MockSandbox {
        async fn run(
            &self,
            code: &str,
            _language: Language,
            _timeout_ms: u64,
        ) -> Result<RunResult, SandboxError> {
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
    async fn test_sandbox_tool_disabled_by_default() {
        // 0.20.0 S4 P-C1: execution is off until explicitly enabled.
        let tool = SandboxTool::new(MockSandbox, Language::Python);
        let result = tool.run(r#"{"code": "print(1+1)"}"#.to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("disabled by default"),
            "expected disabled-by-default gate, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_sandbox_tool_run_success() {
        let tool = SandboxTool::new(MockSandbox, Language::Python).with_dangerously_allow(true);
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

    struct TimeoutSandbox;

    #[async_trait]
    impl CodeSandbox for TimeoutSandbox {
        async fn run(
            &self,
            _code: &str,
            _language: Language,
            timeout_ms: u64,
        ) -> Result<RunResult, SandboxError> {
            Err(SandboxError::Timeout(timeout_ms))
        }
    }

    #[tokio::test]
    async fn test_sandbox_tool_timeout_maps_to_tool_error() {
        let tool = SandboxTool::new(TimeoutSandbox, Language::Python)
            .with_timeout(5000)
            .with_dangerously_allow(true);
        let result = tool
            .run(r#"{"code": "while True: pass"}"#.to_string())
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            ToolError::Timeout(secs) => assert_eq!(secs, 5),
            other => panic!("expected Timeout error, got: {:?}", other),
        }
    }

    struct UnsupportedSandbox;

    #[async_trait]
    impl CodeSandbox for UnsupportedSandbox {
        async fn run(
            &self,
            _code: &str,
            language: Language,
            _timeout_ms: u64,
        ) -> Result<RunResult, SandboxError> {
            Err(SandboxError::UnsupportedLanguage(language.to_string()))
        }
    }

    #[tokio::test]
    async fn test_sandbox_tool_unsupported_language() {
        let tool =
            SandboxTool::new(UnsupportedSandbox, Language::Rust).with_dangerously_allow(true);
        let result = tool.run(r#"{"code": "fn main(){}"}"#.to_string()).await;
        assert!(result.is_err());
    }
}
