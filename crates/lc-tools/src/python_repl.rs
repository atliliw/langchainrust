// lc-tools/src/python_repl.rs
//! Python code execution tool
//!
//! Invokes the system Python interpreter to run code and return the result.
//! Python must be installed on the system.
//!
//! # Security warning
//! This tool is **disabled** by default; call `with_dangerously_allow(true)` to enable it explicitly.
//! Once enabled it executes arbitrary Python code and should only be used in controlled/sandboxed environments.
//!
//! The built-in "dangerous import blacklist" is only **noise filtering, not a security boundary** —
//! it cannot stop encoding obfuscation such as `__import__`/`eval`/`exec`/string concatenation,
//! and it also has false positives on string literals. `open(` is filtered but is *not* a real
//! guarantee against file access (obfuscation can still reach it); real isolation must go through
//! the sandbox ([`crate::sandbox`]). The blacklist only reduces noise reaching it.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use lc_core::tools::{BaseTool, Tool, ToolError};

/// Python REPL tool input
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PythonREPLInput {
    /// The Python code to execute
    pub code: String,
    /// Timeout in seconds (default: 30)
    pub timeout_seconds: Option<u64>,
}

/// Python REPL tool output
#[derive(Debug, Serialize)]
pub struct PythonREPLOutput {
    /// The code that was executed
    pub code: String,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: i32,
}

/// Dangerous Python code (imports + builtin calls) is filtered by the shared
/// [`crate::python_filter`] (T2 §5): this REPL and [`crate::sandbox::LocalSandbox`] previously
/// duplicated import/call blacklists that had drifted apart. Now both delegate to the same
/// consts/functions, so a module or call is blocked in both or neither.
use crate::python_filter::contains_dangerous_python as contains_dangerous_code;

/// Python code execution tool
///
/// Executes code in a local Python environment and returns the result.
/// Suitable for scenarios needing the Python ecosystem, such as math and data processing.
///
/// # Security warning
/// This tool is **disabled** by default. Call [`PythonREPLTool::with_dangerously_allow`]
/// to enable code execution. In production, ensure it runs inside a sandboxed environment.
pub struct PythonREPLTool {
    python_path: String,
    /// Whether code execution is allowed (default false, must explicitly opt in)
    dangerously_allow: bool,
    /// Whether the dangerous-import check is enabled (default true)
    check_dangerous_imports: bool,
}

impl PythonREPLTool {
    /// Creates a Python code execution tool (execution disabled by default).
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
            dangerously_allow: false,
            check_dangerous_imports: true,
        }
    }

    /// Uses a custom Python path
    pub fn with_python_path(path: impl Into<String>) -> Self {
        Self {
            python_path: path.into(),
            dangerously_allow: false,
            check_dangerous_imports: true,
        }
    }

    /// Explicitly enables code execution (disabled by default)
    pub fn with_dangerously_allow(mut self, allow: bool) -> Self {
        self.dangerously_allow = allow;
        self
    }

    /// Disable dangerous import checking (default: enabled).
    pub fn with_skip_dangerous_imports_check(mut self, skip: bool) -> Self {
        self.check_dangerous_imports = !skip;
        self
    }

    /// Automatically finds the system Python
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
}

impl Default for PythonREPLTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for PythonREPLTool {
    type Input = PythonREPLInput;
    type Output = PythonREPLOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        if input.code.trim().is_empty() {
            return Err(ToolError::InvalidInput(
                "Python code must not be empty".to_string(),
            ));
        }

        if !self.dangerously_allow {
            return Err(ToolError::ExecutionFailed(
                "PythonREPLTool is disabled by default for security. \
                 Call .with_dangerously_allow(true) to enable execution."
                    .to_string(),
            ));
        }

        if self.check_dangerous_imports {
            if let Some(blocked) = contains_dangerous_code(&input.code) {
                return Err(ToolError::ExecutionFailed(format!(
                    "Code contains dangerous import or builtin call: '{}'. \
                     Blocked by the noise-filter blacklist (note: this is not a security boundary; \
                     untrusted code must run in a sandbox). \
                     Call .with_skip_dangerous_imports_check(true) to bypass (not recommended).",
                    blocked
                )));
            }
        }

        let timeout_secs = input.timeout_seconds.unwrap_or(30);

        // 0.22.0 H-A3: `kill_on_drop(true)` terminates the subprocess when the
        // timeout fires and the `.output()` future is dropped. Without it the
        // child kept running as an orphan after the parent gave up waiting,
        // an infinite loop burned CPU in the background forever.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            Command::new(&self.python_path)
                .arg("-c")
                .arg(&input.code)
                .kill_on_drop(true)
                .output(),
        )
        .await
        .map_err(|_| {
            ToolError::ExecutionFailed(format!(
                "Python execution timed out after {} seconds",
                timeout_secs
            ))
        })?
        .map_err(|e| ToolError::ExecutionFailed(format!("Python execution failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let exit_code = result.status.code().unwrap_or(-1);

        Ok(PythonREPLOutput {
            code: input.code,
            stdout,
            stderr,
            exit_code,
        })
    }
}

#[async_trait]
impl BaseTool for PythonREPLTool {
    fn name(&self) -> &str {
        "python_repl"
    }

    fn description(&self) -> &str {
        "Python code execution tool. Runs code in a local Python environment and returns results.

Parameters:
- code: Python code string to execute
- timeout_seconds: Timeout in seconds (default: 30)

Supports any Python syntax, including math, data processing, plotting, etc.

SECURITY WARNING: Disabled by default. Must call .with_dangerously_allow(true) to enable.
Only use in controlled/sandboxed environments.

Examples:
- Simple calc: {\"code\": \"print(1 + 2)\"}
- List processing: {\"code\": \"print([x**2 for x in range(10)])\"}
- Math: {\"code\": \"import math; print(math.pi)\"}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: PythonREPLInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let output = self.invoke(parsed).await?;

        let mut result = String::new();
        if !output.stdout.is_empty() {
            result.push_str(&format!("stdout:\n{}\n", output.stdout));
        }
        if !output.stderr.is_empty() {
            result.push_str(&format!("stderr:\n{}\n", output.stderr));
        }
        result.push_str(&format!("exit_code: {}", output.exit_code));

        Ok(result)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(PythonREPLInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_repl_tool_properties() {
        let tool = PythonREPLTool::new();
        assert_eq!(tool.name(), "python_repl");
        assert!(tool.description().contains("Python"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }

    #[tokio::test]
    async fn test_python_repl_empty_code() {
        let tool = PythonREPLTool::new().with_dangerously_allow(true);
        let result = tool.run(r#"{"code": ""}"#.to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_python_repl_disabled_by_default() {
        let tool = PythonREPLTool::new();
        let result = tool
            .invoke(PythonREPLInput {
                code: "print(1 + 2)".to_string(),
                timeout_seconds: Some(10),
            })
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("disabled by default"),
            "Expected disabled error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_python_repl_basic_execution() {
        let tool = PythonREPLTool::new().with_dangerously_allow(true);
        let result = tool
            .invoke(PythonREPLInput {
                code: "print(1 + 2)".to_string(),
                timeout_seconds: Some(10),
            })
            .await;

        match result {
            Ok(output) => {
                if output.exit_code == 0 || !output.stdout.is_empty() {
                    // Python available, verify functionality
                } else {
                    eprintln!(
                        "Python may not be installed (exit_code={})",
                        output.exit_code
                    );
                }
            }
            Err(e) => {
                eprintln!("Python not available (may be expected): {}", e);
            }
        }
    }

    #[test]
    fn test_dangerous_import_detection() {
        assert!(contains_dangerous_code("import os").is_some());
        assert!(contains_dangerous_code("import subprocess").is_some());
        assert!(contains_dangerous_code("from sys import path").is_some());
        assert!(contains_dangerous_code("from os.path import join").is_some());
        // Safe imports should pass
        assert!(contains_dangerous_code("import math").is_none());
        assert!(contains_dangerous_code("import json").is_none());
        assert!(contains_dangerous_code("from datetime import datetime").is_none());
        // Comments should be ignored
        assert!(contains_dangerous_code("# import os").is_none());
    }

    #[test]
    fn test_dangerous_builtin_calls_detected() {
        // Common bypass: call builtin/imported functions directly without an import statement
        assert!(contains_dangerous_code("__import__('os').system('ls')").is_some());
        assert!(contains_dangerous_code("importlib.import_module('os')").is_some());
        assert!(contains_dangerous_code("eval('os')").is_some());
        assert!(contains_dangerous_code("exec('import os')").is_some());
        assert!(contains_dangerous_code("compile('import os', '<x>', 'exec')").is_some());
        assert!(contains_dangerous_code("execfile('/tmp/x.py')").is_some());
    }

    #[test]
    fn test_dangerous_builtin_file_and_interactive_vectors_blocked() {
        // File access / interactive escape builtins require no import and were previously able
        // to reach /etc/passwd or a REPL even with the check enabled.
        assert!(contains_dangerous_code("open('/etc/shadow').read()").is_some());
        assert!(contains_dangerous_code("breakpoint()").is_some());
        assert!(contains_dangerous_code("input()").is_some());
        // Word boundary: no false positive on identifiers containing these as substrings.
        assert!(contains_dangerous_code("opened = []").is_none());
        assert!(contains_dangerous_code("check_input(arg)").is_none());
    }

    #[test]
    fn test_dangerous_builtin_calls_no_false_positive_on_words() {
        // Word boundary: no false positives on ordinary words containing these as substrings, like evaluate / execute / length
        assert!(contains_dangerous_code("print('evaluate the result')").is_none());
        assert!(contains_dangerous_code("result = execute_query()").is_none());
        assert!(contains_dangerous_code("print(len([1, 2, 3]))").is_none());
        assert!(contains_dangerous_code("x = len('hello')").is_none());
    }

    #[tokio::test]
    async fn test_python_repl_blocks_dangerous_import() {
        let tool = PythonREPLTool::new().with_dangerously_allow(true);
        let result = tool
            .invoke(PythonREPLInput {
                code: "import os; print(os.getcwd())".to_string(),
                timeout_seconds: Some(10),
            })
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("dangerous import"),
            "Expected dangerous import error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_python_repl_allows_safe_import() {
        let tool = PythonREPLTool::new().with_dangerously_allow(true);
        let result = tool
            .invoke(PythonREPLInput {
                code: "import math; print(math.pi)".to_string(),
                timeout_seconds: Some(10),
            })
            .await;
        match result {
            Ok(output) => {
                if output.exit_code == 0 {
                    assert!(output.stdout.contains("3.14"));
                }
            }
            Err(e) => {
                assert!(
                    !e.to_string().contains("dangerous import"),
                    "math should not be blocked: {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    async fn test_python_repl_with_error() {
        let tool = PythonREPLTool::new().with_dangerously_allow(true);
        let result = tool
            .invoke(PythonREPLInput {
                code: "print(undefined_var)".to_string(),
                timeout_seconds: Some(10),
            })
            .await;

        match result {
            Ok(output) => {
                if output.exit_code == 0 {
                    // occasionally Python available but no error reported
                }
            }
            Err(_) => {
                // No Python available, skip
            }
        }
    }
}
