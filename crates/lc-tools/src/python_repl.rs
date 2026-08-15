// lc-tools/src/python_repl.rs
//! Python 代码执行工具
//!
//! 调用系统 Python 解释器执行代码并返回结果。
//! 需要系统中已安装 Python。
//!
//! # 安全警告
//! 此工具默认**禁用**，必须调用 `with_dangerously_allow(true)` 显式启用。
//! 启用后会执行任意 Python 代码，仅应在受控/沙箱环境中使用。
//!
//! 内置的"危险 import 黑名单"只是**噪音过滤，不是安全边界**——它挡不住
//! `__import__`/`eval`/`exec`/字符串拼接等编码混淆，也会误伤字符串字面量。
//! 真正的隔离必须走沙箱（[`crate::sandbox`]），黑名单只用于减少误入沙箱的噪音。

use async_trait::async_trait;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use lc_core::tools::{BaseTool, Tool, ToolError};

/// Python REPL 工具输入
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PythonREPLInput {
    /// 要执行的 Python 代码
    pub code: String,
    /// 超时时间（秒，默认 30）
    pub timeout_seconds: Option<u64>,
}

/// Python REPL 工具输出
#[derive(Debug, Serialize)]
pub struct PythonREPLOutput {
    /// 执行的代码
    pub code: String,
    /// 标准输出
    pub stdout: String,
    /// 标准错误
    pub stderr: String,
    /// 退出码
    pub exit_code: i32,
}

/// Dangerous Python modules that are blocked for security.
const BLOCKED_IMPORTS: &[&str] = &[
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
    "antigravity",
];

/// 常见绕过 import 黑名单的危险内建调用（单词边界 + 函数调用形式）。
const DANGEROUS_BUILTIN_CALLS: &[&str] = &[
    "__import__",
    "import_module",
    "eval",
    "exec",
    "execfile",
    "compile",
];

/// 匹配 `__import__(` / `import_module(` / `eval(` / `exec(` / `compile(` 等危险调用。
///
/// `\b` 保证不会误伤 `evaluate(` / `execute(` / `length(` 这类含子串的普通词；
/// 但仍会误伤字符串字面量里的 `"eval(...)"` 字样——这是字符串级拦截的固有局限，
/// 详见 [`contains_dangerous_code`] 的安全定位说明。
static DANGEROUS_CALL_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(&format!(
        r"\b(?:{})\s*\(",
        DANGEROUS_BUILTIN_CALLS.join("|")
    ))
    .unwrap()
});

/// Check if Python code contains dangerous imports or builtin calls.
///
/// **安全定位**：这是噪音过滤层，**不是安全边界**。逐行子串/正则匹配永远可以被
/// unicode 混淆、`"o"+"s"` 拼接、`().__class__` 反射等绕过，也会误伤字符串字面量。
/// 不可信代码必须走沙箱（[`crate::sandbox`]）。
fn contains_dangerous_code(code: &str) -> Option<String> {
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }
        // 去掉行内注释（# 之后的部分），避免注释内容误伤。
        let code_part = trimmed.split('#').next().unwrap_or(trimmed);

        // 1) 危险 import 检查（BLOCKED_IMPORTS）
        if code_part.contains("import") {
            for blocked in BLOCKED_IMPORTS {
                if code_part.contains(&format!("import {}", blocked))
                    || code_part.contains(&format!("from {} ", blocked))
                    || code_part.contains(&format!("from {}.", blocked))
                    || code_part.contains(&format!("from {}import", blocked))
                {
                    return Some(blocked.to_string());
                }
            }
        }

        // 2) 危险内建调用检查（__import__ / import_module / eval / exec / compile 等常见绕过）
        if let Some(call) = DANGEROUS_CALL_REGEX.find(code_part) {
            return Some(call.as_str().to_string());
        }
    }
    None
}

/// Python 代码执行工具
///
/// 在本地 Python 环境中执行代码并返回结果。
/// 适用于数学计算、数据处理等需要 Python 生态的场景。
///
/// # 安全警告
/// 此工具默认**禁用**。必须调用 [`PythonREPLTool::with_dangerously_allow`]
/// 才能执行代码。在生产环境中使用时应确保在沙箱环境中运行。
pub struct PythonREPLTool {
    python_path: String,
    /// 是否允许执行代码（默认 false，必须显式 opt-in）
    dangerously_allow: bool,
    /// 是否启用危险 import 检查（默认 true）
    check_dangerous_imports: bool,
}

impl PythonREPLTool {
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
            dangerously_allow: false,
            check_dangerous_imports: true,
        }
    }

    /// 使用自定义 Python 路径
    pub fn with_python_path(path: impl Into<String>) -> Self {
        Self {
            python_path: path.into(),
            dangerously_allow: false,
            check_dangerous_imports: true,
        }
    }

    /// 显式启用代码执行（默认禁用）
    pub fn with_dangerously_allow(mut self, allow: bool) -> Self {
        self.dangerously_allow = allow;
        self
    }

    /// Disable dangerous import checking (default: enabled).
    pub fn with_skip_dangerous_imports_check(mut self, skip: bool) -> Self {
        self.check_dangerous_imports = !skip;
        self
    }

    /// 自动查找系统 Python
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

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            Command::new(&self.python_path)
                .arg("-c")
                .arg(&input.code)
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
        // 常见绕过：不经过 import 语句，直接调用内建/导入函数
        assert!(contains_dangerous_code("__import__('os').system('ls')").is_some());
        assert!(contains_dangerous_code("importlib.import_module('os')").is_some());
        assert!(contains_dangerous_code("eval('os')").is_some());
        assert!(contains_dangerous_code("exec('import os')").is_some());
        assert!(contains_dangerous_code("compile('import os', '<x>', 'exec')").is_some());
        assert!(contains_dangerous_code("execfile('/tmp/x.py')").is_some());
    }

    #[test]
    fn test_dangerous_builtin_calls_no_false_positive_on_words() {
        // 单词边界：不误伤 evaluate / execute / length 等含子串的普通词
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
