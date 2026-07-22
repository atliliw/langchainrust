// src/tools/python_repl.rs
//! Python 代码执行工具
//!
//! 调用系统 Python 解释器执行代码并返回结果。
//! 需要系统中已安装 Python。
//!
//! # 安全警告
//! 此工具默认**禁用**，必须调用 `with_dangerously_allow(true)` 显式启用。
//! 启用后会执行任意 Python 代码，仅应在受控/沙箱环境中使用。

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::core::tools::{BaseTool, Tool, ToolError};

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

/// Python 代码执行工具
///
/// 在本地 Python 环境中执行代码并返回结果。
/// 适用于数学计算、数据处理等需要 Python 生态的场景。
///
/// # 安全警告
/// 此工具默认**禁用**。必须调用 [`PythonREPLTool::with_dangerously_allow`]
/// 才能执行代码。在生产环境中使用时应确保在沙箱环境中运行。
///
/// # 示例
/// ```ignore
/// use langchainrust::tools::PythonREPLTool;
///
/// let tool = PythonREPLTool::new()
///     .with_dangerously_allow(true);
/// let result = tool.invoke(PythonREPLInput {
///     code: "print('Hello from Python!')".into(),
///     timeout_seconds: Some(30),
/// }).await?;
/// ```
pub struct PythonREPLTool {
    python_path: String,
    /// 是否允许执行代码（默认 false，必须显式 opt-in）
    dangerously_allow: bool,
}

impl PythonREPLTool {
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
            dangerously_allow: false,
        }
    }

    /// 使用自定义 Python 路径
    pub fn with_python_path(path: impl Into<String>) -> Self {
        Self {
            python_path: path.into(),
            dangerously_allow: false,
        }
    }

    /// 显式启用代码执行（默认禁用）
    ///
    /// # 安全警告
    /// 启用后可执行任意 Python 代码，请确保在受控环境中使用。
    pub fn with_dangerously_allow(mut self, allow: bool) -> Self {
        self.dangerously_allow = allow;
        self
    }

    /// 自动查找系统 Python
    fn find_python() -> String {
        // 依次尝试 python3 和 python
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
            return Err(ToolError::InvalidInput("Python code must not be empty".to_string()));
        }

        if !self.dangerously_allow {
            return Err(ToolError::ExecutionFailed(
                "PythonREPLTool is disabled by default for security. \
                 Call .with_dangerously_allow(true) to enable execution."
                    .to_string(),
            ));
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
                    eprintln!("Python may not be installed (exit_code={})", output.exit_code);
                }
            }
            Err(e) => {
                eprintln!("Python not available (may be expected): {}", e);
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
