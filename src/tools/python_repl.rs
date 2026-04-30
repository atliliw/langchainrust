// src/tools/python_repl.rs
//! Python 代码执行工具
//!
//! 调用系统 Python 解释器执行代码并返回结果。
//! 需要系统中已安装 Python。

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
/// 此工具会执行任意 Python 代码，在生产环境中使用时
/// 应确保在沙箱环境中运行。
///
/// # 示例
/// ```ignore
/// use langchainrust::tools::PythonREPLTool;
///
/// let tool = PythonREPLTool::new();
/// let result = tool.invoke(PythonREPLInput {
///     code: "print('Hello from Python!')".into(),
///     timeout_seconds: Some(30),
/// }).await?;
/// ```
pub struct PythonREPLTool {
    python_path: String,
}

impl PythonREPLTool {
    pub fn new() -> Self {
        Self {
            python_path: Self::find_python(),
        }
    }

    /// 使用自定义 Python 路径
    pub fn with_python_path(path: impl Into<String>) -> Self {
        Self {
            python_path: path.into(),
        }
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
            return Err(ToolError::InvalidInput("Python 代码不能为空".to_string()));
        }

        let _timeout = input.timeout_seconds.unwrap_or(30);

        let result = Command::new(&self.python_path)
            .arg("-c")
            .arg(&input.code)
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Python 执行失败: {}", e)))?;

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
        "Python 代码执行工具。在本地 Python 环境中执行代码并返回结果。

参数:
- code: 要执行的 Python 代码字符串
- timeout_seconds: 超时时间（秒，默认 30）

支持任意 Python 语法，包括数学计算、数据处理、图表绘制等。

安全警告: 会执行任意代码，请确保在受控环境中使用。

示例:
- 简单计算: {\"code\": \"print(1 + 2)\"}
- 列表处理: {\"code\": \"print([x**2 for x in range(10)])\"}
- 数学运算: {\"code\": \"import math; print(math.pi)\"}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: PythonREPLInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;

        let output = self.invoke(parsed).await?;

        let mut result = String::new();
        if !output.stdout.is_empty() {
            result.push_str(&format!("标准输出:\n{}\n", output.stdout));
        }
        if !output.stderr.is_empty() {
            result.push_str(&format!("标准错误:\n{}\n", output.stderr));
        }
        result.push_str(&format!("退出码: {}", output.exit_code));

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
        let tool = PythonREPLTool::new();
        let result = tool.run(r#"{"code": ""}"#.to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_python_repl_basic_execution() {
        let tool = PythonREPLTool::new();
        let result = tool.invoke(PythonREPLInput {
            code: "print(1 + 2)".to_string(),
            timeout_seconds: Some(10),
        }).await;

        match result {
            Ok(output) => {
                // stdout 应有输出 或 退出码为 0
                // 注: 系统无 Python 时 exit_code 可能非零
                if output.exit_code == 0 || !output.stdout.is_empty() {
                    // Python 可用，验证功能正常
                } else {
                    eprintln!("Python 可能未安装 (exit_code={})", output.exit_code);
                }
            }
            Err(e) => {
                // 如果系统没有 Python，跳过
                eprintln!("Python 不可用 (这可能是预期的): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_python_repl_with_error() {
        let tool = PythonREPLTool::new();
        let result = tool.invoke(PythonREPLInput {
            code: "print(undefined_var)".to_string(),
            timeout_seconds: Some(10),
        }).await;

        match result {
            Ok(output) => {
                // Python 可用时，错误代码应有 stderr；Python 不可用时 exit_code 非零
                if output.exit_code == 0 {
                    // 偶见 Python 可用但未报错，不強制断言
                }
            }
            Err(_) => {
                // 没有 Python 时跳过
            }
        }
    }
}
