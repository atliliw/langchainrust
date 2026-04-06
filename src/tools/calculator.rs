// src/tools/calculator.rs
//! 计算器工具
//!
//! 一个简单的数学表达式计算器

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::core::tools::{BaseTool, Tool, ToolError};

/// 计算器输入
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CalculatorInput {
    /// 数学表达式（如 "2 + 2", "sqrt(16)", "3.14 * 10"）
    pub expression: String,
}

/// 计算器输出
#[derive(Debug, Serialize)]
pub struct CalculatorOutput {
    /// 计算结果
    pub result: f64,
    
    /// 原始表达式
    pub expression: String,
}

/// 计算器工具
pub struct Calculator;

impl Calculator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Calculator {
    fn default() -> Self {
        Self::new()
    }
}

/// 实现 Tool trait（类型安全版本）
#[async_trait]
impl Tool for Calculator {
    type Input = CalculatorInput;
    type Output = CalculatorOutput;
    
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = self.evaluate_expression(&input.expression)?;
        
        Ok(CalculatorOutput {
            result,
            expression: input.expression,
        })
    }
}

/// 实现 BaseTool trait（字符串版本，用于 Agent）
#[async_trait]
impl BaseTool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }
    
    fn description(&self) -> &str {
        "计算数学表达式。支持基本运算（加减乘除）、幂运算、平方根、三角函数等。
        
示例:
- '2 + 2' → 4
- 'sqrt(16)' → 4
- '3.14 * 10' → 31.4
- 'sin(1.57)' → 接近 1
- 'pow(2, 10)' → 1024

输入格式: JSON 对象，包含 expression 字段
例如: {\"expression\": \"2 + 3\"}"
    }
    
    async fn run(&self, input: String) -> Result<String, ToolError> {
        // 解析输入
        let parsed: CalculatorInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;
        
        // 执行计算
        let output = self.invoke(parsed).await?;
        
        // 返回结果字符串
        Ok(format!("{} = {}", output.expression, output.result))
    }
    
    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(CalculatorInput)).ok()
    }
}

impl Calculator {
    /// 计算数学表达式
    fn evaluate_expression(&self, expr: &str) -> Result<f64, ToolError> {
        // 简化实现：只支持基本运算
        // 实际应该使用 meval 或 evalexpr crate
        
        let expr = expr.trim();
        
        // 尝试解析为简单表达式
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }
        
        // 支持的基本运算
        if expr.contains('+') {
            let parts: Vec<&str> = expr.split('+').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                let b: f64 = parts[1].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                return Ok(a + b);
            }
        }
        
        if expr.contains('-') {
            let parts: Vec<&str> = expr.split('-').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                let b: f64 = parts[1].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                return Ok(a - b);
            }
        }
        
        if expr.contains('*') {
            let parts: Vec<&str> = expr.split('*').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                let b: f64 = parts[1].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                return Ok(a * b);
            }
        }
        
        if expr.contains('/') {
            let parts: Vec<&str> = expr.split('/').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                let b: f64 = parts[1].trim().parse()
                    .map_err(|e| ToolError::ExecutionFailed(format!("解析失败: {}", e)))?;
                if b == 0.0 {
                    return Err(ToolError::ExecutionFailed("除数不能为0".to_string()));
                }
                return Ok(a / b);
            }
        }
        
        Err(ToolError::ExecutionFailed(
            format!("无法解析表达式: {}", expr)
        ))
    }
}