// src/tools/math.rs
//! Advanced math tool for agents.
//!
//! Provides complex mathematical operations including exponent, logarithm, trigonometry, etc.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::core::tools::{BaseTool, Tool, ToolError};

/// Math tool input parameters.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MathInput {
    /// Operation type: "power", "sqrt", "log", "ln", "sin", "cos", "tan", "abs", "factorial", "mod", "gcd", "lcm".
    pub operation: String,
    
    /// First value for calculation.
    pub value: Option<f64>,
    
    /// Second value for operations requiring two parameters (power, mod, gcd, lcm).
    pub value2: Option<f64>,
    
    /// Logarithm base for log operation (default: 10).
    pub base: Option<f64>,
}

/// Math tool output result.
#[derive(Debug, Serialize)]
pub struct MathOutput {
    /// Calculation result.
    pub result: f64,
    
    /// Operation type.
    pub operation: String,
    
    /// Additional details.
    pub details: Option<String>,
}

/// Advanced math tool for agents.
pub struct SimpleMathTool;

impl SimpleMathTool {
    /// Creates a new SimpleMathTool instance.
    pub fn new() -> Self {
        Self
    }
    
    /// 幂运算
    fn power(&self, base: f64, exponent: f64) -> Result<MathOutput, ToolError> {
        let result = base.powf(exponent);
        Ok(MathOutput {
            result,
            operation: "power".to_string(),
            details: Some(format!("{}^{} = {}", base, exponent, result)),
        })
    }
    
    /// 平方根
    fn sqrt(&self, value: f64) -> Result<MathOutput, ToolError> {
        if value < 0.0 {
            return Err(ToolError::InvalidInput(
                "平方根操作要求非负数值".to_string()
            ));
        }
        let result = value.sqrt();
        Ok(MathOutput {
            result,
            operation: "sqrt".to_string(),
            details: Some(format!("√{} = {}", value, result)),
        })
    }
    
    /// 对数（指定底数）
    fn log(&self, value: f64, base: f64) -> Result<MathOutput, ToolError> {
        if value <= 0.0 || base <= 0.0 || base == 1.0 {
            return Err(ToolError::InvalidInput(
                "对数操作要求正数值且底数不为1".to_string()
            ));
        }
        let result = value.log(base);
        Ok(MathOutput {
            result,
            operation: "log".to_string(),
            details: Some(format!("log_{}({}) = {}", base, value, result)),
        })
    }
    
    /// 自然对数
    fn ln(&self, value: f64) -> Result<MathOutput, ToolError> {
        if value <= 0.0 {
            return Err(ToolError::InvalidInput(
                "自然对数操作要求正数值".to_string()
            ));
        }
        let result = value.ln();
        Ok(MathOutput {
            result,
            operation: "ln".to_string(),
            details: Some(format!("ln({}) = {}", value, result)),
        })
    }
    
    /// 正弦函数
    fn sin(&self, value: f64) -> Result<MathOutput, ToolError> {
        let result = value.sin();
        Ok(MathOutput {
            result,
            operation: "sin".to_string(),
            details: Some(format!("sin({}弧度) = {}", value, result)),
        })
    }
    
    /// 余弦函数
    fn cos(&self, value: f64) -> Result<MathOutput, ToolError> {
        let result = value.cos();
        Ok(MathOutput {
            result,
            operation: "cos".to_string(),
            details: Some(format!("cos({}弧度) = {}", value, result)),
        })
    }
    
    /// 正切函数
    fn tan(&self, value: f64) -> Result<MathOutput, ToolError> {
        let result = value.tan();
        Ok(MathOutput {
            result,
            operation: "tan".to_string(),
            details: Some(format!("tan({}弧度) = {}", value, result)),
        })
    }
    
    /// 绝对值
    fn abs(&self, value: f64) -> Result<MathOutput, ToolError> {
        let result = value.abs();
        Ok(MathOutput {
            result,
            operation: "abs".to_string(),
            details: Some(format!("|{}| = {}", value, result)),
        })
    }
    
    /// 阶乘
    fn factorial(&self, value: f64) -> Result<MathOutput, ToolError> {
        if value < 0.0 {
            return Err(ToolError::InvalidInput(
                "阶乘操作要求非负整数".to_string()
            ));
        }
        let n = value as u64;
        if n > 20 {
            // 防止溢出，限制最大值为20
            return Err(ToolError::InvalidInput(
                "阶乘值过大，最大支持20".to_string()
            ));
        }
        let result = self.compute_factorial(n);
        Ok(MathOutput {
            result: result as f64,
            operation: "factorial".to_string(),
            details: Some(format!("{}! = {}", n, result)),
        })
    }
    
    /// 计算阶乘
    fn compute_factorial(&self, n: u64) -> u64 {
        if n == 0 || n == 1 {
            1
        } else {
            n * self.compute_factorial(n - 1)
        }
    }
    
    /// 取模运算
    fn mod_op(&self, a: f64, b: f64) -> Result<MathOutput, ToolError> {
        if b == 0.0 {
            return Err(ToolError::InvalidInput(
                "取模运算的除数不能为零".to_string()
            ));
        }
        let result = a % b;
        Ok(MathOutput {
            result,
            operation: "mod".to_string(),
            details: Some(format!("{} mod {} = {}", a, b, result)),
        })
    }
    
    /// 最大公约数（GCD）
    fn gcd(&self, a: f64, b: f64) -> Result<MathOutput, ToolError> {
        let a_int = a as i64;
        let b_int = b as i64;
        
        if a_int < 0 || b_int < 0 {
            return Err(ToolError::InvalidInput(
                "GCD 操作要求正整数".to_string()
            ));
        }
        
        let result = self.compute_gcd(a_int.abs(), b_int.abs());
        Ok(MathOutput {
            result: result as f64,
            operation: "gcd".to_string(),
            details: Some(format!("gcd({}, {}) = {}", a_int, b_int, result)),
        })
    }
    
    /// 计算 GCD（欧几里得算法）
    fn compute_gcd(&self, a: i64, b: i64) -> i64 {
        if b == 0 {
            a
        } else {
            self.compute_gcd(b, a % b)
        }
    }
    
    /// 最小公倍数（LCM）
    fn lcm(&self, a: f64, b: f64) -> Result<MathOutput, ToolError> {
        let a_int = a as i64;
        let b_int = b as i64;
        
        if a_int <= 0 || b_int <= 0 {
            return Err(ToolError::InvalidInput(
                "LCM 操作要求正整数".to_string()
            ));
        }
        
        let gcd = self.compute_gcd(a_int, b_int);
        let result = (a_int * b_int) / gcd;
        Ok(MathOutput {
            result: result as f64,
            operation: "lcm".to_string(),
            details: Some(format!("lcm({}, {}) = {}", a_int, b_int, result)),
        })
    }
    
    /// 圆周率
    fn pi(&self) -> MathOutput {
        MathOutput {
            result: std::f64::consts::PI,
            operation: "pi".to_string(),
            details: Some("π ≈ 3.141592653589793".to_string()),
        }
    }
    
    /// 自然常数 e
    fn e(&self) -> MathOutput {
        MathOutput {
            result: std::f64::consts::E,
            operation: "e".to_string(),
            details: Some("e ≈ 2.718281828459045".to_string()),
        }
    }
}

impl Default for SimpleMathTool {
    fn default() -> Self {
        Self::new()
    }
}

/// 实现 Tool trait
#[async_trait]
impl Tool for SimpleMathTool {
    type Input = MathInput;
    type Output = MathOutput;
    
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        match input.operation.as_str() {
            "power" => {
                let base = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("power 操作需要 value 参数作为底数".to_string()))?;
                let exp = input.value2.ok_or_else(|| 
                    ToolError::InvalidInput("power 操作需要 value2 参数作为指数".to_string()))?;
                self.power(base, exp)
            }
            "sqrt" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("sqrt 操作需要 value 参数".to_string()))?;
                self.sqrt(value)
            }
            "log" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("log 操作需要 value 参数".to_string()))?;
                let base = input.base.unwrap_or(10.0); // 默认以10为底
                self.log(value, base)
            }
            "ln" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("ln 操作需要 value 参数".to_string()))?;
                self.ln(value)
            }
            "sin" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("sin 操作需要 value 参数（弧度）".to_string()))?;
                self.sin(value)
            }
            "cos" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("cos 操作需要 value 参数（弧度）".to_string()))?;
                self.cos(value)
            }
            "tan" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("tan 操作需要 value 参数（弧度）".to_string()))?;
                self.tan(value)
            }
            "abs" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("abs 操作需要 value 参数".to_string()))?;
                self.abs(value)
            }
            "factorial" => {
                let value = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("factorial 操作需要 value 参数".to_string()))?;
                self.factorial(value)
            }
            "mod" => {
                let a = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("mod 操作需要 value 参数".to_string()))?;
                let b = input.value2.ok_or_else(|| 
                    ToolError::InvalidInput("mod 操作需要 value2 参数".to_string()))?;
                self.mod_op(a, b)
            }
            "gcd" => {
                let a = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("gcd 操作需要 value 参数".to_string()))?;
                let b = input.value2.ok_or_else(|| 
                    ToolError::InvalidInput("gcd 操作需要 value2 参数".to_string()))?;
                self.gcd(a, b)
            }
            "lcm" => {
                let a = input.value.ok_or_else(|| 
                    ToolError::InvalidInput("lcm 操作需要 value 参数".to_string()))?;
                let b = input.value2.ok_or_else(|| 
                    ToolError::InvalidInput("lcm 操作需要 value2 参数".to_string()))?;
                self.lcm(a, b)
            }
            "pi" => Ok(self.pi()),
            "e" => Ok(self.e()),
            _ => Err(ToolError::InvalidInput(
                format!("不支持的操作: {}，请使用: power, sqrt, log, ln, sin, cos, tan, abs, factorial, mod, gcd, lcm, pi, e", input.operation)
            )),
        }
    }
}

/// 实现 BaseTool trait
#[async_trait]
impl BaseTool for SimpleMathTool {
    fn name(&self) -> &str {
        "math"
    }
    
    fn description(&self) -> &str {
        "高级数学工具。支持多种数学运算：
        
操作类型:
- power: 幂运算 (value^value2)
- sqrt: 平方根
- log: 对数（可指定底数，默认为10）
- ln: 自然对数
- sin, cos, tan: 三角函数（参数为弧度）
- abs: 绝对值
- factorial: 阶乘（最大支持20）
- mod: 取模运算
- gcd: 最大公约数
- lcm: 最小公倍数
- pi: 圆周率
- e: 自然常数

示例:
- 幂运算: {\"operation\": \"power\", \"value\": 2, \"value2\": 10}
- 平方根: {\"operation\": \"sqrt\", \"value\": 16}
- 对数: {\"operation\": \"log\", \"value\": 100, \"base\": 10}
- 三角函数: {\"operation\": \"sin\", \"value\": 1.5708}
- GCD: {\"operation\": \"gcd\", \"value\": 12, \"value2\": 18}"
    }
    
    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: MathInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON 解析失败: {}", e)))?;
        
        let output = self.invoke(parsed).await?;
        
        Ok(format!(
            "结果: {}\n详细信息: {}",
            output.result,
            output.details.unwrap_or_default()
        ))
    }
    
    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(MathInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_math_power() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "power".to_string(),
            value: Some(2.0),
            value2: Some(10.0),
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 1024.0);
    }
    
    #[tokio::test]
    async fn test_math_sqrt() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "sqrt".to_string(),
            value: Some(16.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 4.0);
    }
    
    #[tokio::test]
    async fn test_math_log() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "log".to_string(),
            value: Some(100.0),
            value2: None,
            base: Some(10.0),
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 2.0);
    }
    
    #[tokio::test]
    async fn test_math_ln() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "ln".to_string(),
            value: Some(std::f64::consts::E),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!((result.result - 1.0).abs() < 0.0001);
    }
    
    #[tokio::test]
    async fn test_math_sin() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "sin".to_string(),
            value: Some(std::f64::consts::PI / 2.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!((result.result - 1.0).abs() < 0.0001);
    }
    
    #[tokio::test]
    async fn test_math_factorial() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "factorial".to_string(),
            value: Some(5.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 120.0);
    }
    
    #[tokio::test]
    async fn test_math_gcd() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "gcd".to_string(),
            value: Some(12.0),
            value2: Some(18.0),
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 6.0);
    }
    
    #[tokio::test]
    async fn test_math_lcm() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "lcm".to_string(),
            value: Some(4.0),
            value2: Some(6.0),
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 12.0);
    }
    
    #[tokio::test]
    async fn test_math_pi() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "pi".to_string(),
            value: None,
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert!((result.result - std::f64::consts::PI).abs() < 0.0001);
    }
    
    #[tokio::test]
    async fn test_math_abs() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "abs".to_string(),
            value: Some(-5.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await.unwrap();
        assert_eq!(result.result, 5.0);
    }
    
    #[tokio::test]
    async fn test_math_sqrt_negative_error() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "sqrt".to_string(),
            value: Some(-4.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_math_factorial_overflow_error() {
        let tool = SimpleMathTool::new();
        
        let input = MathInput {
            operation: "factorial".to_string(),
            value: Some(25.0),
            value2: None,
            base: None,
        };
        
        let result = tool.invoke(input).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_math_base_tool_run() {
        let tool = SimpleMathTool::new();
        
        let input = "{\"operation\": \"power\", \"value\": 3, \"value2\": 4}".to_string();
        let result = tool.run(input).await.unwrap();
        
        assert!(result.contains("81"));
        assert!(result.contains("3^4"));
    }
}