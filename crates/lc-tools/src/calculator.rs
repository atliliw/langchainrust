// lc-tools/src/calculator.rs
//! Calculator tool
//!
//! A math expression calculator backed by the crate-internal
//! [`crate::expr_eval`] evaluator (a dependency-free port of the `meval`
//! expression language).

use async_trait::async_trait;
use lc_core::tools::{BaseTool, Tool, ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Calculator input
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CalculatorInput {
    /// Math expression (e.g., "2 + 3", "sqrt(16)", "3.14 * 10", "2 + 3 * 4")
    pub expression: String,
}

/// Calculator output
#[derive(Debug, Serialize)]
pub struct CalculatorOutput {
    /// Calculation result
    pub result: f64,

    /// Original expression
    pub expression: String,
}

/// Calculator tool
///
/// Evaluates math expressions. Supports: basic arithmetic (+, -, *, /, %),
/// power (^), functions (sin, cos, tan, sqrt, log, exp, abs, …), and
/// constants (pi, e).
pub struct Calculator;

impl Calculator {
    /// Creates a new calculator tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for Calculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement Tool trait (type-safe version)
#[async_trait]
impl Tool for Calculator {
    type Input = CalculatorInput;
    type Output = CalculatorOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = Self::evaluate_expression(&input.expression)?;

        Ok(CalculatorOutput {
            result,
            expression: input.expression,
        })
    }
}

/// Implement BaseTool trait (string version, for Agent)
#[async_trait]
impl BaseTool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Calculate math expressions. Supports basic arithmetic, power, trig functions, sqrt, log, exp, abs, and constants (pi, e).

Examples:
- '2 + 3' -> 5
- '2 + 3 * 4' -> 14
- 'sqrt(16)' -> 4
- '3.14 * 10' -> 31.4
- 'sin(pi/2)' -> 1
- '2^10' -> 1024
- 'log(e)' -> 1

Input format: JSON object with 'expression' field
Example: {\"expression\": \"2 + 3\"}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        // Parse input
        let parsed: CalculatorInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse error: {}", e)))?;

        // Execute calculation
        let output = self.invoke(parsed).await?;

        // Return result string
        Ok(format!("{} = {}", output.expression, output.result))
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(CalculatorInput)).ok()
    }
}

impl Calculator {
    /// Evaluate a math expression
    fn evaluate_expression(expr: &str) -> Result<f64, ToolError> {
        crate::expr_eval::eval(expr).map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to evaluate expression '{}': {}", expr, e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_addition() {
        assert_eq!(Calculator::evaluate_expression("2 + 3").unwrap(), 5.0);
    }

    #[test]
    fn test_operator_precedence() {
        // 2 + 3 * 4 = 14 (not 20)
        assert_eq!(Calculator::evaluate_expression("2 + 3 * 4").unwrap(), 14.0);
    }

    #[test]
    fn test_chained_addition() {
        assert_eq!(Calculator::evaluate_expression("1 + 2 + 3").unwrap(), 6.0);
    }

    #[test]
    fn test_subtraction() {
        assert_eq!(Calculator::evaluate_expression("10 - 3").unwrap(), 7.0);
    }

    #[test]
    fn test_multiplication() {
        let result = Calculator::evaluate_expression("3.14 * 10").unwrap();
        assert!((result - 31.4).abs() < 1e-10);
    }

    #[test]
    fn test_division() {
        let result = Calculator::evaluate_expression("10 / 3").unwrap();
        assert!((result - 3.3333333333333335).abs() < 1e-10);
    }

    #[test]
    fn test_power() {
        assert_eq!(Calculator::evaluate_expression("2^10").unwrap(), 1024.0);
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(Calculator::evaluate_expression("sqrt(16)").unwrap(), 4.0);
    }

    #[test]
    fn test_sin_pi() {
        let result = Calculator::evaluate_expression("sin(pi/2)").unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plain_number() {
        assert_eq!(Calculator::evaluate_expression("42").unwrap(), 42.0);
    }

    #[test]
    fn test_invalid_expression() {
        assert!(Calculator::evaluate_expression("hello").is_err());
    }

    #[tokio::test]
    async fn test_tool_run() {
        let tool = Calculator::new();
        let result = tool
            .run(r#"{"expression": "2 + 3"}"#.to_string())
            .await
            .unwrap();
        assert!(result.contains("5"));
    }
}
