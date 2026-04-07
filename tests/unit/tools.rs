//! 单元测试 - Tools

use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool};
use langchainrust::BaseTool;
use std::sync::Arc;

#[test]
fn test_calculator_addition() {
    let calc = Calculator::new();
    assert_eq!(calc.name(), "calculator");
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(calc.run("{\"expression\": \"2 + 3\"}".to_string())).unwrap();
    assert!(result.contains("5"));
}

#[test]
fn test_calculator_multiplication() {
    let calc = Calculator::new();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(calc.run("{\"expression\": \"10 * 5\"}".to_string())).unwrap();
    assert!(result.contains("50"));
}

#[test]
fn test_calculator_division() {
    let calc = Calculator::new();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(calc.run("{\"expression\": \"100 / 4\"}".to_string())).unwrap();
    assert!(result.contains("25"));
}

#[test]
fn test_datetime_now() {
    let dt_tool = DateTimeTool::new();
    assert_eq!(dt_tool.name(), "datetime");
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(dt_tool.run("{\"operation\": \"now\"}".to_string())).unwrap();
    assert!(!result.is_empty());
}

#[test]
fn test_math_sqrt() {
    let math_tool = SimpleMathTool::new();
    assert_eq!(math_tool.name(), "math");
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(math_tool.run("{\"operation\": \"sqrt\", \"value\": 16}".to_string())).unwrap();
    assert!(result.contains("4"));
}

#[test]
fn test_math_factorial() {
    let math_tool = SimpleMathTool::new();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(math_tool.run("{\"operation\": \"factorial\", \"value\": 5}".to_string())).unwrap();
    assert!(result.contains("120"));
}

#[test]
fn test_math_power() {
    let math_tool = SimpleMathTool::new();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(math_tool.run("{\"operation\": \"power\", \"value\": 2, \"value2\": 8}".to_string())).unwrap();
    assert!(result.contains("256"));
}

#[test]
fn test_multiple_tools_collection() {
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    assert_eq!(tools.len(), 3);
    assert_eq!(tools[0].name(), "calculator");
    assert_eq!(tools[1].name(), "datetime");
    assert_eq!(tools[2].name(), "math");
}

#[tokio::test]
async fn test_calculator_async() {
    let calc = Calculator::new();
    let result = calc.run("{\"expression\": \"15 + 27\"}".to_string()).await.unwrap();
    assert!(result.contains("42"));
}