//! Agent 测试 - 需要 API Key
//!
//! 测试 ReActAgent 和 AgentExecutor 的工具调用能力

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool};
use langchainrust::{BaseAgent, BaseTool, AgentExecutor, ReActAgent};
use std::sync::Arc;

/// 测试 Agent 使用计算器工具
///
/// 测试内容：
/// - Agent 接收数学问题
/// - 自动选择 Calculator 工具执行计算
/// - 返回正确结果 42
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_agent_with_calculator() {
    let llm = TestConfig::get().openai_chat();
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
    ];
    
    let agent = ReActAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(3)
        .with_verbose(true);
    
    let result = executor.invoke("What is 25 + 17?".to_string()).await.unwrap();
    
    println!("Result: {}", result);
    assert!(result.contains("42"));
}

/// 测试 Agent 多工具选择
///
/// 测试内容：
/// - Agent 有 Calculator、DateTimeTool、SimpleMathTool 三个工具
/// - 根据问题自动选择正确的工具组合
/// - 完成计算和时间查询两个任务
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_agent_multi_tool_selection() {
    let llm = TestConfig::get().openai_chat();
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    let agent = ReActAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(5)
        .with_verbose(true);
    
    let result = executor.invoke("Calculate 15 * 4 and tell me the current time".to_string())
        .await
        .unwrap();
    
    println!("Result: {}", result);
    assert!(!result.is_empty());
}

/// 测试 Agent 数学运算
///
/// 测试内容：
/// - Agent 使用 SimpleMathTool 执行平方根运算
/// - 验证 sqrt(144) = 12
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_agent_math_operation() {
    let llm = TestConfig::get().openai_chat();
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(SimpleMathTool::new()),
    ];
    
    let agent = ReActAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(3);
    
    let result = executor.invoke("What is the square root of 144?".to_string()).await.unwrap();
    
    println!("Result: {}", result);
    assert!(result.contains("12"));
}