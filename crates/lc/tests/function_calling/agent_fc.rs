//! Function Calling Agent 测试 - 需要 API Key
//!
//! 测试 FunctionCallingAgent 使用原生 Function Calling 的工具调用能力

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool};
use langchainrust::{AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent};
use std::sync::Arc;

/// 测试 FunctionCallingAgent 使用计算器工具
///
/// 测试内容：
/// - Agent 接收数学问题
/// - 通过 Function Calling 调用 Calculator 工具
/// - 返回正确结果 42
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_fc_agent_with_calculator() {
    let llm = TestConfig::get().openai_chat();

    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(3)
        .with_verbose(true);

    let result = executor
        .invoke("What is 25 + 17?".to_string())
        .await
        .unwrap();

    println!("Result: {}", result);
    assert!(result.contains("42"));
}

/// 测试 FunctionCallingAgent 多工具选择
///
/// 测试内容：
/// - Agent 有 Calculator、DateTimeTool、SimpleMathTool 三个工具
/// - 通过 Function Calling 自动选择正确的工具组合
/// - 完成计算和时间查询两个任务
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_fc_agent_multi_tool_selection() {
    let llm = TestConfig::get().openai_chat();

    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(SimpleMathTool::new()),
    ];

    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(5)
        .with_verbose(true);

    let result = executor
        .invoke("Calculate 15 * 4 and tell me the current time".to_string())
        .await
        .unwrap();

    println!("Result: {}", result);
    assert!(!result.is_empty());
}

/// 测试 FunctionCallingAgent 数学运算
///
/// 测试内容：
/// - Agent 使用 SimpleMathTool 执行平方根运算
/// - 验证 sqrt(144) = 12
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_fc_agent_math_operation() {
    let llm = TestConfig::get().openai_chat();

    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(SimpleMathTool::new())];

    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor =
        AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools).with_max_iterations(3);

    let result = executor
        .invoke("What is the square root of 144?".to_string())
        .await
        .unwrap();

    println!("Result: {}", result);
    assert!(result.contains("12"));
}

/// 测试 FunctionCallingAgent 带自定义系统提示词
///
/// 测试内容：
/// - Agent 使用自定义系统提示词
/// - 验证系统提示词影响 Agent 行为
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_fc_agent_with_system_prompt() {
    let llm = TestConfig::get().openai_chat();

    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

    let agent = FunctionCallingAgent::new(
        llm,
        tools.clone(),
        Some("你是一个数学助手，专门帮助用户解决数学问题。".to_string()),
    );
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(3)
        .with_verbose(true);

    let result = executor
        .invoke("帮我计算 100 除以 4".to_string())
        .await
        .unwrap();

    println!("Result: {}", result);
    assert!(result.contains("25"));
}

/// 对比测试：ReActAgent vs FunctionCallingAgent
///
/// 测试内容：
/// - 同样的问题用两种 Agent 分别测试
/// - 对比结果差异
#[tokio::test]
#[ignore = "需要配置 API Key"]
async fn test_compare_react_vs_fc_agent() {
    use langchainrust::ReActAgent;

    let llm1 = TestConfig::get().openai_chat();
    let llm2 = TestConfig::get().openai_chat();
    let tools1: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let tools2: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let question = "计算 37 + 48".to_string();

    // ReActAgent 测试
    let react_agent = ReActAgent::new(llm1, tools1.clone(), None);
    let react_executor = AgentExecutor::new(Arc::new(react_agent) as Arc<dyn BaseAgent>, tools1)
        .with_max_iterations(3)
        .with_verbose(true);

    let react_result = react_executor.invoke(question.clone()).await.unwrap();
    println!("ReActAgent Result: {}", react_result);

    // FunctionCallingAgent 测试
    let fc_agent = FunctionCallingAgent::new(llm2, tools2.clone(), None);
    let fc_executor = AgentExecutor::new(Arc::new(fc_agent) as Arc<dyn BaseAgent>, tools2)
        .with_max_iterations(3)
        .with_verbose(true);

    let fc_result = fc_executor.invoke(question).await.unwrap();
    println!("FunctionCallingAgent Result: {}", fc_result);

    // 两者应该都得到正确结果
    assert!(react_result.contains("85") || react_result.contains("85.0"));
    assert!(fc_result.contains("85") || fc_result.contains("85.0"));
}
