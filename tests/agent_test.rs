#[path = "common.rs"]
mod common;

use langchainrust::llms::LLM;
use std::sync::Arc;
use langchainrust::agent::{ReActAgent, AgentExecutor};
use langchainrust::tools::{Calculator, Tool}; // ← 替换为你的实际路径

#[tokio::test]
async fn test_react_agent_with_calculator() {
    // Arrange: 创建真实 LLM（使用你的测试配置）
    let llm = LLM::new(common::create_test_llm_config_streaming());

    // 创建工具（必须是你已实现的 Tool）
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(Calculator),
    ];

    // 创建 Agent 和 Executor
    let agent = ReActAgent::new(llm, tools.clone());
    let executor = AgentExecutor::new(Box::new(agent), tools)
        .with_max_iterations(3); // 防止无限循环

    // Act: 执行用户问题
    println!(" 正在处理问题: '37 加 48 等于多少？'");
    let result = executor.run("37 加 48 等于多少？").await;

    // Assert
    match &result {
        Ok(answer) => {
            println!(" 最终答案: {}", answer);
            // 基本断言：答案应包含 "85"
            assert!(
                answer.contains("85") || answer.contains("八十五"),
                "答案应包含计算结果 85，但得到: {}",
                answer
            );
        }
        Err(e) => {
            eprintln!("Agent 执行失败: {}", e);
            panic!("Agent failed: {}", e);
        }
    }

    // 确保没有错误
    assert!(result.is_ok(), "Agent execution failed");
}