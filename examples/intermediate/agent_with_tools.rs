// examples/intermediate/agent_with_tools.rs
//! 中级示例 1: Agent 与工具
//!
//! 运行: cargo run --example agent_with_tools
//!
//! 功能: 演示如何创建 ReActAgent 并使用工具回答问题

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, SimpleMathTool,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 中级示例 1: Agent 与工具 ===\n");
    
    // 1. 创建 LLM 配置
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.0),  // Agent 通常用较低的温度
        max_tokens: Some(1000),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm = OpenAIChat::new(config);
    
    // 2. 创建工具列表
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),
        Arc::new(DateTimeTool::new()),
        Arc::new(SimpleMathTool::new()),
    ];
    
    println!("可用工具:");
    for tool in &tools {
        println!("  - {}: {}", tool.name(), tool.description().lines().next().unwrap_or(""));
    }
    println!();
    
    // 3. 创建 ReActAgent
    let agent: Arc<dyn BaseAgent> = Arc::new(ReActAgent::new(llm, tools.clone(), None));
    
    // 4. 创建 AgentExecutor
    let executor = AgentExecutor::new(agent, tools)
        .with_verbose(true)         // 打印详细日志
        .with_max_iterations(5);    // 最大迭代次数
    
    // 5. 执行问题
    let questions = vec![
        "计算 37 + 48 等于多少？",
        "今天是星期几？",
        "计算 2 的 10 次方等于多少？",
    ];
    
    for question in questions {
        println!("\n--- 问题: {} ---\n", question);
        
        match executor.invoke(question.to_string()).await {
            Ok(answer) => {
                println!("\n答案: {}", answer);
            }
            Err(e) => {
                eprintln!("错误: {}", e);
            }
        }
        
        println!("\n{}", "=".repeat(50));
    }
    
    println!("\n=== 示例完成 ===");
    Ok(())
}