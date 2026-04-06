// examples/advanced/multi_tool_agent.rs
//! 高级示例 2: 多工具 Agent
//!
//! 运行: cargo run --example multi_tool_agent
//!
//! 功能: 演示 Agent 如何自动选择和使用多个工具完成复杂任务

use langchainrust::{
    OpenAIChat, OpenAIConfig, BaseChatModel,
    ReActAgent, AgentExecutor, BaseAgent, BaseTool,
    Calculator, DateTimeTool, SimpleMathTool, URLFetchTool,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 高级示例 2: 多工具 Agent ===\n");
    
    // 1. 创建 LLM
    let config = OpenAIConfig {
        api_key: std::env::var("OPENAI_API_KEY")
            .unwrap_or_else(|_| "your-api-key-here".to_string()),
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        streaming: false,
        temperature: Some(0.0),
        max_tokens: Some(1000),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        organization: None,
    };
    
    let llm = OpenAIChat::new(config);
    
    // 2. 创建工具集
    println!("--- 可用工具集 ---\n");
    
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(Calculator::new()),        // 基础计算
        Arc::new(SimpleMathTool::new()),    // 高级数学
        Arc::new(DateTimeTool::new()),      // 日期时间
        Arc::new(URLFetchTool::new()),      // 网页抓取
    ];
    
    for tool in &tools {
        println!("工具: {}", tool.name());
        let desc = tool.description();
        for line in desc.lines().take(3) {
            println!("  {}", line);
        }
        println!();
    }
    
    // 3. 创建 Agent
    let agent: Arc<dyn BaseAgent> = Arc::new(ReActAgent::new(llm, tools.clone(), None));
    
    let executor = AgentExecutor::new(agent, tools)
        .with_verbose(true)
        .with_max_iterations(10);
    
    // 4. 执行复杂任务
    println!("--- 执行复杂任务 ---\n");
    
    let tasks = vec![
        // 需要数学计算
        ("计算任务", "计算 123 乘以 456，然后求结果的平方根"),
        
        // 需要日期时间
        ("时间任务", "告诉我今天是星期几，以及当前日期"),
        
        // 需要多步数学
        ("多步数学", "计算 5 的阶乘，然后计算这个结果除以 6"),
        
        // 需要工具组合
        ("组合任务", "告诉我今天的日期，然后计算 100 天后是哪一天"),
    ];
    
    for (task_name, task) in tasks {
        println!("{}\n任务: {}\n", "=".repeat(50), task);
        
        match executor.invoke(task.to_string()).await {
            Ok(answer) => {
                println!("\n最终答案: {}\n", answer);
            }
            Err(e) => {
                eprintln!("错误: {}\n", e);
            }
        }
    }
    
    println!("=== 示例完成 ===");
    Ok(())
}