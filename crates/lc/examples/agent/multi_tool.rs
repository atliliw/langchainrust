//! 多工具 Agent 示例
//!
//! 展示 Agent 携带多个工具(Calculator / DateTime / Math),自动选择调用。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_multi_tool
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)

use langchainrust::tools::{Calculator, DateTimeTool, SimpleMathTool};
use langchainrust::{
    AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent, OpenAIChat, OpenAIConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").expect("请设置 OPENAI_API_KEY 环境变量");
    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let llm = OpenAIChat::new(OpenAIConfig {
        api_key,
        base_url,
        model: "gpt-4o-mini".to_string(),
        ..Default::default()
    });

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
        .await?;
    println!("结果: {}", result);
    Ok(())
}
