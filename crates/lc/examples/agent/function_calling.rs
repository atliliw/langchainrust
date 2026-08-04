//! Function Calling Agent 示例
//!
//! 展示 FunctionCallingAgent 通过原生 Function Calling 调用工具。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_function_calling
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)

use langchainrust::tools::Calculator;
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

    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor = AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools)
        .with_max_iterations(3)
        .with_verbose(true);

    let result = executor.invoke("What is 25 + 17?".to_string()).await?;
    println!("结果: {}", result);
    Ok(())
}
