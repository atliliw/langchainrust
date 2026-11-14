//! 5-minute quickstart: a function-calling agent with one tool.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! export OPENAI_MODEL=gpt-4o-mini   # optional
//! cargo run -p langchainrust --example quickstart_agent
//! ```

use std::sync::Arc;

use langchainrust::agents::{AgentExecutor, FunctionCallingAgent};
use langchainrust::core::tools::BaseTool;
use langchainrust::tools::Calculator;
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];

    let agent = FunctionCallingAgent::new(
        llm,
        tools.clone(),
        Some("You are a concise assistant. Use tools when needed.".to_string()),
    );

    let executor = AgentExecutor::new(Arc::new(agent), tools).with_max_iterations(5);
    let answer = executor.invoke("What is 25 * 37?".to_string()).await?;
    println!("{answer}");
    Ok(())
}
