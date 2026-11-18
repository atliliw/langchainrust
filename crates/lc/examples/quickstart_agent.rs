//! 5-minute quickstart: a function-calling agent with one tool.
//!
//! ```bash
//! export OPENAI_API_KEY=sk-...
//! export OPENAI_MODEL=gpt-4o-mini   # optional
//! cargo run -p langchainrust --example quickstart_agent
//! ```

use langchainrust::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = AgentBuilder::new()
        .llm(OpenAIChat::new(OpenAIConfig::from_env_result()?))
        .system("You are a concise assistant. Use tools when needed.")
        .tool(Calculator::new())
        .max_iterations(5)
        .build_executor()?;

    let answer = executor.invoke("What is 25 * 37?".to_string()).await?;
    println!("{answer}");
    Ok(())
}
