//! What: a parent agent delegates a sub-task to a fresh-context child agent.
//! Run: cargo run -p langchainrust --example agent_subagent
//! Env: OPENAI_API_KEY (optional: OPENAI_MODEL)

use std::sync::Arc;

use langchainrust::agents::{
    AgentExecutor, FunctionCallingAgent, SubagentConfig, SubagentFactory, SubagentTool,
};
use langchainrust::core::tools::BaseTool;
use langchainrust::tools::Calculator;
use langchainrust::{OpenAIChat, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = OpenAIChat::new(OpenAIConfig::from_env_result()?);

    // The factory builds a fresh child executor per call: empty history,
    // its own tool set and iteration cap.
    let child_llm = llm.clone();
    let factory: SubagentFactory = Arc::new(move |_depth| {
        let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
        let agent = FunctionCallingAgent::new(
            child_llm.clone(),
            tools.clone(),
            Some("You are a careful calculator. Use the tool.".to_string()),
        );
        Ok(Arc::new(AgentExecutor::new(Arc::new(agent), tools).with_max_iterations(5)))
    });

    let subagent = SubagentTool::new(
        SubagentConfig::new("calculator", "Delegate arithmetic to a child agent")
            .with_max_depth(1),
        factory,
    );

    let parent_tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(subagent)];
    let parent = FunctionCallingAgent::new(
        llm,
        parent_tools.clone(),
        Some("You are a manager. Delegate arithmetic to the `calculator` subagent.".to_string()),
    );
    let executor = AgentExecutor::new(Arc::new(parent), parent_tools).with_max_iterations(5);

    let answer = executor.invoke("What is 25 * 37?".to_string()).await?;
    println!("{answer}");
    Ok(())
}
