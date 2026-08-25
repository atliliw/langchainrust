//! Plan-Execute Agent example
//!
//! Shows how to use PlanExecuteAgent for a plan-execute-replan loop.
//!
//! # Run
//! ```bash
//! cargo run --example agent_plan_execute
//! ```
//!
//! # Environment variables
//! - `OPENAI_API_KEY`: OpenAI API key (required)

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Plan-Execute Agent example ===\n");

    // Plan-Execute Agent workflow:
    println!("Plan-Execute Agent flow:");
    println!("1. Plan: the LLM analyzes the task and generates an execution plan");
    println!("2. Execute: each step runs in order, calling tools as needed");
    println!("3. Replan: based on the results, the plan may be adjusted");
    println!("4. Done: when all steps are complete, return the final result");

    println!("\nUse cases:");
    println!("- Complex multi-step tasks");
    println!("- Tasks that need to adjust the strategy based on intermediate results");
    println!("- Composite tasks that need multiple tools");

    println!("\nUsage:");
    println!("  let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];");
    println!("  let agent = PlanExecuteAgent::new(llm, tools);");
    println!("  let result = agent.run(\"Compute (15 + 27) * 3\").await?;");

    println!("\nNote: set the OPENAI_API_KEY environment variable to make real calls.");
    Ok(())
}
