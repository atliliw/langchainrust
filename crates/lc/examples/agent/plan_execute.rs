//! Plan-Execute Agent 示例
//!
//! 展示如何使用 PlanExecuteAgent 进行规划-执行-重规划循环。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_plan_execute
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Plan-Execute Agent 示例 ===\n");

    // Plan-Execute Agent 工作流程:
    println!("Plan-Execute Agent 流程:");
    println!("1. 规划: LLM 分析任务,生成执行计划");
    println!("2. 执行: 按计划逐步执行,调用工具");
    println!("3. 重规划: 根据执行结果,可能调整计划");
    println!("4. 完成: 所有步骤执行完毕,返回最终结果");

    println!("\n适用场景:");
    println!("- 复杂的多步骤任务");
    println!("- 需要根据中间结果调整策略的任务");
    println!("- 需要使用多种工具的复合任务");

    println!("\n使用方式:");
    println!("  let agent = PlanExecuteAgent::new(llm)");
    println!("    .with_tool(Calculator::new())");
    println!("    .with_tool(SimpleMathTool::new());");
    println!("  let result = agent.run(\"计算 (15 + 27) * 3\").await?;");

    println!("\n提示: 需要设置 OPENAI_API_KEY 才能进行真实调用。");
    Ok(())
}
