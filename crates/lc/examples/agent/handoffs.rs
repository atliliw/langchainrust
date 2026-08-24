//! Handoffs 示例
//!
//! 展示如何使用 HandoffManager 在多个 Agent 之间切换。
//!
//! # 运行
//! ```bash
//! cargo run --example agent_handoffs
//! ```

// Handoffs 示例仅展示用法说明，无需导入 LLM 类型

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Handoffs 示例 ===\n");

    // Handoffs 允许在多个专业 Agent 之间切换
    println!("Handoffs 工作流程:");
    println!("1. 主 Agent 接收用户请求");
    println!("2. 判断需要哪种专业技能");
    println!("3. 将对话交接给专业 Agent");
    println!("4. 专业 Agent 处理完成后可交接回主 Agent");

    println!("\n示例场景:");
    println!("- 客服 Agent → 技术支持 Agent → 回到客服");
    println!("- 通用助手 → 代码专家 → 文档专家");
    println!("- 销售顾问 → 产品专家 → 售后服务");

    println!("\nHandoffManager 使用方式:");
    println!("  let manager = HandoffManager::new();");
    println!("  manager.register_agent(\"tech\", tech_executor)?;");
    println!("  manager.register_agent(\"docs\", docs_executor)?;");
    println!("  manager.set_primary(\"tech\")?;");
    println!("  let handoff_tools = manager.handoff_tools();");

    println!("\n提示: 需要设置 OPENAI_API_KEY 才能进行真实调用。");
    Ok(())
}
