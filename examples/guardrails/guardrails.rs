//! Guardrails 示例
//!
//! 展示如何使用 GuardedAgent 保护 LLM 输出安全。
//!
//! # 运行
//! ```bash
//! cargo run --example guardrails
//! ```

// Guardrails 示例仅展示用法说明，无需导入 LLM 类型

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Guardrails 示例 ===\n");

    // Guardrails 在 LLM 输出返回给用户前进行检查和过滤
    println!("Guardrails 类型:");
    println!("1. SensitiveInfoGuardrail - 检测和过滤敏感信息(手机号/身份证/邮箱)");
    println!("2. ForbiddenWordsGuardrail - 禁止特定词汇");
    println!("3. MaxLengthGuardrail - 限制输出长度");

    println!("\nGuardedAgent 使用方式:");
    println!("  let guardrail = SensitiveInfoGuardrail::new();");
    println!("  let agent = GuardedAgent::new(base_agent)");
    println!("    .with_output_guardrail(guardrail);");
    println!("  let result = agent.invoke(input).await?;");

    println!("\n工作流程:");
    println!("  用户输入 → Agent 处理 → Guardrail 检查 → 返回安全输出");
    println!("                              ↓");
    println!("                     检测到敏感信息 → 过滤/替换/拒绝");

    println!("\n提示: 需要设置 OPENAI_API_KEY 才能进行真实调用。");
    Ok(())
}
