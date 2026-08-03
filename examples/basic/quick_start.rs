//! Quick Start — 3 行代码创建 Agent
//!
//! 演示 langchainrust v0.7.2 的新 API：
//! - `LLMClient::from_env()` — 零配置自动检测 Provider
//! - `AgentBuilder` — 流畅 Builder 创建 Agent
//! - `FunctionCallingAgent` — 现在支持任何 LLM Provider
//!
//! # 运行方式
//!
//! ```bash
//! # 自动检测（设置任意一个环境变量）
//! export OPENAI_API_KEY="sk-..."
//! cargo run --example quick_start
//!
//! # 或显式指定 Provider
//! export ANTHROPIC_API_KEY="sk-..."
//! cargo run --example quick_start
//! ```

use langchainrust::{AgentBuilder, AgentExecutor, BaseAgent, LLMClient};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================
    // 方式 1: LLMClient::from_env() — 自动检测
    // ========================================
    println!("=== 方式 1: LLMClient::from_env() ===");

    match LLMClient::from_env() {
        Ok(llm) => {
            println!("✓ 检测到 LLM Provider: {}", llm.model_name());

            // 用 AgentBuilder 创建 Agent — 3 行代码
            let agent = AgentBuilder::new()
                .llm_from_arc(llm.into_inner())
                .system("You are a helpful assistant. Answer concisely.")
                .build()?;

            let executor = AgentExecutor::new(
                Arc::new(agent) as Arc<dyn BaseAgent>,
                vec![],
            );

            let result = executor.invoke("What is 2+2?".into()).await?;
            println!("回答: {}", result);
        }
        Err(e) => {
            println!("✗ 未检测到 LLM Provider: {}", e);
            println!("  请设置以下环境变量之一:");
            println!("  - OPENAI_API_KEY");
            println!("  - ANTHROPIC_API_KEY");
            println!("  - OLLAMA_BASE_URL");
        }
    }

    // ========================================
    // 方式 2: 显式指定 Provider
    // ========================================
    println!("\n=== 方式 2: 显式指定 Provider ===");

    // 用 OpenAI
    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if !openai_key.is_empty() {
        use langchainrust::OpenAIConfig;

        let config = OpenAIConfig::new(&openai_key).with_model("gpt-4o-mini");
        let llm = LLMClient::openai(config);

        let agent = AgentBuilder::new()
            .llm_from_arc(llm.into_inner())
            .system("You are a math tutor.")
            .build()?;

        println!("✓ OpenAI Agent 创建成功: {}", agent.system_prompt().unwrap_or("none"));
    }

    // ========================================
    // 方式 3: 从环境读配置，再覆盖参数
    // ========================================
    println!("\n=== 方式 3: from_env_result() + 覆盖参数 ===");

    if !openai_key.is_empty() {
        use langchainrust::OpenAIConfig;

        let config = OpenAIConfig::from_env_result()?.with_model("gpt-4o-mini");
        let llm = LLMClient::openai(config);

        println!("✓ 模型: {}", llm.model_name());
    }

    println!("\n=== Done ===");
    Ok(())
}
