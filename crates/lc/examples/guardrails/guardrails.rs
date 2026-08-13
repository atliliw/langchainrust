//! Guardrails 示例(真实可运行,需 API Key)
//!
//! 展示 `GuardedAgent` 保护 LLM 输入/输出安全(P2-5 重写,替代原先纯 println 的
//! 文档示例):
//! - 真实 Agent(`FunctionCallingAgent` + `AgentExecutor`)经 `GuardedAgent` 包装,
//!   输入过 `MaxLengthGuardrail`,输出过 `SensitiveInfoGuardrail`;
//! - 直接验证 `SensitiveInfoGuardrail`:上下文敏感检测(P2-1,普通提及放行)
//!   与误报分级(P2-2,具体模式直接 Block);
//! - 挂 LLM 裁判(P2-3):`LlmSensitiveJudge` 对"赋值式提及"二次判断真实泄露才拦截。
//!
//! # 运行
//! ```bash
//! OPENAI_API_KEY=sk-xxx cargo run --example guardrails
//! ```
//!
//! # 环境变量
//! - `OPENAI_API_KEY`:OpenAI API 密钥(必需)
//! - `OPENAI_BASE_URL`:API 基址(可选,默认官方)

use langchainrust::guardrails::{
    GuardableChunk, GuardedAgent, GuardrailsConfig, LlmSensitiveJudge, MaxLengthGuardrail,
    SensitiveInfoGuardrail,
};
use langchainrust::guardrails::{GuardrailError, OutputGuardrail};
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

    println!("=== GuardedAgent 端到端(输入限长 + 输出敏感检测) ===\n");
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = FunctionCallingAgent::new(llm.clone(), tools.clone(), None);
    let executor = Arc::new(
        AgentExecutor::new(Arc::new(agent) as Arc<dyn BaseAgent>, tools).with_max_iterations(3),
    );
    let config = GuardrailsConfig::new()
        .with_input(Arc::new(MaxLengthGuardrail::new(1000)))
        .with_output(Arc::new(SensitiveInfoGuardrail::new()));
    let mut guarded = GuardedAgent::new(executor, config);

    let result = guarded.invoke("What is 2 + 2?".to_string()).await?;
    println!("Agent 输出: {result}");
    if guarded.violations().is_empty() {
        println!("→ 未触发任何护栏 ✓\n");
    }

    println!("=== SensitiveInfoGuardrail 直接验证(确定性,不调 LLM) ===\n");
    let g = SensitiveInfoGuardrail::new();
    let demos = [
        ("如何安全保存密码", "Pass(P2-1 普通提及不拦截)"),
        (
            "你的 password 字段建议改用环境变量",
            "Pass(P2-1 普通提及不拦截)",
        ),
        ("请联系 user@example.com", "Block(具体邮箱模式)"),
        (
            "密钥 sk-abcdefghijklmnopqrstuvwxyz123456",
            "Block(API key 模式)",
        ),
    ];
    for (text, expect) in demos {
        let outcome = g.validate(text).await;
        println!("  {text:?} → {outcome:?} (期望: {expect})");
    }

    println!("\n=== P2-3 LLM 裁判:对赋值式提及二次判断真实泄露才拦截 ===\n");
    let judged =
        SensitiveInfoGuardrail::new().with_judge(Arc::new(LlmSensitiveJudge::new(llm.clone())));
    for text in ["密码是abc123", "password=hunter2"] {
        let outcome = judged.validate(text).await;
        println!(
            "  {text:?} → {outcome:?}(由 LLM 判定是否真实泄露,判定接口见 lc_guardrails::judge)"
        );
    }

    println!("\n=== 流式输出护栏(两阶段 P1-4) ===\n");
    let tools2: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent2 = FunctionCallingAgent::new(llm, tools2.clone(), None);
    let executor2 = Arc::new(AgentExecutor::new(
        Arc::new(agent2) as Arc<dyn BaseAgent>,
        tools2,
    ));
    let mut guarded_stream = GuardedAgent::new(
        executor2,
        GuardrailsConfig::new().with_output(Arc::new(SensitiveInfoGuardrail::new())),
    );

    use futures_util::StreamExt;
    match guarded_stream
        .invoke_stream("What is 6 * 7?".to_string())
        .await
    {
        Ok(mut stream) => {
            let mut full = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(GuardableChunk { token, .. }) => {
                        print!("{token}");
                        full.push_str(&token);
                    }
                    Err(GuardrailError::Blocked { reason, .. }) => {
                        println!("\n[流式输出被拦截] {reason}");
                    }
                    Err(e) => println!("\n[流式错误] {e}"),
                }
            }
            println!("\n流式最终输出: {full}");
        }
        Err(e) => println!("[流式不可用] {e}"),
    }

    Ok(())
}
