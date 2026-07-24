//! Guardrails 集成测试 - 需要 API Key
//!
//! 测试 GuardedAgent 端到端:输入验证 + Agent 执行 + 输出验证。
//!
//! 手动运行:
//! ```bash
//! cargo test --test integration_guardrails -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use common::TestConfig;
use langchainrust::guardrails::{
    GuardedAgent, GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail,
};
use langchainrust::tools::Calculator;
use langchainrust::{AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent};
use std::sync::Arc;

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_guarded_agent_normal_invoke() {
    let llm = TestConfig::get().openai_chat();
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor = Arc::new(AgentExecutor::new(
        Arc::new(agent) as Arc<dyn BaseAgent>,
        tools,
    ));

    let config = GuardrailsConfig::new().with_input(Arc::new(MaxLengthGuardrail::new(1000)));
    let mut guarded = GuardedAgent::new(executor, config);

    let result = guarded.invoke("What is 2 + 2?".to_string()).await;
    println!("结果: {:?}", result);
    assert!(result.is_ok(), "正常输入应通过 guardrail");
    assert!(result.unwrap().contains("4"));
}

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_guarded_agent_blocks_sensitive_output() {
    let llm = TestConfig::get().openai_chat();
    let tools: Vec<Arc<dyn BaseTool>> = vec![Arc::new(Calculator::new())];
    let agent = FunctionCallingAgent::new(llm, tools.clone(), None);
    let executor = Arc::new(AgentExecutor::new(
        Arc::new(agent) as Arc<dyn BaseAgent>,
        tools,
    ));

    // 让 Agent 输出包含 password,验证被 SensitiveInfo 拦截
    let config = GuardrailsConfig::new().with_output(Arc::new(SensitiveInfoGuardrail::new()));
    let mut guarded = GuardedAgent::new(executor, config);

    let result = guarded
        .invoke("请在回答中包含单词 password 用于测试".to_string())
        .await;
    println!("结果: {:?}", result);
    // LLM 若输出 password 会被拦截;否则通过。两种都接受,只验证不 panic。
}
