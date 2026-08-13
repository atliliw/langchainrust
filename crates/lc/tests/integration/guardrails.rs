//! Guardrails 集成测试
//!
//! - mock 执行单元(P2-6):固定返回含敏感信息/正常信息的输出,断言确实被 `Blocked`/
//!   放行,不依赖 API Key、不触网,默认运行;
//! - 真实 Agent 端到端(输入验证 + 执行 + 输出验证,需 API Key,`#[ignore]`):
//! ```bash
//! cargo test --test integration_guardrails -- --ignored
//! ```

#[path = "../common/mod.rs"]
mod common;

use async_trait::async_trait;
use futures_util::Stream;
use langchainrust::guardrails::GuardrailError;
use langchainrust::guardrails::{
    Guardable, GuardableChunk, GuardedAgent, GuardrailsConfig, MaxLengthGuardrail,
    SensitiveInfoGuardrail,
};
use langchainrust::tools::Calculator;
use langchainrust::{AgentExecutor, BaseAgent, BaseTool, FunctionCallingAgent};
use std::pin::Pin;
use std::sync::Arc;

/// `GuardedAgent` 包装的执行单元错误类型(`Guardable` trait 签名所需)。
type DynError = Box<dyn std::error::Error + Send + Sync>;

/// 固定返回含敏感信息输出的 mock 执行单元(P2-6):输出含邮箱(低误报具体模式),
/// 必然被 `SensitiveInfoGuardrail` 拦截。
struct LeakyOutputAgent;

#[async_trait]
impl Guardable for LeakyOutputAgent {
    async fn invoke_str(&self, _input: &str) -> Result<String, DynError> {
        Ok("用户邮箱是 user@example.com,请勿外传".to_string())
    }

    async fn stream_str(
        &self,
        _input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        Err("stream not supported".into())
    }
}

/// 固定返回正常输出的 mock 执行单元(P2-6 对照组):不触发任何护栏。
struct BenignOutputAgent;

#[async_trait]
impl Guardable for BenignOutputAgent {
    async fn invoke_str(&self, _input: &str) -> Result<String, DynError> {
        Ok("今天是周三".to_string())
    }

    async fn stream_str(
        &self,
        _input: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<GuardableChunk, DynError>> + Send>>, DynError>
    {
        Err("stream not supported".into())
    }
}

#[tokio::test]
#[ignore = "需要 API Key"]
async fn test_guarded_agent_normal_invoke() {
    let llm = common::TestConfig::get().openai_chat();
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

/// P2-6:mock 固定返回含敏感信息的输出,断言确实被 `Blocked` 且记录违规。
/// 替代原先"两种都接受,只验证不 panic"的无断言测试。
#[tokio::test]
async fn test_guarded_agent_blocks_sensitive_output() {
    let config = GuardrailsConfig::new().with_output(Arc::new(SensitiveInfoGuardrail::new()));
    let mut guarded = GuardedAgent::new(Arc::new(LeakyOutputAgent), config);

    let result = guarded.invoke("anything".to_string()).await;
    match result {
        Err(GuardrailError::Blocked { reason, .. }) => {
            assert!(
                reason.contains("敏感"),
                "拦截原因应说明敏感, 实际: {reason}"
            );
        }
        other => panic!("含敏感信息的输出应被 Blocked, 实际: {other:?}"),
    }
    assert!(!guarded.violations().is_empty(), "拦截应记录违规");
}

/// P2-6 对照组:正常输出不被拦截,也不记录违规。
#[tokio::test]
async fn test_guarded_agent_passes_benign_output() {
    let config = GuardrailsConfig::new().with_output(Arc::new(SensitiveInfoGuardrail::new()));
    let mut guarded = GuardedAgent::new(Arc::new(BenignOutputAgent), config);

    let result = guarded.invoke("anything".to_string()).await;
    assert_eq!(result.unwrap(), "今天是周三");
    assert!(guarded.violations().is_empty(), "正常输出不应记录违规");
}
