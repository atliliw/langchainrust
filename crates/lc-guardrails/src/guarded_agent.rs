//! GuardedAgent - 带 Guardrails 的 Agent 包装器

use std::sync::Arc;

use lc_agents::AgentExecutor;

use super::guardrail::{GuardrailError, GuardrailsConfig};
use super::runner::{GuardrailRunner, GuardrailViolation};

/// 带 Guardrails 的 Agent 包装器
///
/// `invoke` 时:验证输入 -> 执行 Agent -> 验证输出。
pub struct GuardedAgent {
    executor: Arc<AgentExecutor>,
    runner: GuardrailRunner,
}

impl GuardedAgent {
    pub fn new(executor: Arc<AgentExecutor>, config: GuardrailsConfig) -> Self {
        Self {
            executor,
            runner: GuardrailRunner::new(config),
        }
    }

    /// 执行:验输入 -> Agent -> 验输出
    pub async fn invoke(&mut self, input: String) -> Result<String, GuardrailError> {
        self.runner.validate_input(&input).await?;
        let output = self
            .executor
            .invoke(input)
            .await
            .map_err(|e| GuardrailError::AgentError(e.to_string()))?;
        self.runner.validate_output(&output).await
    }

    /// 获取违规记录
    pub fn violations(&self) -> &[GuardrailViolation] {
        self.runner.violations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validators::MaxLengthGuardrail;
    use lc_agents::{BaseAgent, FunctionCallingAgent};
    use lc_providers::{OpenAIChat, OpenAIConfig};

    fn guarded_with_maxlen(max: usize) -> GuardedAgent {
        let llm = OpenAIChat::new(OpenAIConfig::default());
        let agent = FunctionCallingAgent::new(llm, vec![], None);
        let executor = Arc::new(AgentExecutor::new(
            Arc::new(agent) as Arc<dyn BaseAgent>,
            vec![],
        ));
        let config =
            GuardrailsConfig::new().with_input(Arc::new(MaxLengthGuardrail::new(max)) as Arc<_>);
        GuardedAgent::new(executor, config)
    }

    #[tokio::test]
    async fn test_blocks_long_input_before_agent() {
        // 输入超过 3 字符,被 MaxLength 拦截,不调用 Agent(不触网)
        let mut g = guarded_with_maxlen(3);
        let result = g.invoke("this is too long input".to_string()).await;
        assert!(result.is_err());
        assert_eq!(g.violations().len(), 1);
        // 确认是 Blocked 而非 AgentError
        match result.unwrap_err() {
            GuardrailError::Blocked(_) => {}
            other => panic!("应为 Blocked, 实际: {:?}", other),
        }
    }
}
