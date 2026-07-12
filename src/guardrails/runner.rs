//! Guardrail 执行器

use super::guardrail::{GuardrailError, GuardrailResult, GuardrailsConfig};

/// 违规记录
#[derive(Debug, Clone)]
pub struct GuardrailViolation {
    pub guardrail_name: String,
    pub stage: String,
    pub reason: String,
}

/// Guardrail 执行器:按配置依次执行 input/output guardrails
pub struct GuardrailRunner {
    config: GuardrailsConfig,
    violations: Vec<GuardrailViolation>,
}

impl GuardrailRunner {
    pub fn new(config: GuardrailsConfig) -> Self {
        Self {
            config,
            violations: Vec::new(),
        }
    }

    /// 验证输入
    pub async fn validate_input(&mut self, input: &str) -> Result<(), GuardrailError> {
        let mut first_block: Option<String> = None;
        for g in &self.config.input_guardrails {
            if let GuardrailResult::Block { reason } = g.validate(input).await {
                self.violations.push(GuardrailViolation {
                    guardrail_name: g.name().to_string(),
                    stage: "input".to_string(),
                    reason: reason.clone(),
                });
                if first_block.is_none() {
                    first_block = Some(reason);
                }
                if self.config.fail_fast {
                    return Err(GuardrailError::Blocked(first_block.unwrap()));
                }
            }
        }
        if let Some(r) = first_block {
            return Err(GuardrailError::Blocked(r));
        }
        Ok(())
    }

    /// 验证输出(支持 Modify)
    pub async fn validate_output(&mut self, output: &str) -> Result<String, GuardrailError> {
        let mut current = output.to_string();
        let mut first_block: Option<String> = None;
        for g in &self.config.output_guardrails {
            match g.validate(&current).await {
                GuardrailResult::Pass => {}
                GuardrailResult::Block { reason } => {
                    self.violations.push(GuardrailViolation {
                        guardrail_name: g.name().to_string(),
                        stage: "output".to_string(),
                        reason: reason.clone(),
                    });
                    if first_block.is_none() {
                        first_block = Some(reason);
                    }
                    if self.config.fail_fast {
                        return Err(GuardrailError::Blocked(first_block.unwrap()));
                    }
                }
                GuardrailResult::Modify { new_value } => {
                    current = new_value;
                }
            }
        }
        if let Some(r) = first_block {
            return Err(GuardrailError::Blocked(r));
        }
        Ok(current)
    }

    /// 获取违规记录
    pub fn violations(&self) -> &[GuardrailViolation] {
        &self.violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrails::guardrail::InputGuardrail;
    use async_trait::async_trait;
    use std::sync::Arc;

    struct AlwaysBlock;
    #[async_trait]
    impl InputGuardrail for AlwaysBlock {
        fn name(&self) -> &str {
            "AlwaysBlock"
        }
        async fn validate(&self, _input: &str) -> GuardrailResult {
            GuardrailResult::Block {
                reason: "always".to_string(),
            }
        }
    }

    struct AlwaysPass;
    #[async_trait]
    impl InputGuardrail for AlwaysPass {
        fn name(&self) -> &str {
            "AlwaysPass"
        }
        async fn validate(&self, _input: &str) -> GuardrailResult {
            GuardrailResult::Pass
        }
    }

    #[tokio::test]
    async fn test_runner_pass() {
        let config = GuardrailsConfig::new().with_input(Arc::new(AlwaysPass));
        let mut runner = GuardrailRunner::new(config);
        assert!(runner.validate_input("hi").await.is_ok());
    }

    #[tokio::test]
    async fn test_runner_block() {
        let config = GuardrailsConfig::new().with_input(Arc::new(AlwaysBlock));
        let mut runner = GuardrailRunner::new(config);
        assert!(runner.validate_input("hi").await.is_err());
        assert_eq!(runner.violations().len(), 1);
    }

    #[tokio::test]
    async fn test_runner_fail_fast_collects_one() {
        // fail_fast=true:第一个 block 即返回,只记录 1 个
        let config = GuardrailsConfig::new()
            .with_input(Arc::new(AlwaysBlock))
            .with_input(Arc::new(AlwaysBlock))
            .fail_fast(true);
        let mut runner = GuardrailRunner::new(config);
        let _ = runner.validate_input("hi").await;
        assert_eq!(runner.violations().len(), 1);
    }

    #[tokio::test]
    async fn test_runner_no_fail_fast_collects_all() {
        // fail_fast=false:检查所有,记录 2 个
        let config = GuardrailsConfig::new()
            .with_input(Arc::new(AlwaysBlock))
            .with_input(Arc::new(AlwaysBlock))
            .fail_fast(false);
        let mut runner = GuardrailRunner::new(config);
        let _ = runner.validate_input("hi").await;
        assert_eq!(runner.violations().len(), 2);
    }
}
