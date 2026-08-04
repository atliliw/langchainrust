//! Guardrail trait、结果类型、配置、错误

use async_trait::async_trait;
use std::sync::Arc;

/// Guardrail 验证结果
#[derive(Debug, Clone)]
pub enum GuardrailResult {
    /// 通过
    Pass,
    /// 拦截
    Block { reason: String },
    /// 修改后通过(仅输出侧)
    Modify { new_value: String },
}

impl GuardrailResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, GuardrailResult::Pass)
    }
    pub fn is_block(&self) -> bool {
        matches!(self, GuardrailResult::Block { .. })
    }
}

/// 输入 Guardrail trait
#[async_trait]
pub trait InputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    async fn validate(&self, input: &str) -> GuardrailResult;
}

/// 输出 Guardrail trait
#[async_trait]
pub trait OutputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    async fn validate(&self, output: &str) -> GuardrailResult;
}

/// Guardrails 配置
pub struct GuardrailsConfig {
    pub input_guardrails: Vec<Arc<dyn InputGuardrail>>,
    pub output_guardrails: Vec<Arc<dyn OutputGuardrail>>,
    pub fail_fast: bool,
}

impl Default for GuardrailsConfig {
    fn default() -> Self {
        Self {
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            fail_fast: true,
        }
    }
}

impl GuardrailsConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input(mut self, g: Arc<dyn InputGuardrail>) -> Self {
        self.input_guardrails.push(g);
        self
    }

    pub fn with_output(mut self, g: Arc<dyn OutputGuardrail>) -> Self {
        self.output_guardrails.push(g);
        self
    }

    pub fn fail_fast(mut self, v: bool) -> Self {
        self.fail_fast = v;
        self
    }
}

/// Guardrail 错误
#[derive(Debug)]
pub enum GuardrailError {
    /// 被 Guardrail 拦截
    Blocked(String),
    /// Agent 执行错误
    AgentError(String),
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardrailError::Blocked(msg) => write!(f, "Guardrail 拦截: {}", msg),
            GuardrailError::AgentError(msg) => write!(f, "Agent 执行错误: {}", msg),
        }
    }
}

impl std::error::Error for GuardrailError {}
