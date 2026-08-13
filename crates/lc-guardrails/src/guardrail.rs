//! Guardrail trait、结果类型、配置、错误

use async_trait::async_trait;
use std::sync::Arc;

/// 输入 Guardrail 验证结果
///
/// 输入侧不允许 `Modify`:输入护栏要么放行、要么拦截。
/// 与 [`OutputGuardrailResult`] 分离后,类型系统强制"Modify 仅输出侧",
/// 输入护栏在编译期就无法返回改写结果。
#[derive(Debug, Clone)]
pub enum InputGuardrailResult {
    /// 通过
    Pass,
    /// 拦截
    Block { reason: String },
}

impl InputGuardrailResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, InputGuardrailResult::Pass)
    }
    pub fn is_block(&self) -> bool {
        matches!(self, InputGuardrailResult::Block { .. })
    }
}

/// 输出 Guardrail 验证结果
///
/// `Modify` 是输出侧专属:输出护栏可以改写结果后放行。
#[derive(Debug, Clone)]
pub enum OutputGuardrailResult {
    /// 通过
    Pass,
    /// 拦截
    Block { reason: String },
    /// 修改后通过(仅输出侧)
    Modify { new_value: String },
}

impl OutputGuardrailResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, OutputGuardrailResult::Pass)
    }
    pub fn is_block(&self) -> bool {
        matches!(self, OutputGuardrailResult::Block { .. })
    }
    pub fn is_modify(&self) -> bool {
        matches!(self, OutputGuardrailResult::Modify { .. })
    }
}

/// 输入 Guardrail trait
///
/// 返回 [`InputGuardrailResult`](没有 `Modify` 变体),输入侧天然无法改写。
#[async_trait]
pub trait InputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    async fn validate(&self, input: &str) -> InputGuardrailResult;
}

/// 输出 Guardrail trait
///
/// 返回 [`OutputGuardrailResult`](含 `Modify` 变体),是改写的唯一合法入口。
#[async_trait]
pub trait OutputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    async fn validate(&self, output: &str) -> OutputGuardrailResult;
}

/// 流式块动作
///
/// 流式护栏对单个 chunk 的处置结果(P1-4)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkAction {
    /// 放行
    Pass,
    /// 改写后放行
    Replace(String),
    /// 拦截丢弃
    Block,
}

/// 流式输出护栏 trait(P1-4)
///
/// 第一阶段:对每个增量 chunk 做快速检查,在敏感信息展示给用户之前拦截。
/// 调用方维护滑动窗口(`tail + chunk`)以避免跨块切断关键词(如 `"passwo" + "rd"`)。
/// 完整输出后的二次复查由 [`OutputGuardrail`] 承担(`GuardrailRunner::validate_output`)。
#[async_trait]
pub trait StreamingOutputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    /// 增量检查一个 chunk(可能是 `tail + chunk` 的组合串)。
    async fn validate_chunk(&self, chunk: &str) -> ChunkAction;
}

/// Guardrails 配置
#[derive(Clone)]
pub struct GuardrailsConfig {
    pub input_guardrails: Vec<Arc<dyn InputGuardrail>>,
    pub output_guardrails: Vec<Arc<dyn OutputGuardrail>>,
    /// 流式护栏:流式输出时逐块增量检查(P1-4)。
    pub streaming_guardrails: Vec<Arc<dyn StreamingOutputGuardrail>>,
    /// 审计持久化 sink(P1-7)。
    pub audit_sink: Option<Arc<dyn crate::audit::AuditSink>>,
    pub fail_fast: bool,
}

impl Default for GuardrailsConfig {
    fn default() -> Self {
        Self {
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            streaming_guardrails: Vec::new(),
            audit_sink: None,
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

    /// 添加一个流式护栏(两阶段流式检查的第一阶段)。
    pub fn with_streaming(mut self, g: Arc<dyn StreamingOutputGuardrail>) -> Self {
        self.streaming_guardrails.push(g);
        self
    }

    /// 配置审计持久化 sink:每次违规记录时同步写入(P1-7)。
    pub fn with_audit_sink(mut self, sink: Arc<dyn crate::audit::AuditSink>) -> Self {
        self.audit_sink = Some(sink);
        self
    }

    pub fn fail_fast(mut self, v: bool) -> Self {
        self.fail_fast = v;
        self
    }
}

/// Guardrail 错误
///
/// `Blocked` 携带拦截原因 + 已处理部分 + 面向用户的建议(P1-1/P1-6),
/// 让上层能给用户"被拦截"而非"系统错误"的反馈。
#[derive(Debug)]
pub enum GuardrailError {
    /// 被 Guardrail 拦截
    Blocked {
        /// 拦截原因(护栏侧说明)
        reason: String,
        /// 拦截前已处理的部分内容(供上层展示局部结果 / 决定是否重生成)
        partial: Option<String>,
        /// 面向用户的建议(如何重述输入 / 修正输出)
        suggestion: Option<String>,
    },
    /// Agent 执行错误
    AgentError(String),
}

impl GuardrailError {
    /// 从 `OutputValidation::Blocked` 构造带用户建议的 `GuardrailError`。
    pub(crate) fn from_blocked(reason: String, partial: String, suggestion: String) -> Self {
        GuardrailError::Blocked {
            reason,
            partial: Some(partial),
            suggestion: Some(suggestion),
        }
    }
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardrailError::Blocked {
                reason,
                partial,
                suggestion,
            } => {
                write!(f, "Guardrail 拦截: {}", reason)?;
                if let Some(p) = partial {
                    write!(f, " (已处理部分: {})", p)?;
                }
                if let Some(s) = suggestion {
                    write!(f, " 建议: {}", s)?;
                }
                Ok(())
            }
            GuardrailError::AgentError(msg) => write!(f, "Agent 执行错误: {}", msg),
        }
    }
}

impl std::error::Error for GuardrailError {}
