//! Guardrail 执行器

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::guardrail::{
    ChunkAction, GuardrailError, GuardrailsConfig, InputGuardrailResult, OutputGuardrailResult,
};

/// 违规记录上限:超过后丢弃最旧记录,防止内存无界增长(P1-2)。
const MAX_VIOLATIONS: usize = 1000;

/// 单次 Guardrail 违规(P1-7:可序列化,供审计持久化)。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailViolation {
    pub guardrail_name: String,
    pub stage: String,
    pub reason: String,
}

/// 输出验证结果三态(P1-5)
///
/// `fail_fast=false` 时,Block 不再立即抛错丢弃当前值;剩余护栏仍会执行,
/// 后续 `Modify` 依然保留,最终把(可能被改写过的)部分输出放进
/// `Blocked::partial`。
#[derive(Debug)]
pub enum OutputValidation {
    /// 全部通过,`value` 为最终值(可能被多个 `Modify` 改写)。
    Passed(String),
    /// 被拦截,`partial` 为拦截前的已处理输出。
    Blocked { reason: String, partial: String },
}

/// Guardrail 执行器:按配置依次执行 input / output / streaming guardrails。
#[derive(Clone)]
pub struct GuardrailRunner {
    config: GuardrailsConfig,
    /// 违规日志:Arc 共享,`Clone` 出的 runner 与本体记录同一份日志。
    ///
    /// `invoke_stream` 的两阶段各持一份 runner 克隆,流内违规即时可见于
    /// `GuardedAgent::violations()`,无需事后合并。
    violations: Arc<std::sync::Mutex<Vec<GuardrailViolation>>>,
}

impl GuardrailRunner {
    pub fn new(config: GuardrailsConfig) -> Self {
        Self {
            config,
            violations: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// 记录违规:写入有界共享日志 + (可选)异步审计持久化(P1-2/P1-7)。
    async fn record_violation(&mut self, violation: GuardrailViolation) {
        {
            let mut violations = self
                .violations
                .lock()
                .expect("guardrail violations mutex poisoned");
            violations.push(violation.clone());
            // 有界:丢弃最旧,保持固定上限。
            if violations.len() > MAX_VIOLATIONS {
                violations.remove(0);
            }
        } // 守卫在此释放,不跨 await 持有。
        if let Some(sink) = &self.config.audit_sink {
            sink.record(&violation).await;
        }
    }

    /// 验证输入。
    ///
    /// 拦截时返回 [`GuardrailError::Blocked`](输入侧无 partial/建议)。
    pub async fn validate_input(&mut self, input: &str) -> Result<(), GuardrailError> {
        let mut first_block: Option<String> = None;
        // 克隆列表:避免在持有 `&self.config` 借用时调用 `&mut self` 方法。
        let guardrails = self.config.input_guardrails.clone();
        for g in &guardrails {
            // 输入侧结果类型没有 `Modify`,这里不可能静默丢弃改写结果。
            if let InputGuardrailResult::Block { reason } = g.validate(input).await {
                self.record_violation(GuardrailViolation {
                    guardrail_name: g.name().to_string(),
                    stage: "input".to_string(),
                    reason: reason.clone(),
                })
                .await;
                if first_block.is_none() {
                    first_block = Some(reason);
                }
                if self.config.fail_fast {
                    break;
                }
            }
        }
        if let Some(reason) = first_block {
            return Err(GuardrailError::Blocked {
                reason,
                partial: None,
                suggestion: None,
            });
        }
        Ok(())
    }

    /// 验证输出(支持 Modify)。返回三态结果而非直接抛错:
    ///
    /// - `fail_fast=true`:首个 Block 即停,`partial` 为当前已处理输出。
    /// - `fail_fast=false`:继续执行剩余护栏,后续 `Modify` 仍保留,最终以
    ///   `Blocked::partial` 携带(可能被改写过的)部分输出。
    pub async fn validate_output(&mut self, output: &str) -> OutputValidation {
        let mut current = output.to_string();
        let mut first_block: Option<String> = None;
        // 克隆列表:避免在持有 `&self.config` 借用时调用 `&mut self` 方法。
        let guardrails = self.config.output_guardrails.clone();
        for g in &guardrails {
            match g.validate(&current).await {
                OutputGuardrailResult::Pass => {}
                OutputGuardrailResult::Block { reason } => {
                    self.record_violation(GuardrailViolation {
                        guardrail_name: g.name().to_string(),
                        stage: "output".to_string(),
                        reason: reason.clone(),
                    })
                    .await;
                    if first_block.is_none() {
                        first_block = Some(reason);
                    }
                    if self.config.fail_fast {
                        break;
                    }
                }
                OutputGuardrailResult::Modify { new_value } => {
                    // Modify 是护栏干预,记录违规便于审计,同时保留改写结果。
                    self.record_violation(GuardrailViolation {
                        guardrail_name: g.name().to_string(),
                        stage: "output".to_string(),
                        reason: format!("output modified by {}", g.name()),
                    })
                    .await;
                    current = new_value;
                }
            }
        }
        match first_block {
            Some(reason) => OutputValidation::Blocked {
                reason,
                partial: current,
            },
            None => OutputValidation::Passed(current),
        }
    }

    /// 两阶段流式检查的第一阶段:对增量 chunk(可能是 `tail + chunk`)逐块验证。
    ///
    /// 返回 [`ChunkAction`]:放行 / 改写后放行 / 拦截丢弃。
    /// 完整输出的二次复查由 [`GuardrailRunner::validate_output`] 承担(P1-4)。
    pub async fn validate_stream_chunk(&mut self, chunk: &str) -> ChunkAction {
        let mut action = ChunkAction::Pass;
        // 克隆列表:避免在持有 `&self.config` 借用时调用 `&mut self` 方法。
        let guardrails = self.config.streaming_guardrails.clone();
        for g in &guardrails {
            match g.validate_chunk(chunk).await {
                ChunkAction::Pass => {}
                ChunkAction::Replace(new_value) => {
                    self.record_violation(GuardrailViolation {
                        guardrail_name: g.name().to_string(),
                        stage: "stream".to_string(),
                        reason: "chunk replaced".to_string(),
                    })
                    .await;
                    action = ChunkAction::Replace(new_value);
                }
                ChunkAction::Block => {
                    self.record_violation(GuardrailViolation {
                        guardrail_name: g.name().to_string(),
                        stage: "stream".to_string(),
                        reason: "chunk blocked".to_string(),
                    })
                    .await;
                    return ChunkAction::Block;
                }
            }
        }
        action
    }

    /// 获取违规记录快照(共享日志的克隆)。
    ///
    /// 返回 owned `Vec` 而非切片:Mutex 守卫不能跨调用存活。
    pub fn violations(&self) -> Vec<GuardrailViolation> {
        self.violations
            .lock()
            .expect("guardrail violations mutex poisoned")
            .clone()
    }

    /// 清理违规记录(P1-2)。共享同一日志的 runner 一起清空。
    pub fn clear_violations(&mut self) {
        self.violations
            .lock()
            .expect("guardrail violations mutex poisoned")
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditSink;
    use crate::guardrail::{InputGuardrail, OutputGuardrail, StreamingOutputGuardrail};
    use async_trait::async_trait;
    use std::sync::Arc;

    struct AlwaysBlock;
    #[async_trait]
    impl InputGuardrail for AlwaysBlock {
        fn name(&self) -> &str {
            "AlwaysBlock"
        }
        async fn validate(&self, _input: &str) -> InputGuardrailResult {
            InputGuardrailResult::Block {
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
        async fn validate(&self, _input: &str) -> InputGuardrailResult {
            InputGuardrailResult::Pass
        }
    }

    /// 输出侧 Modify 护栏:把邮箱替换为 [REDACTED]。
    struct RedactEmail;
    #[async_trait]
    impl OutputGuardrail for RedactEmail {
        fn name(&self) -> &str {
            "RedactEmail"
        }
        async fn validate(&self, output: &str) -> OutputGuardrailResult {
            let redacted = output.replace("user@example.com", "[REDACTED]");
            if redacted != output {
                OutputGuardrailResult::Modify {
                    new_value: redacted,
                }
            } else {
                OutputGuardrailResult::Pass
            }
        }
    }

    /// 恒 Block 的输出护栏。
    struct AlwaysBlockOutput;
    #[async_trait]
    impl OutputGuardrail for AlwaysBlockOutput {
        fn name(&self) -> &str {
            "AlwaysBlockOutput"
        }
        async fn validate(&self, _output: &str) -> OutputGuardrailResult {
            OutputGuardrailResult::Block {
                reason: "always output".to_string(),
            }
        }
    }

    /// 命中关键词即 Block 的流式护栏。
    struct KeywordStreamGuard;
    #[async_trait]
    impl StreamingOutputGuardrail for KeywordStreamGuard {
        fn name(&self) -> &str {
            "KeywordStreamGuard"
        }
        async fn validate_chunk(&self, chunk: &str) -> ChunkAction {
            if chunk.contains("SECRET") {
                ChunkAction::Block
            } else {
                ChunkAction::Pass
            }
        }
    }

    /// 命中关键词即 Replace 的流式护栏。
    struct RedactStreamGuard;
    #[async_trait]
    impl StreamingOutputGuardrail for RedactStreamGuard {
        fn name(&self) -> &str {
            "RedactStreamGuard"
        }
        async fn validate_chunk(&self, chunk: &str) -> ChunkAction {
            if chunk.contains("secret") {
                ChunkAction::Replace(chunk.replace("secret", "***"))
            } else {
                ChunkAction::Pass
            }
        }
    }

    /// 计数审计 sink。
    struct CountingSink {
        recorded: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl AuditSink for CountingSink {
        fn name(&self) -> &str {
            "CountingSink"
        }
        async fn record(&self, _violation: &GuardrailViolation) {
            self.recorded
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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

    #[tokio::test]
    async fn test_runner_output_modify() {
        // 输出护栏返回 Modify,validate_output 返回改写后的值(Passed),并记录一条违规。
        let config = GuardrailsConfig::new().with_output(Arc::new(RedactEmail));
        let mut runner = GuardrailRunner::new(config);
        match runner.validate_output("contact user@example.com").await {
            OutputValidation::Passed(value) => assert_eq!(value, "contact [REDACTED]"),
            other => panic!("应为 Passed, 实际: {:?}", other),
        }
        assert_eq!(runner.violations().len(), 1);
    }

    #[tokio::test]
    async fn test_runner_output_modify_then_block_fail_fast() {
        // fail_fast=true:先 Modify 再 Block,立即返回 Blocked,partial 携带已改写值。
        let config = GuardrailsConfig::new()
            .with_output(Arc::new(RedactEmail))
            .with_output(Arc::new(AlwaysBlockOutput))
            .fail_fast(true);
        let mut runner = GuardrailRunner::new(config);
        match runner.validate_output("contact user@example.com").await {
            OutputValidation::Blocked { reason, partial } => {
                assert!(reason.contains("always output"));
                assert_eq!(partial, "contact [REDACTED]");
            }
            other => panic!("应为 Blocked, 实际: {:?}", other),
        }
        assert_eq!(runner.violations().len(), 2);
    }

    #[tokio::test]
    async fn test_runner_output_blocked_preserves_later_modify_no_fail_fast() {
        // fail_fast=false(P1-5):Block 后仍执行剩余护栏,后续 Modify 保留进 partial。
        let config = GuardrailsConfig::new()
            .with_output(Arc::new(AlwaysBlockOutput))
            .with_output(Arc::new(RedactEmail))
            .fail_fast(false);
        let mut runner = GuardrailRunner::new(config);
        match runner.validate_output("contact user@example.com").await {
            OutputValidation::Blocked { reason, partial } => {
                assert!(reason.contains("always output"));
                assert_eq!(partial, "contact [REDACTED]");
            }
            other => panic!("应为 Blocked, 实际: {:?}", other),
        }
        assert_eq!(runner.violations().len(), 2);
    }

    #[tokio::test]
    async fn test_runner_stream_chunk_block() {
        let config = GuardrailsConfig::new().with_streaming(Arc::new(KeywordStreamGuard));
        let mut runner = GuardrailRunner::new(config);
        // 滑动窗口探测串内含 SECRET → Block
        assert_eq!(
            runner.validate_stream_chunk("x SECRET y").await,
            ChunkAction::Block
        );
        assert_eq!(runner.violations().len(), 1);
    }

    #[tokio::test]
    async fn test_runner_stream_chunk_replace() {
        let config = GuardrailsConfig::new().with_streaming(Arc::new(RedactStreamGuard));
        let mut runner = GuardrailRunner::new(config);
        match runner.validate_stream_chunk("a secret b").await {
            ChunkAction::Replace(v) => assert_eq!(v, "a *** b"),
            other => panic!("应为 Replace, 实际: {:?}", other),
        }
        assert_eq!(runner.violations().len(), 1);
    }

    #[tokio::test]
    async fn test_runner_violations_bounded() {
        // 超过 MAX_VIOLATIONS 后丢弃最旧,保持有界(P1-2)。
        let config = GuardrailsConfig::new();
        let mut runner = GuardrailRunner::new(config);
        for i in 0..(MAX_VIOLATIONS + 5) {
            runner
                .record_violation(GuardrailViolation {
                    guardrail_name: format!("g{}", i),
                    stage: "test".to_string(),
                    reason: "x".to_string(),
                })
                .await;
        }
        assert_eq!(runner.violations().len(), MAX_VIOLATIONS);
        // 最旧记录已被丢弃
        assert_ne!(runner.violations()[0].guardrail_name, "g0");
    }

    #[tokio::test]
    async fn test_runner_clear_violations() {
        let config = GuardrailsConfig::new().with_input(Arc::new(AlwaysBlock));
        let mut runner = GuardrailRunner::new(config);
        let _ = runner.validate_input("hi").await;
        assert_eq!(runner.violations().len(), 1);
        runner.clear_violations();
        assert!(runner.violations().is_empty());
    }

    #[tokio::test]
    async fn test_runner_audit_sink_records() {
        let sink = Arc::new(CountingSink {
            recorded: std::sync::atomic::AtomicUsize::new(0),
        });
        let config = GuardrailsConfig::new()
            .with_input(Arc::new(AlwaysBlock))
            .with_audit_sink(sink.clone());
        let mut runner = GuardrailRunner::new(config);
        let _ = runner.validate_input("hi").await;
        assert_eq!(sink.recorded.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
