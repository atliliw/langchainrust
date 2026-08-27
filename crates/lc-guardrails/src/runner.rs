//! Guardrail runner

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::guardrail::{
    ChunkAction, GuardrailError, GuardrailsConfig, InputGuardrailResult, OutputGuardrailResult,
};

/// Violation log cap: beyond it the oldest record is dropped to prevent unbounded memory growth (P1-2).
const MAX_VIOLATIONS: usize = 1000;

/// A single Guardrail violation (P1-7: serializable, for audit persistence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailViolation {
    /// Name of the guardrail that fired
    pub guardrail_name: String,
    /// Violation stage (input / output / stream)
    pub stage: String,
    /// Violation reason
    pub reason: String,
}

/// Three-state output validation result (P1-5)
///
/// With `fail_fast=false`, a Block no longer immediately errors out and discards the current
/// value; the remaining guardrails still run, later `Modify`s are still kept, and the (possibly
/// rewritten) partial output ends up in `Blocked::partial`.
#[derive(Debug)]
pub enum OutputValidation {
    /// All passed; `value` is the final value (possibly rewritten by multiple `Modify`s).
    Passed(String),
    /// Blocked; `partial` is the processed output before blocking.
    Blocked {
        /// Block reason
        reason: String,
        /// The partial output processed before blocking
        partial: String,
    },
}

/// Guardrail runner: runs input / output / streaming guardrails in order per the config.
#[derive(Clone)]
pub struct GuardrailRunner {
    config: GuardrailsConfig,
    /// Violation log: shared via Arc; a runner produced by `Clone` records into the same log as the original.
    ///
    /// Each phase of `invoke_stream` holds its own runner clone, so in-stream violations are
    /// immediately visible via `GuardedAgent::violations()` without a later merge.
    violations: Arc<std::sync::Mutex<Vec<GuardrailViolation>>>,
}

impl GuardrailRunner {
    /// Creates a runner with the given config.
    pub fn new(config: GuardrailsConfig) -> Self {
        Self {
            config,
            violations: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Records a violation: writes to the bounded shared log + (optional) async audit persistence (P1-2/P1-7).
    async fn record_violation(&mut self, violation: GuardrailViolation) {
        {
            let mut violations = self.violations.lock().unwrap_or_else(|e| e.into_inner());
            violations.push(violation.clone());
            // bounded: drop the oldest, keep a fixed cap.
            if violations.len() > MAX_VIOLATIONS {
                violations.remove(0);
            }
        } // guard is released here, not held across an await.
        if let Some(sink) = &self.config.audit_sink {
            sink.record(&violation).await;
        }
    }

    /// Validates input.
    ///
    /// On blocking, returns [`GuardrailError::Blocked`] (the input side has no partial/suggestion).
    pub async fn validate_input(&mut self, input: &str) -> Result<(), GuardrailError> {
        let mut first_block: Option<String> = None;
        // clone the list: avoid calling `&mut self` methods while holding a `&self.config` borrow.
        let guardrails = self.config.input_guardrails.clone();
        for g in &guardrails {
            // the input-side result type has no `Modify`, so a rewritten result cannot be silently dropped here.
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

    /// Validates output (supports Modify). Returns a three-state result instead of erroring directly:
    ///
    /// - `fail_fast=true`: stops at the first Block; `partial` is the currently processed output.
    /// - `fail_fast=false`: continues running the remaining guardrails, keeps later `Modify`s, and
    ///   `Blocked::partial` carries the (possibly rewritten) partial output.
    pub async fn validate_output(&mut self, output: &str) -> OutputValidation {
        let mut current = output.to_string();
        let mut first_block: Option<String> = None;
        // clone the list: avoid calling `&mut self` methods while holding a `&self.config` borrow.
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
                    // Modify is a guardrail intervention: record the violation for auditing while keeping the rewritten result.
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

    /// Phase one of the two-phase streaming check: validates each incremental chunk (possibly a `tail + chunk`).
    ///
    /// Returns a [`ChunkAction`]: pass / pass after rewrite / block and drop.
    /// The second re-check of the full output is handled by [`GuardrailRunner::validate_output`] (P1-4).
    pub async fn validate_stream_chunk(&mut self, chunk: &str) -> ChunkAction {
        let mut action = ChunkAction::Pass;
        // clone the list: avoid calling `&mut self` methods while holding a `&self.config` borrow.
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

    /// Returns a snapshot of violation records (a clone of the shared log).
    ///
    /// Returns an owned `Vec` rather than a slice: a Mutex guard cannot outlive the call.
    pub fn violations(&self) -> Vec<GuardrailViolation> {
        self.violations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clears violation records (P1-2). All runners sharing the same log are cleared together.
    pub fn clear_violations(&mut self) {
        self.violations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
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

    /// Output-side Modify guardrail: replaces emails with [REDACTED].
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

    /// Output guardrail that always blocks.
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

    /// Streaming guardrail that blocks on a keyword match.
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

    /// Streaming guardrail that replaces on a keyword match.
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

    /// Counting audit sink.
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
        // fail_fast=true: returns at the first block, records only 1
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
        // fail_fast=false: checks all, records 2
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
        // the output guardrail returns Modify; validate_output returns the rewritten value (Passed) and records one violation.
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
        // fail_fast=true: Modify then Block -> Blocked returned immediately, partial carries the rewritten value.
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
        // fail_fast=false (P1-5): guardrails still run after a Block, and later Modify results are kept in partial.
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
        // sliding-window probe contains SECRET -> Block
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
        // beyond MAX_VIOLATIONS the oldest is dropped, keeping it bounded (P1-2).
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
        // the oldest record has been dropped
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
