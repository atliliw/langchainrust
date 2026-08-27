//! Guardrail trait, result types, configuration, and errors

use async_trait::async_trait;
use std::sync::Arc;

/// Input Guardrail validation result
///
/// The input side does not allow `Modify`: an input guardrail either passes or blocks.
/// Split from [`OutputGuardrailResult`], the type system enforces "Modify is output-only",
/// so an input guardrail cannot return a rewritten result at compile time.
#[derive(Debug, Clone)]
pub enum InputGuardrailResult {
    /// Pass
    Pass,
    /// Block
    Block {
        /// Block reason
        reason: String,
    },
}

impl InputGuardrailResult {
    /// Whether it passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, InputGuardrailResult::Pass)
    }
    /// Whether it was blocked.
    pub fn is_block(&self) -> bool {
        matches!(self, InputGuardrailResult::Block { .. })
    }
}

/// Output Guardrail validation result
///
/// `Modify` is output-side only: an output guardrail can rewrite the result before passing it.
#[derive(Debug, Clone)]
pub enum OutputGuardrailResult {
    /// Pass
    Pass,
    /// Block
    Block {
        /// Block reason
        reason: String,
    },
    /// Passed after modification (output side only)
    Modify {
        /// The new value after modification
        new_value: String,
    },
}

impl OutputGuardrailResult {
    /// Whether it passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, OutputGuardrailResult::Pass)
    }
    /// Whether it was blocked.
    pub fn is_block(&self) -> bool {
        matches!(self, OutputGuardrailResult::Block { .. })
    }
    /// Whether it was modified and then passed.
    pub fn is_modify(&self) -> bool {
        matches!(self, OutputGuardrailResult::Modify { .. })
    }
}

/// Input Guardrail trait
///
/// Returns [`InputGuardrailResult`] (no `Modify` variant), so the input side cannot rewrite by construction.
#[async_trait]
pub trait InputGuardrail: Send + Sync {
    /// The guardrail's name.
    fn name(&self) -> &str;
    /// Validates the input and returns a result.
    async fn validate(&self, input: &str) -> InputGuardrailResult;
}

/// Output Guardrail trait
///
/// Returns [`OutputGuardrailResult`] (with the `Modify` variant); this is the only legal entry point for rewriting.
#[async_trait]
pub trait OutputGuardrail: Send + Sync {
    /// The guardrail's name.
    fn name(&self) -> &str;
    /// Validates the output and returns a result.
    async fn validate(&self, output: &str) -> OutputGuardrailResult;
}

/// Streaming chunk action
///
/// The streaming guardrail's disposition for a single chunk (P1-4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkAction {
    /// Pass
    Pass,
    /// Passed after rewriting
    Replace(String),
    /// Blocked and dropped
    Block,
}

/// Streaming output guardrail trait (P1-4)
///
/// Phase one: quickly check each incremental chunk, blocking sensitive information before it is
/// shown to the user. The caller maintains a sliding window (`tail + chunk`) to avoid keywords
/// split across chunks (e.g. `"passwo" + "rd"`). The second re-check after the full output is
/// handled by [`OutputGuardrail`] (`GuardrailRunner::validate_output`).
#[async_trait]
pub trait StreamingOutputGuardrail: Send + Sync {
    /// The guardrail's name.
    fn name(&self) -> &str;
    /// Incrementally checks a chunk (possibly a `tail + chunk` combined string).
    async fn validate_chunk(&self, chunk: &str) -> ChunkAction;
}

/// Guardrails configuration
#[derive(Clone)]
pub struct GuardrailsConfig {
    /// Input guardrail list
    pub input_guardrails: Vec<Arc<dyn InputGuardrail>>,
    /// Output guardrail list
    pub output_guardrails: Vec<Arc<dyn OutputGuardrail>>,
    /// Streaming guardrails: incrementally check each chunk during streaming output (P1-4).
    pub streaming_guardrails: Vec<Arc<dyn StreamingOutputGuardrail>>,
    /// Audit persistence sink (P1-7).
    pub audit_sink: Option<Arc<dyn crate::audit::AuditSink>>,
    /// Whether to fail fast (stop at the first block)
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
    /// Creates a default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an input guardrail.
    pub fn with_input(mut self, g: Arc<dyn InputGuardrail>) -> Self {
        self.input_guardrails.push(g);
        self
    }

    /// Adds an output guardrail.
    pub fn with_output(mut self, g: Arc<dyn OutputGuardrail>) -> Self {
        self.output_guardrails.push(g);
        self
    }

    /// Adds a streaming guardrail (phase one of the two-phase streaming check).
    pub fn with_streaming(mut self, g: Arc<dyn StreamingOutputGuardrail>) -> Self {
        self.streaming_guardrails.push(g);
        self
    }

    /// Configures the audit persistence sink: written synchronously on every violation record (P1-7).
    pub fn with_audit_sink(mut self, sink: Arc<dyn crate::audit::AuditSink>) -> Self {
        self.audit_sink = Some(sink);
        self
    }

    /// Sets whether to fail fast.
    pub fn fail_fast(mut self, v: bool) -> Self {
        self.fail_fast = v;
        self
    }
}

/// Guardrail error
///
/// `Blocked` carries the block reason + the already-handled part + a user-facing suggestion
/// (P1-1/P1-6), letting the upper layer tell the user "blocked" rather than "system error".
#[derive(Debug)]
#[non_exhaustive]
pub enum GuardrailError {
    /// Blocked by a Guardrail
    Blocked {
        /// Block reason (explanation from the guardrail side)
        reason: String,
        /// The part already processed before blocking (for the upper layer to show partial results / decide whether to regenerate)
        partial: Option<String>,
        /// User-facing suggestion (how to rephrase the input / fix the output)
        suggestion: Option<String>,
    },
    /// Agent execution error
    AgentError(String),
    /// Sensitive-leak judge error (P2-3): returned when the LLM judge cannot make a decision.
    Judge(String),
}

impl GuardrailError {
    /// Constructs a `GuardrailError` with a user suggestion from `OutputValidation::Blocked`.
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
                write!(f, "guardrail blocked: {}", reason)?;
                if let Some(p) = partial {
                    write!(f, " (partial handled: {})", p)?;
                }
                if let Some(s) = suggestion {
                    write!(f, " suggestion: {}", s)?;
                }
                Ok(())
            }
            GuardrailError::AgentError(msg) => write!(f, "agent execution error: {}", msg),
            GuardrailError::Judge(msg) => write!(f, "Sensitive judge error: {}", msg),
        }
    }
}

impl std::error::Error for GuardrailError {}
