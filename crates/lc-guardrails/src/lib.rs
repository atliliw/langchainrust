#![warn(missing_docs)]
//! Guardrails — safety guardrails
//!
//! Provides input/output validation to protect agents in production: defend against malicious input and sensitive-information leaks.
//!
//! # Example
//! ```no_run
//! use lc_guardrails::{
//!     GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail,
//! };
//! use std::sync::Arc;
//!
//! let config = GuardrailsConfig::new()
//!     .with_input(Arc::new(MaxLengthGuardrail::new(1000)))
//!     .with_output(Arc::new(SensitiveInfoGuardrail::new()));
//! ```

pub mod audit;
pub mod guarded_agent;
pub mod guardrail;
pub mod judge;
pub mod runner;
pub mod validators;

pub use audit::{AuditSink, FileAuditSink};
pub use guarded_agent::{ChainGuardable, Guardable, GuardableChunk, GuardedAgent};
pub use guardrail::{
    ChunkAction, GuardrailError, GuardrailsConfig, InputGuardrail, InputGuardrailResult,
    OutputGuardrail, OutputGuardrailResult, StreamingOutputGuardrail,
};
pub use judge::{LlmSensitiveJudge, SensitiveJudge};
pub use runner::{GuardrailRunner, GuardrailViolation, OutputValidation};
pub use validators::{ForbiddenWordsGuardrail, MaxLengthGuardrail, SensitiveInfoGuardrail};
