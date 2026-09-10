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
pub mod disclosure;
pub mod guarded_agent;
pub mod guardrail;
pub mod judge;
pub mod retrieval_rail;
pub mod rule_of_two;
pub mod runner;
pub mod spotlighting;
pub mod validators;

pub use audit::{AuditSink, FileAuditSink};
pub use disclosure::{disclose, disclose_to_file, DisclosureConfig, DEFAULT_DISCLOSURE};
pub use guarded_agent::{ChainGuardable, Guardable, GuardableChunk, GuardedAgent};
pub use guardrail::{
    ChunkAction, GuardrailError, GuardrailsConfig, InputGuardrail, InputGuardrailResult,
    OutputGuardrail, OutputGuardrailResult, StreamingOutputGuardrail,
};
pub use judge::{LlmSensitiveJudge, SensitiveJudge};
pub use retrieval_rail::{GuardedRetriever, RailAction, RailReport, RetrievalRail, RAIL_FLAG_KEY};
pub use rule_of_two::{is_high_risk, triage, RuleOfTwoVerdict, RULE_OF_TWO_ARMED_THRESHOLD};
pub use spotlighting::{
    escape, is_wrapped, spotlight, unwrap, wrap_tool_output, SpotlightedRetriever,
    DEFAULT_CLOSE_MARKER, DEFAULT_OPEN_MARKER,
};
pub use runner::{GuardrailRunner, GuardrailViolation, OutputValidation};
pub use validators::{ForbiddenWordsGuardrail, MaxLengthGuardrail, SensitiveInfoGuardrail};
