//! Guardrails 安全护栏
//!
//! 提供输入/输出验证,保护生产环境的 Agent:防恶意输入、防敏感信息泄露。
//!
//! # 示例
//! ```no_run
//! use langchainrust::guardrails::{
//!     GuardrailsConfig, MaxLengthGuardrail, SensitiveInfoGuardrail,
//! };
//! use std::sync::Arc;
//!
//! let config = GuardrailsConfig::new()
//!     .with_input(Arc::new(MaxLengthGuardrail::new(1000)))
//!     .with_output(Arc::new(SensitiveInfoGuardrail::new()));
//! ```

pub mod guardrail;
pub mod guarded_agent;
pub mod runner;
pub mod validators;

pub use guardrail::{
    GuardrailError, GuardrailResult, GuardrailsConfig, InputGuardrail, OutputGuardrail,
};
pub use guarded_agent::GuardedAgent;
pub use runner::{GuardrailRunner, GuardrailViolation};
pub use validators::{ForbiddenWordsGuardrail, MaxLengthGuardrail, SensitiveInfoGuardrail};
