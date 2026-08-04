// crates/lc/src/core/mod.rs
//! Core — re-export from lc-core crate, plus wrapper that depends on the unified Error type

pub use lc_core::*;

// Wrapper stays in the facade crate because it depends on crate::error::Error
// (the unified error type that aggregates all sub-module errors).
pub mod language_models_wrapper;

// Re-export wrapper types for backward compatibility
pub use language_models_wrapper::{wrap_chat_model, wrap_provider_model, ChatModelWrapper};

// Re-export into language_models namespace so existing code
// using `crate::core::language_models::wrap_chat_model` still works
pub mod language_models {
    pub use crate::core::language_models_wrapper::{
        wrap_chat_model, wrap_provider_model, ChatModelWrapper,
    };
    pub use lc_core::language_models::*;
}
