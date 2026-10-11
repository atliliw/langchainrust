// crates/lc/src/core/mod.rs
//! Core — re-export from lc-core crate, plus wrapper that depends on the unified Error type

pub use lc_core::*;

// Wrapper stays in the facade crate because it depends on crate::error::Error
// (the unified error type that aggregates all sub-module errors).
pub mod language_models_wrapper;

// Re-export wrapper types for backward compatibility
pub use language_models_wrapper::{wrap_chat_model, wrap_provider_model, ChatModelWrapper};

// T2-1 (0.27.0) fragility note: the glob above (`pub use lc_core::*;`) ALREADY re-exports
// `lc_core::language_models` as `crate::core::language_models`; the explicit module below
// works ONLY because a spelled-out `pub mod` shadows the glob re-export at the same path.
// A future layout change that drops this `pub mod language_models` (e.g. "the glob covers
// it") silently reverts to `lc_core::language_models` and hides the facade's wrapper
// constructors without any compile error in the facade itself. The regression is caught
// by the `core_language_models_override_keeps_wrapper_surface` guard in `lib.rs`'s
// `reexport_guard`, which pin-references these wrapper paths. Keep the two in sync.
// Re-export into language_models namespace so existing code
// using `crate::core::language_models::wrap_chat_model` still works
/// Language models — re-exports from `lc_core::language_models` plus the wrapper types.
pub mod language_models {
    pub use crate::core::language_models_wrapper::{
        wrap_chat_model, wrap_provider_model, ChatModelWrapper,
    };
    pub use lc_core::language_models::*;
}
