// lc-providers/src/sampling.rs
//! Per-call sampling overrides (providers Q2).
//!
//! `RunnableConfig` can carry `temperature` / `max_tokens` for a single
//! call. This is how wrapper layers such as `LLMClient` make their
//! `with_temperature` / `with_max_tokens` effective through a trait object:
//! the override is merged into the config, and each provider applies it over
//! its own configured sampling when building the request.

use lc_core::RunnableConfig;

/// Extract the per-call sampling overrides from a `RunnableConfig`.
///
/// Returns `(temperature, max_tokens)`; each is `None` when the config does
/// not set it.
pub(crate) fn sampling_overrides(config: &Option<RunnableConfig>) -> (Option<f32>, Option<usize>) {
    config
        .as_ref()
        .map(|c| (c.temperature, c.max_tokens))
        .unwrap_or((None, None))
}
