// lc-providers/src/sampling.rs
//! Per-call sampling overrides (providers Q2).
//!
//! `RunnableConfig` can carry `temperature` / `max_tokens` for a single
//! call. This is how wrapper layers such as `LLMClient` make their
//! `with_temperature` / `with_max_tokens` effective through a trait object:
//! the override is merged into the config, and each provider applies it over
//! its own configured sampling when building the request.
//!
//! §4: [`SamplingOverride`] + [`with_sampling_overrides`] are the single
//! landing point for that merge, replacing the `let mut effective =
//! self.clone(); if let Some(t) {…} if let Some(m) {…}` block that used to be
//! inlined (and always cloned) at every provider call site.

use std::borrow::Cow;

use lc_core::RunnableConfig;

/// A provider whose configured sampling can take per-call overrides.
///
/// Implemented by each provider; the impl writes the override into the
/// provider's own config so the two shapes (Anthropic's non-optional
/// `max_tokens` vs. every other provider's `Option<usize>`) are handled where
/// they live, and the call site stays uniform.
pub(crate) trait SamplingOverride {
    /// Applies per-call overrides; a `None` field leaves that sampling value
    /// untouched.
    fn apply_sampling_overrides(&mut self, temperature: Option<f32>, max_tokens: Option<usize>);
}

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

/// Returns `value` unchanged when no sampling override is set, otherwise a
/// clone with the overrides applied.
///
/// §4: consolidates the clone-self boilerplate and stops the unconditional
/// clone — a no-override call is a zero-copy borrow, matching the pattern
/// `ResponsesModel` already used.
pub(crate) fn with_sampling_overrides<'a, T: SamplingOverride + Clone>(
    value: &'a T,
    config: &Option<RunnableConfig>,
) -> Cow<'a, T> {
    let (temperature, max_tokens) = sampling_overrides(config);
    // Borrow — nothing to change, no allocation.
    if temperature.is_none() && max_tokens.is_none() {
        return Cow::Borrowed(value);
    }
    let mut effective = value.clone();
    effective.apply_sampling_overrides(temperature, max_tokens);
    Cow::Owned(effective)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Dummy {
        temperature: Option<f32>,
        max_tokens: Option<usize>,
    }

    impl SamplingOverride for Dummy {
        fn apply_sampling_overrides(
            &mut self,
            temperature: Option<f32>,
            max_tokens: Option<usize>,
        ) {
            if let Some(t) = temperature {
                self.temperature = Some(t);
            }
            if let Some(m) = max_tokens {
                self.max_tokens = Some(m);
            }
        }
    }

    fn cfg(temp: Option<f32>, max: Option<usize>) -> Option<RunnableConfig> {
        Some(RunnableConfig {
            temperature: temp,
            max_tokens: max,
            ..Default::default()
        })
    }

    #[test]
    fn no_override_is_a_zero_copy_borrow() {
        let base = Dummy {
            temperature: Some(0.7),
            max_tokens: Some(128),
        };
        let overridden = with_sampling_overrides(&base, &None);
        // Borrowed, not cloned.
        assert!(matches!(overridden, Cow::Borrowed(_)));
        assert_eq!(overridden.temperature, Some(0.7));
        assert_eq!(overridden.max_tokens, Some(128));
    }

    #[test]
    fn temperature_override_applies_and_clones() {
        let base = Dummy {
            temperature: Some(0.7),
            max_tokens: Some(128),
        };
        let overridden = with_sampling_overrides(&base, &cfg(Some(0.2), None));
        assert!(matches!(overridden, Cow::Owned(_)));
        assert_eq!(overridden.temperature, Some(0.2), "override wins");
        assert_eq!(overridden.max_tokens, Some(128), "untouched by None");
    }

    #[test]
    fn max_tokens_override_applies() {
        let base = Dummy {
            temperature: None,
            max_tokens: Some(128),
        };
        let overridden = with_sampling_overrides(&base, &cfg(None, Some(4096)));
        assert_eq!(overridden.temperature, None);
        assert_eq!(overridden.max_tokens, Some(4096));
    }
}
