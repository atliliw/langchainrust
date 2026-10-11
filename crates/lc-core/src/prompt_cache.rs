//! Provider-agnostic prompt-cache control.
//!
//! Prompt caching is a provider-native feature with different shapes: Anthropic
//! takes explicit `cache_control` breakpoints, OpenAI caches automatically and
//! only lets you route a request to a cache bucket with `prompt_cache_key`.
//! [`PromptCache`] is the single option callers set; each provider maps the
//! parts it supports and **silently ignores the rest**.
//!
//! Provider mapping:
//!
//! | Provider | Mapping |
//! |----------|---------|
//! | Anthropic | `cache_control: {"type":"ephemeral"[,"ttl":…]}` breakpoints on the stable prefix (system prompt + last tool) and, when [`caches_last_message`](PromptCache::caches_last_message) is set, on the last message content block. [`ttl`](PromptCache::ttl) selects the long-lived marker. `key` is ignored. |
//! | OpenAI | Caching is automatic and server-side, so no breakpoints are sent; [`key`](PromptCache::key) becomes the request's `prompt_cache_key`, which pins related requests to the same cache bucket. `ttl` and the last-message breakpoint are ignored. |
//! | Everything else (OpenAI-compatible routers, Azure, Gemini, Cohere, Ollama, the OpenAI-protocol vendor wrappers) | No native equivalent — the option is accepted and ignored. |
//!
//! ```
//! use lc_core::PromptCache;
//!
//! let cache = PromptCache::new()
//!     .with_ttl("1h")
//!     .with_key("tenant-42")
//!     .with_last_message_breakpoint(true);
//! assert_eq!(cache.ttl(), Some("1h"));
//! assert!(cache.caches_last_message());
//! ```

/// Native prompt-cache control, applied by whichever provider understands it.
///
/// Construct with [`PromptCache::new`] and refine with the `with_*` builders.
/// An all-default value is meaningful: it asks the provider for its plain
/// caching behavior (Anthropic's 5-minute ephemeral breakpoints, OpenAI's
/// automatic cache with no explicit key).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptCache {
    ttl: Option<String>,
    key: Option<String>,
    last_message_breakpoint: bool,
}

impl PromptCache {
    /// Creates a default prompt-cache request (provider defaults, no key).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a cache TTL for the breakpoints (Anthropic only).
    ///
    /// Anthropic accepts `"5m"` (the default ephemeral window) and `"1h"`;
    /// anything else is forwarded verbatim and rejected by the API. Providers
    /// without a TTL knob ignore it.
    pub fn with_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.ttl = Some(ttl.into());
        self
    }

    /// Sets the OpenAI `prompt_cache_key`.
    ///
    /// OpenAI routes requests that share a key to the same cache bucket;
    /// without a key the cache is keyed on the prompt prefix alone. Providers
    /// that have no such field ignore it.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Also marks the last message content block as a cache breakpoint
    /// (Anthropic only).
    ///
    /// Off by default: a breakpoint on the conversation tail is only worth its
    /// slot when the same prefix is replayed across turns (multi-turn agent
    /// loops). It is ignored by providers that have no breakpoint concept.
    pub fn with_last_message_breakpoint(mut self, on: bool) -> Self {
        self.last_message_breakpoint = on;
        self
    }

    /// The configured TTL, if any.
    pub fn ttl(&self) -> Option<&str> {
        self.ttl.as_deref()
    }

    /// The configured cache key, if any.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Whether the last message block should carry a breakpoint.
    pub fn caches_last_message(&self) -> bool {
        self.last_message_breakpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_off() {
        let cache = PromptCache::new();
        assert_eq!(cache.ttl(), None);
        assert_eq!(cache.key(), None);
        assert!(!cache.caches_last_message());
    }

    #[test]
    fn builders_set_each_field() {
        let cache = PromptCache::new()
            .with_ttl("1h")
            .with_key("tenant-42")
            .with_last_message_breakpoint(true);
        assert_eq!(cache.ttl(), Some("1h"));
        assert_eq!(cache.key(), Some("tenant-42"));
        assert!(cache.caches_last_message());
    }

    #[test]
    fn builders_are_chainable_and_replaceable() {
        let cache = PromptCache::new()
            .with_ttl("5m")
            .with_ttl("1h")
            .with_last_message_breakpoint(true)
            .with_last_message_breakpoint(false);
        assert_eq!(cache.ttl(), Some("1h"));
        assert!(!cache.caches_last_message());
    }
}
