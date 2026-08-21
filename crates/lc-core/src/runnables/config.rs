// src/core/runnables/config.rs
//! Runnable execution configuration.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use lc_callbacks::CallbackManager;

use super::cancellation::CancellationToken;

/// Runnable execution configuration.
#[derive(Debug, Clone, Default)]
pub struct RunnableConfig {
    /// Tags for filtering and tracking.
    pub tags: Vec<String>,

    /// Metadata - custom data (JSON serializable).
    pub metadata: HashMap<String, Value>,

    /// Max concurrency for batch operations.
    pub max_concurrency: Option<usize>,

    /// Run ID for tracking.
    pub run_id: Option<Uuid>,

    /// Run name for debugging.
    pub run_name: Option<String>,

    /// Callback manager for tracing and monitoring.
    pub callbacks: Option<Arc<CallbackManager>>,

    /// Cancellation token for aborting long-running operations.
    pub cancellation_token: Option<CancellationToken>,

    /// Sampling temperature override for the current call.
    ///
    /// When set, takes precedence over the model's own configured
    /// temperature. This is how wrapper layers (e.g. `LLMClient`) make
    /// `with_temperature` effective through a trait object (providers Q2).
    pub temperature: Option<f32>,

    /// Max-tokens override for the current call.
    ///
    /// When set, takes precedence over the model's own configured max
    /// tokens (providers Q2).
    pub max_tokens: Option<usize>,

    /// Configurable values — the Rust counterpart of Python LCEL's
    /// `config["configurable"]`.
    ///
    /// Runtime selection keys live here: `configurable_alternatives` reads
    /// its `which` key from this map, and `RunnableWithMessageHistory`
    /// (session mode) reads `session_id` from it.
    pub configurable: HashMap<String, Value>,
}

impl RunnableConfig {
    /// Creates an empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Sets max concurrency.
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Sets run ID.
    pub fn with_run_id(mut self, id: Uuid) -> Self {
        self.run_id = Some(id);
        self
    }

    /// Sets run name.
    pub fn with_run_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = Some(name.into());
        self
    }

    /// Sets callback manager.
    pub fn with_callbacks(mut self, callbacks: Arc<CallbackManager>) -> Self {
        self.callbacks = Some(callbacks);
        self
    }

    /// Sets cancellation token.
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Sets a sampling temperature override for the call.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets a max-tokens override for the call.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets a configurable value — the Rust counterpart of Python LCEL's
    /// `config["configurable"][key] = value`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 给 session 记忆选槽 / 给 configurable_alternatives 路由
    /// let config = RunnableConfig::new().with_configurable("session_id", json!("s1"));
    /// ```
    pub fn with_configurable(mut self, key: impl Into<String>, value: Value) -> Self {
        self.configurable.insert(key.into(), value);
        self
    }

    /// Reads a configurable value by key.
    pub fn configurable_value(&self, key: &str) -> Option<&Value> {
        self.configurable.get(key)
    }

    /// Checks if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token
            .as_ref()
            .is_some_and(|t| t.is_cancelled())
    }

    /// Merges two configurations (later overrides earlier).
    pub fn merge(mut self, other: RunnableConfig) -> Self {
        // Merge tags as an order-preserving union (dedup without sorting).
        for tag in other.tags {
            if !self.tags.iter().any(|existing| existing == &tag) {
                self.tags.push(tag);
            }
        }

        // Merge metadata (override)
        self.metadata.extend(other.metadata);

        // Merge configurable values (override)
        self.configurable.extend(other.configurable);

        // Override other fields
        if other.max_concurrency.is_some() {
            self.max_concurrency = other.max_concurrency;
        }
        if other.run_id.is_some() {
            self.run_id = other.run_id;
        }
        if other.run_name.is_some() {
            self.run_name = other.run_name;
        }
        if other.cancellation_token.is_some() {
            self.cancellation_token = other.cancellation_token;
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }

        // Merge callbacks: append `other`'s handlers instead of replacing the
        // whole manager (Q8), so observers configured at different layers of
        // the pipeline all keep firing.
        if let (Some(self_cb), Some(other_cb)) = (&self.callbacks, &other.callbacks) {
            self.callbacks = Some(Arc::new(self_cb.merge_with(other_cb)));
        } else if other.callbacks.is_some() {
            self.callbacks = other.callbacks;
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configurable_roundtrip() {
        let cfg = RunnableConfig::new().with_configurable("session_id", json!("s1"));
        assert_eq!(cfg.configurable_value("session_id"), Some(&json!("s1")));
        assert_eq!(cfg.configurable_value("missing"), None);
    }

    #[test]
    fn configurable_merge_overrides() {
        let base = RunnableConfig::new().with_configurable("which", json!("a"));
        let other = RunnableConfig::new().with_configurable("which", json!("b"));
        let merged = base.merge(other);
        assert_eq!(merged.configurable_value("which"), Some(&json!("b")));
        // 不存在的键不进 merge 结果
        assert_eq!(merged.configurable_value("none"), None);
    }
}
