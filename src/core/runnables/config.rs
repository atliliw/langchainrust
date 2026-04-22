// src/core/runnables/config.rs
//! Runnable execution configuration.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::callbacks::CallbackManager;

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

    /// Merges two configurations (later overrides earlier).
    pub fn merge(mut self, other: RunnableConfig) -> Self {
        // Merge tags (union)
        self.tags.extend(other.tags);
        self.tags.sort();
        self.tags.dedup();

        // Merge metadata (override)
        self.metadata.extend(other.metadata);

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
        if other.callbacks.is_some() {
            self.callbacks = other.callbacks;
        }

        self
    }
}
