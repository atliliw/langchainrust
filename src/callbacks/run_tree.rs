// src/callbacks/run_tree.rs
//! Run tree data structure for tracing

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::RunType;

/// Run tree node for tracing
///
/// Each run is a node in a tree, with optional parent and children.
/// The entire trace forms a tree structure, with the root being the top-level call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTree {
    /// Unique run ID (UUID v7 with timestamp)
    pub id: Uuid,

    /// Run name
    pub name: String,

    /// Run type
    pub run_type: RunType,

    /// Input data
    pub inputs: serde_json::Value,

    /// Output data (set when run ends)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<serde_json::Value>,

    /// Error message (if run failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Parent run ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<Uuid>,

    /// Trace ID (ID of the root run)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<Uuid>,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// End time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<DateTime<Utc>>,

    /// Metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,

    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,

    /// Project name (LangSmith project)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_name: Option<String>,

    /// Serialized representation of the component
    #[serde(default)]
    pub serialized: serde_json::Value,
}

impl RunTree {
    /// Create a new run
    pub fn new(name: impl Into<String>, run_type: RunType, inputs: serde_json::Value) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            run_type,
            inputs,
            outputs: None,
            error: None,
            parent_run_id: None,
            trace_id: None,
            start_time: Utc::now(),
            end_time: None,
            metadata: HashMap::new(),
            tags: Vec::new(),
            project_name: None,
            serialized: serde_json::Value::Null,
        }
    }

    /// Create a child run from this run
    pub fn create_child(
        &self,
        name: impl Into<String>,
        run_type: RunType,
        inputs: serde_json::Value,
    ) -> Self {
        let mut child = Self::new(name, run_type, inputs);
        child.parent_run_id = Some(self.id);
        child.trace_id = self.trace_id.or(Some(self.id));
        child.project_name = self.project_name.clone();
        child
    }

    /// End the run with outputs
    pub fn end(&mut self, outputs: serde_json::Value) {
        self.outputs = Some(outputs);
        self.end_time = Some(Utc::now());
    }

    /// End the run with an error
    pub fn end_with_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
        self.end_time = Some(Utc::now());
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set project name
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project_name = Some(project.into());
        self
    }

    /// Calculate run duration in milliseconds
    pub fn duration_ms(&self) -> Option<i64> {
        self.end_time
            .map(|end| (end - self.start_time).num_milliseconds())
    }
}

/// Simplified run structure for API requests
#[derive(Debug, Serialize)]
pub struct RunCreate {
    pub id: String,
    pub name: String,
    pub run_type: String,
    pub inputs: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
    pub start_time: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl From<&RunTree> for RunCreate {
    fn from(run: &RunTree) -> Self {
        Self {
            id: run.id.to_string(),
            name: run.name.clone(),
            run_type: run.run_type.as_str().to_string(),
            inputs: run.inputs.clone(),
            outputs: run.outputs.clone(),
            error: run.error.clone(),
            parent_run_id: run.parent_run_id.map(|id| id.to_string()),
            start_time: run.start_time.to_rfc3339(),
            end_time: run.end_time.map(|t| t.to_rfc3339()),
            session_name: run.project_name.clone(),
            tags: run.tags.clone(),
            metadata: run.metadata.clone(),
        }
    }
}

/// Run update structure for PATCH requests
#[derive(Debug, Serialize)]
pub struct RunUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
}

impl From<&RunTree> for RunUpdate {
    fn from(run: &RunTree) -> Self {
        Self {
            outputs: run.outputs.clone(),
            error: run.error.clone(),
            end_time: run.end_time.map(|t| t.to_rfc3339()),
        }
    }
}
