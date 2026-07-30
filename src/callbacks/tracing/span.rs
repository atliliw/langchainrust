use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Unique span identifier.
pub type SpanId = String;

/// Kind of span for categorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    /// LLM inference call
    Llm,
    /// Chain execution
    Chain,
    /// Tool invocation
    Tool,
    /// Retriever query
    Retriever,
    /// Agent execution
    Agent,
    /// Custom span kind
    Custom(String),
}

impl std::fmt::Display for SpanKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpanKind::Llm => write!(f, "llm"),
            SpanKind::Chain => write!(f, "chain"),
            SpanKind::Tool => write!(f, "tool"),
            SpanKind::Retriever => write!(f, "retriever"),
            SpanKind::Agent => write!(f, "agent"),
            SpanKind::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

/// Token usage recorded in a span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpanTokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Status of a span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    /// Span completed successfully
    Ok,
    /// Span ended with an error
    Error(String),
}

/// A single trace span with parent-child relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Unique span identifier
    pub id: SpanId,
    /// Parent span ID (None for root spans)
    pub parent_id: Option<SpanId>,
    /// Human-readable span name
    pub name: String,
    /// Span category
    pub kind: SpanKind,
    /// ISO 8601 start time
    pub start_time: Option<String>,
    /// ISO 8601 end time
    pub end_time: Option<String>,
    /// Token usage (for LLM spans)
    pub tokens: Option<SpanTokenUsage>,
    /// Estimated cost in USD
    pub cost: Option<f64>,
    /// Measured latency in milliseconds
    pub latency_ms: Option<u64>,
    /// Arbitrary key-value metadata
    pub metadata: serde_json::Value,
    /// Span completion status
    pub status: SpanStatus,
}

/// A node in the trace tree (span + children).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceNode {
    pub span: TraceSpan,
    pub children: Vec<TraceNode>,
}

pub(crate) fn build_tree(root: &TraceSpan, all_spans: &[TraceSpan]) -> TraceNode {
    let children: Vec<TraceNode> = all_spans
        .iter()
        .filter(|s| s.parent_id.as_deref() == Some(root.id.as_str()))
        .map(|child| build_tree(child, all_spans))
        .collect();

    TraceNode {
        span: root.clone(),
        children,
    }
}

/// Helper to create a new span with common defaults.
pub(crate) fn make_span(
    id: String,
    parent_id: Option<SpanId>,
    name: &str,
    kind: SpanKind,
) -> TraceSpan {
    TraceSpan {
        id,
        parent_id,
        name: name.to_string(),
        kind,
        start_time: Some(Utc::now().to_rfc3339()),
        end_time: None,
        tokens: None,
        cost: None,
        latency_ms: None,
        metadata: serde_json::Value::Object(serde_json::Map::new()),
        status: SpanStatus::Ok,
    }
}
