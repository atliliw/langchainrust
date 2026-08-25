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

impl From<crate::RunType> for SpanKind {
    fn from(run_type: crate::RunType) -> Self {
        match run_type {
            crate::RunType::Llm => SpanKind::Llm,
            crate::RunType::Chain => SpanKind::Chain,
            crate::RunType::Tool => SpanKind::Tool,
            crate::RunType::Retriever => SpanKind::Retriever,
            // RunType variants without a dedicated SpanKind collapse to Custom
            crate::RunType::Embedding => SpanKind::Custom("embedding".to_string()),
            crate::RunType::Prompt => SpanKind::Custom("prompt".to_string()),
            crate::RunType::Parser => SpanKind::Custom("parser".to_string()),
        }
    }
}

/// Token usage recorded in a span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpanTokenUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens in the completion.
    pub completion_tokens: usize,
    /// Total number of tokens.
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

    // --- OTel GenAI SemConv fields ---
    /// gen_ai.system: The LLM provider name (e.g., "openai", "anthropic")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_system: Option<String>,
    /// gen_ai.request.model: The model requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_request_model: Option<String>,
    /// gen_ai.response.model: The actual model used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_response_model: Option<String>,
    /// gen_ai.response.finish_reason: Why the model stopped generating
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_finish_reason: Option<String>,
    /// gen_ai.request.max_tokens: Maximum tokens requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_request_max_tokens: Option<u64>,
    /// gen_ai.request.temperature: Temperature parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_request_temperature: Option<f64>,
    /// gen_ai.operation.name: The operation name (chat, completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_operation_name: Option<String>,
    /// gen_ai.tool.name: The tool name (for tool spans)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_ai_tool_name: Option<String>,
}

/// A node in the trace tree (span + children).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceNode {
    /// The span itself.
    pub span: TraceSpan,
    /// Child nodes of this span.
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
        gen_ai_system: None,
        gen_ai_request_model: None,
        gen_ai_response_model: None,
        gen_ai_finish_reason: None,
        gen_ai_request_max_tokens: None,
        gen_ai_request_temperature: None,
        gen_ai_operation_name: None,
        gen_ai_tool_name: None,
    }
}
