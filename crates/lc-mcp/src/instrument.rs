//! OpenTelemetry instrumentation for MCP client `tools/call` (T10, v0.23).
//!
//! Follows the 2026 MCP GenAI semantic-convention draft (`mcp.md`,
//! Development stability): one CLIENT span per `tools/call`, named
//! `tools/call {tool}`, carrying `mcp.method.name` (Required),
//! `gen_ai.operation.name="execute_tool"`, `gen_ai.tool.name`,
//! `mcp.protocol.version`, optional `mcp.session.id` and `network.transport`
//! (`"tcp"` for the HTTP tracks, `"pipe"` for stdio). JSON-RPC failures record
//! `error.type` — the one Stable attribute in the draft — as the decimal
//! JSON-RPC error code string (e.g. `"-32603"`).
//!
//! lc-mcp cannot depend on lc-callbacks, so the attribute names are mirrored
//! here rather than imported (the mirror is documented in the
//! `lc_callbacks::semconv` module header); if the registry moves a name,
//! update both places.
//!
//! Opt-In parity with `OtelHandler`: tool-call arguments/results can contain
//! secrets or PII, so they are recorded only after an explicit
//! [`McpInstrumentation::with_tool_payloads`] opt-in (call it with `true`), and
//! truncated to `max_payload_chars` (default 2048).

use std::future::Future;

use opentelemetry::global::BoxedTracer;
use opentelemetry::trace::{FutureExt, SpanBuilder, SpanKind, Status, TraceContextExt, Tracer};
use opentelemetry::{Context, KeyValue};
use serde_json::Value as JsonValue;

use crate::protocol::MCPError;
use crate::types::MCPToolResult;

// --- MCP semconv names (mirror of upstream mcp.md) -------------------------

/// `mcp.method.name` (Required): the JSON-RPC method (`tools/call`).
pub(crate) const MCP_METHOD_NAME: &str = "mcp.method.name";
/// `mcp.protocol.version`: negotiated MCP protocol version.
pub(crate) const MCP_PROTOCOL_VERSION: &str = "mcp.protocol.version";
/// `mcp.session.id`: server-assigned session id (stateful HTTP sessions only).
pub(crate) const MCP_SESSION_ID: &str = "mcp.session.id";
/// `network.transport` value for the HTTP tracks (Streamable / stateless).
pub(crate) const NETWORK_TRANSPORT_TCP: &str = "tcp";
/// `network.transport` value for the stdio track.
pub(crate) const NETWORK_TRANSPORT_PIPE: &str = "pipe";
/// OTel general-semconv `network.transport` attribute name.
pub(crate) const NETWORK_TRANSPORT: &str = "network.transport";

// --- GenAI semconv names (mirror of lc_callbacks::semconv) -----------------

const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
/// Stable OTel general-semconv error attribute.
const ERROR_TYPE: &str = "error.type";
/// `error.type` for a JSON-RPC-200 tool result carrying `isError: true`.
const TOOL_ERROR_TYPE: &str = "tool_error";

/// Default cap on recorded payload strings, matching `OtelHandler`.
const DEFAULT_MAX_PAYLOAD_CHARS: usize = 2048;

impl std::fmt::Debug for McpInstrumentation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpInstrumentation")
            .field("record_payloads", &self.record_payloads)
            .field("max_payload_chars", &self.max_payload_chars)
            .finish_non_exhaustive()
    }
}

/// Emits one MCP `tools/call` client span per invocation.
///
/// Shared per client through an `Arc` (`StreamableMcpClient::with_instrumentation`
/// et al.); construct from a configured global tracer provider with
/// [`McpInstrumentation::from_global`].
pub struct McpInstrumentation {
    tracer: BoxedTracer,
    record_payloads: bool,
    max_payload_chars: usize,
}

impl McpInstrumentation {
    /// Wraps an explicit tracer (typically `provider.tracer("…")` boxed via
    /// `BoxedTracer::new`).
    pub fn new(tracer: BoxedTracer) -> Self {
        Self {
            tracer,
            record_payloads: false,
            max_payload_chars: DEFAULT_MAX_PAYLOAD_CHARS,
        }
    }

    /// Uses the global tracer provider with the crate tracer name.
    pub fn from_global() -> Self {
        Self::new(opentelemetry::global::tracer("langchainrust-mcp"))
    }

    /// Opt in to recording `gen_ai.tool.call.arguments` /
    /// `gen_ai.tool.call.result`. Defaults to off (payloads may contain
    /// secrets/PII).
    pub fn with_tool_payloads(mut self, enabled: bool) -> Self {
        self.record_payloads = enabled;
        self
    }

    /// Overrides the truncation cap for recorded payload strings.
    pub fn with_max_payload_chars(mut self, max_chars: usize) -> Self {
        self.max_payload_chars = max_chars;
        self
    }

    /// Runs `call` inside a `tools/call {tool}` CLIENT span and finalizes the
    /// span attributes/status from the outcome.
    pub(crate) async fn record_tool_call<F>(
        &self,
        tool: &str,
        arguments: &JsonValue,
        network_transport: &str,
        protocol_version: &str,
        session_id: Option<&str>,
        call: F,
    ) -> Result<MCPToolResult, MCPError>
    where
        F: Future<Output = Result<MCPToolResult, MCPError>> + Send,
    {
        let mut attributes = vec![
            KeyValue::new(MCP_METHOD_NAME, "tools/call"),
            KeyValue::new(GEN_AI_OPERATION_NAME, "execute_tool"),
            KeyValue::new(GEN_AI_TOOL_NAME, tool.to_string()),
            KeyValue::new(MCP_PROTOCOL_VERSION, protocol_version.to_string()),
            KeyValue::new(NETWORK_TRANSPORT, network_transport.to_string()),
        ];
        if let Some(sid) = session_id {
            attributes.push(KeyValue::new(MCP_SESSION_ID, sid.to_string()));
        }
        if self.record_payloads {
            attributes.push(KeyValue::new(
                GEN_AI_TOOL_CALL_ARGUMENTS,
                truncate_json(arguments, self.max_payload_chars),
            ));
        }

        let span = self.tracer.build_with_context(
            SpanBuilder::from_name(format!("tools/call {tool}"))
                .with_kind(SpanKind::Client)
                .with_attributes(attributes),
            &Context::current(),
        );
        let cx = Context::current_with_span(span);

        let result = call.with_context(cx.clone()).await;

        let span_ref = cx.span();
        match &result {
            Ok(tool_result) => {
                if self.record_payloads {
                    let rendered = serde_json::to_string(tool_result).unwrap_or_default();
                    span_ref.set_attribute(KeyValue::new(
                        GEN_AI_TOOL_CALL_RESULT,
                        truncate(&rendered, self.max_payload_chars),
                    ));
                }
                // MCP distinguishes transport success from a tool-reported
                // failure (JSON-RPC 200 with `isError: true`).
                if tool_result.is_error {
                    span_ref.set_attribute(KeyValue::new(ERROR_TYPE, TOOL_ERROR_TYPE));
                    span_ref.set_status(Status::error("MCP tool returned isError=true"));
                }
            }
            Err(e) => {
                // The draft pins error.type to the JSON-RPC error code string.
                span_ref.set_attribute(KeyValue::new(ERROR_TYPE, e.code.to_string()));
                span_ref.set_status(Status::error(e.message.clone()));
            }
        }
        span_ref.end();
        result
    }
}

/// Renders a JSON value compactly and truncates it (chars, not bytes).
fn truncate_json(value: &JsonValue, max_chars: usize) -> String {
    let rendered = serde_json::to_string(value).unwrap_or_default();
    truncate(&rendered, max_chars)
}

/// Unicode-safe truncation; appends an ellipsis when cut.
fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        None => s.to_string(),
        Some((idx, _)) => format!("{}…", &s[..idx]),
    }
}

#[cfg(all(test, feature = "opentelemetry"))]
mod tests {
    use super::*;
    use crate::types::MCPContent;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::testing::trace::InMemorySpanExporterBuilder;
    use opentelemetry_sdk::trace::{SimpleSpanProcessor, TracerProvider};
    use serde_json::json;

    fn instrumentation(
        record_payloads: bool,
    ) -> (
        McpInstrumentation,
        opentelemetry_sdk::testing::trace::InMemorySpanExporter,
    ) {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
            .build();
        let tracer = BoxedTracer::new(Box::new(provider.tracer("test-mcp")));
        (
            McpInstrumentation::new(tracer).with_tool_payloads(record_payloads),
            exporter,
        )
    }

    fn attr_str(span: &opentelemetry_sdk::export::trace::SpanData, key: &str) -> Option<String> {
        span.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.as_str().into_owned())
    }

    fn ok_result() -> MCPToolResult {
        MCPToolResult {
            content: vec![MCPContent::Text {
                text: "pong".into(),
            }],
            is_error: false,
        }
    }

    #[tokio::test]
    async fn success_span_carries_mcp_semconv_attributes() {
        let (instr, exporter) = instrumentation(false);
        let result = instr
            .record_tool_call(
                "echo",
                &json!({"msg": "hi"}),
                NETWORK_TRANSPORT_TCP,
                "2025-06-18",
                Some("sess-1"),
                async { Ok(ok_result()) },
            )
            .await;
        assert!(result.is_ok());

        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name, "tools/call echo");
        assert_eq!(span.span_kind, opentelemetry::trace::SpanKind::Client);
        assert_eq!(
            attr_str(span, MCP_METHOD_NAME).as_deref(),
            Some("tools/call")
        );
        assert_eq!(
            attr_str(span, GEN_AI_OPERATION_NAME).as_deref(),
            Some("execute_tool")
        );
        assert_eq!(attr_str(span, GEN_AI_TOOL_NAME).as_deref(), Some("echo"));
        assert_eq!(
            attr_str(span, MCP_PROTOCOL_VERSION).as_deref(),
            Some("2025-06-18")
        );
        assert_eq!(attr_str(span, MCP_SESSION_ID).as_deref(), Some("sess-1"));
        assert_eq!(attr_str(span, "network.transport").as_deref(), Some("tcp"));
        // Payloads are opt-in: absent by default.
        assert!(attr_str(span, GEN_AI_TOOL_CALL_ARGUMENTS).is_none());
        assert!(attr_str(span, GEN_AI_TOOL_CALL_RESULT).is_none());
        assert!(attr_str(span, ERROR_TYPE).is_none());
    }

    #[tokio::test]
    async fn stdio_track_uses_pipe_transport_without_session() {
        let (instr, exporter) = instrumentation(false);
        instr
            .record_tool_call(
                "bash",
                &json!({}),
                NETWORK_TRANSPORT_PIPE,
                "2025-03-26",
                None,
                async { Ok(ok_result()) },
            )
            .await
            .unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(attr_str(span, "network.transport").as_deref(), Some("pipe"));
        assert!(attr_str(span, MCP_SESSION_ID).is_none());
    }

    #[tokio::test]
    async fn jsonrpc_failure_records_error_type_as_code_string() {
        let (instr, exporter) = instrumentation(false);
        let result = instr
            .record_tool_call(
                "echo",
                &json!({}),
                NETWORK_TRANSPORT_TCP,
                "2025-06-18",
                None,
                async { Err(MCPError::new(-32603, "internal boom")) },
            )
            .await;
        assert!(result.is_err());

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(attr_str(span, ERROR_TYPE).as_deref(), Some("-32603"));
        assert!(matches!(
            span.status,
            opentelemetry::trace::Status::Error { .. }
        ));
    }

    #[tokio::test]
    async fn tool_is_error_result_marks_span_errored() {
        let (instr, exporter) = instrumentation(false);
        instr
            .record_tool_call(
                "echo",
                &json!({}),
                NETWORK_TRANSPORT_TCP,
                "2025-06-18",
                None,
                async {
                    Ok(MCPToolResult {
                        content: vec![],
                        is_error: true,
                    })
                },
            )
            .await
            .unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        assert_eq!(attr_str(span, ERROR_TYPE).as_deref(), Some("tool_error"));
    }

    #[tokio::test]
    async fn payloads_are_recorded_only_when_opted_in() {
        let (instr, exporter) = instrumentation(true);
        instr
            .record_tool_call(
                "echo",
                &json!({"msg": "1234567890"}),
                NETWORK_TRANSPORT_TCP,
                "2025-06-18",
                None,
                async { Ok(ok_result()) },
            )
            .await
            .unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        let arguments = attr_str(span, GEN_AI_TOOL_CALL_ARGUMENTS).expect("arguments recorded");
        assert!(arguments.contains("1234567890"));
        let result = attr_str(span, GEN_AI_TOOL_CALL_RESULT).expect("result recorded");
        assert!(result.contains("pong"));
    }

    #[tokio::test]
    async fn recorded_payloads_are_truncated() {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = TracerProvider::builder()
            .with_span_processor(SimpleSpanProcessor::new(Box::new(exporter.clone())))
            .build();
        let tracer = BoxedTracer::new(Box::new(provider.tracer("test-mcp")));
        let instr = McpInstrumentation::new(tracer)
            .with_tool_payloads(true)
            .with_max_payload_chars(4);
        instr
            .record_tool_call(
                "echo",
                &json!({"msg": "1234567890"}),
                NETWORK_TRANSPORT_TCP,
                "2025-06-18",
                None,
                async { Ok(ok_result()) },
            )
            .await
            .unwrap();

        let spans = exporter.get_finished_spans().unwrap();
        let span = &spans[0];
        let arguments = attr_str(span, GEN_AI_TOOL_CALL_ARGUMENTS).unwrap();
        assert_eq!(arguments.chars().count(), 5); // 4 payload chars + ellipsis
        assert!(arguments.ends_with('…'));
    }
}
