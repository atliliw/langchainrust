// lc-observability/src/otlp.rs
//! OpenTelemetry (OTLP) sink (feature `otlp`).
//!
//! Pipes each framework [`ObsEvent`] into an OTel span carrying the GenAI
//! semantic-convention attributes, then hands it to a batch exporter that ships
//! it to a collector over OTLP/HTTP-JSON. One event becomes exactly one span:
//!
//! - `TokenUsage` → an `llm` span with `gen_ai.usage.input_tokens` /
//!   `gen_ai.usage.output_tokens`
//! - `Cost` → a `cost` span with provider / `gen_ai.request.model` / usage plus
//!   the priced `langchainrust.cost_usd`
//! - `AgentMetrics` → an `agent` span with per-run counters and duration
//!
//! The attribute names are the workspace's shared GenAI semconv spelling (the
//! same `gen_ai.*` keys used by `lc-callbacks`), kept as local constants so this
//! crate needs no dependency on `lc-callbacks`.
//!
//! # Trade-off
//!
//! `export` only records the span; eventual delivery is the batch exporter's
//! job on its own Tokio runtime. When the collector is unreachable the exporter
//! logs the failure internally and `export` still returns `Ok` — the framework
//! only ever `warn`s on `Err` and sinks never interrupt the main flow, so a
//! down collector is absorbed, exactly like the `jsonl` / `mongo` sinks.
//!
//! `OtlpSink::install` must be called from within a Tokio runtime (the batch
//! exporter spawns its background task there).

use lc_core::observability::{AgentMetrics, MetricsSink, ObsError, ObsEvent};
use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::{Tracer, TracerProvider};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

const DEFAULT_ENDPOINT: &str = "http://localhost:4318";
const DEFAULT_SERVICE_NAME: &str = "langchainrust";
const DEFAULT_TIMEOUT_SECS: u64 = 10;
const INSTRUMENTATION_NAME: &str = "lc-observability";

// GenAI semconv keys — same spellings as `lc-callbacks::semconv`.
const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
const TRACE_ID_ATTR: &str = "langchainrust.trace_id";
// Framework-namespaced extensions (no standard GenAI attribute exists yet for
// price or agent counters, so they live under `langchainrust.*`).
const COST_USD_ATTR: &str = "langchainrust.cost_usd";
const COST_SCOPE_ATTR: &str = "langchainrust.scope";
const AGENT_LLM_CALLS_ATTR: &str = "langchainrust.agent.llm_calls";
const AGENT_CACHE_HITS_ATTR: &str = "langchainrust.agent.cache_hits";
const AGENT_TOOL_CALLS_ATTR: &str = "langchainrust.agent.tool_calls";
const AGENT_COMPACTIONS_ATTR: &str = "langchainrust.agent.compactions";
const AGENT_TOTAL_TOKENS_ATTR: &str = "langchainrust.agent.total_tokens";
const AGENT_DURATION_MS_ATTR: &str = "langchainrust.agent.duration_ms";

/// Configuration for the OTLP sink. [`Default`] reads the standard
/// `OTEL_EXPORTER_OTLP_*` / `OTEL_SERVICE_NAME` environment variables.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OtlpConfig {
    /// Collector base URL, e.g. `http://localhost:4318`.
    pub endpoint: String,
    /// Resource `service.name`.
    pub service_name: String,
    /// Extra HTTP headers sent with every export request.
    pub headers: HashMap<String, String>,
    /// Per-request export timeout.
    pub timeout: Duration,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            endpoint: env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string()),
            service_name: env::var("OTEL_SERVICE_NAME")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_string()),
            headers: env::var("OTEL_EXPORTER_OTLP_HEADERS")
                .map(|raw| parse_headers(&raw))
                .unwrap_or_default(),
            timeout: env::var("OTEL_EXPORTER_OTLP_TIMEOUT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS)),
        }
    }
}

/// Sinks each [`ObsEvent`] as one OTLP span with GenAI semconv attributes.
///
/// Holds a shared [`TracerProvider`] (clonable — every clone shares the same
/// batch exporter); the config (endpoint/headers) is consumed at `install` and
/// never stored, so [`Debug`] prints nothing sensitive.
#[derive(Clone)]
pub struct OtlpSink {
    provider: TracerProvider,
}

impl std::fmt::Debug for OtlpSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpSink").finish_non_exhaustive()
    }
}

impl OtlpSink {
    /// Builds an OTLP/HTTP-JSON tracer provider from `config` and returns a sink
    /// that emits through it. May fail only if the exporter cannot be built; a
    /// down collector is not an error here (batch export failures are logged
    /// internally by the SDK). Must be called from within a Tokio runtime.
    pub fn install(config: OtlpConfig) -> Result<Self, ObsError> {
        let exporter: SpanExporter = SpanExporter::builder()
            .with_http()
            .with_endpoint(config.endpoint)
            // Wire format is fixed to HTTP/JSON (no native TLS toolchain needed).
            .with_protocol(Protocol::HttpJson)
            .with_timeout(config.timeout)
            .with_headers(config.headers)
            .build()
            .map_err(|e| ObsError::Transport(format!("otlp: build exporter: {e}")))?;

        let resource =
            Resource::new([KeyValue::new("service.name", config.service_name)]);

        let provider = TracerProvider::builder()
            .with_batch_exporter(exporter, runtime::Tokio)
            .with_resource(resource)
            .build();

        Ok(Self { provider })
    }

    /// The instrumentation tracer used by [`MetricsSink::export`].
    fn tracer(&self) -> Tracer {
        self.provider.tracer(INSTRUMENTATION_NAME)
    }
}

#[async_trait::async_trait]
impl MetricsSink for OtlpSink {
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
        let mut span = self.tracer().start(span_name_for(event));
        for kv in span_attributes_for(event) {
            span.set_attribute(kv);
        }
        // Recording is fire-and-forget: the batch processor ships it and, on a
        // down collector, logs the export failure internally. Never panics.
        span.end();
        Ok(())
    }
}

/// Semconv span name for an event: `llm` / `cost` / `agent`.
pub fn span_name_for(event: &ObsEvent) -> String {
    match event {
        ObsEvent::TokenUsage(_) => "llm".to_string(),
        ObsEvent::Cost(_) => "cost".to_string(),
        ObsEvent::AgentMetrics(_) => "agent".to_string(),
    }
}

/// GenAI semconv attributes for an event (pure mapping, testable offline).
pub fn span_attributes_for(event: &ObsEvent) -> Vec<KeyValue> {
    match event {
        ObsEvent::TokenUsage(usage) => usage_attributes(usage.prompt_tokens, usage.completion_tokens),
        ObsEvent::Cost(cost) => {
            let mut attrs =
                usage_attributes(cost.prompt_tokens, cost.completion_tokens).into_iter()
                    .chain(Some(KeyValue::new(GEN_AI_REQUEST_MODEL, cost.model.clone())))
                    .collect::<Vec<_>>();
            if let Some(provider) = &cost.provider {
                attrs.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider.clone()));
            }
            if let Some(scope) = &cost.scope {
                attrs.push(KeyValue::new(COST_SCOPE_ATTR, scope.clone()));
            }
            attrs.push(KeyValue::new(COST_USD_ATTR, cost.cost_usd));
            attrs
        }
        ObsEvent::AgentMetrics(metrics) => agent_attributes(metrics),
    }
}

/// Usage + operation attributes shared by the token-usage and cost spans.
fn usage_attributes(prompt_tokens: usize, completion_tokens: usize) -> Vec<KeyValue> {
    vec![
        KeyValue::new(GEN_AI_OPERATION_NAME, "chat"),
        KeyValue::new(GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens as i64),
        KeyValue::new(GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens as i64),
    ]
}

/// Per-run agent counters as span attributes.
fn agent_attributes(metrics: &AgentMetrics) -> Vec<KeyValue> {
    let mut attrs = vec![
        KeyValue::new(GEN_AI_OPERATION_NAME, "invoke_agent"),
        KeyValue::new(AGENT_LLM_CALLS_ATTR, metrics.llm_calls as i64),
        KeyValue::new(AGENT_CACHE_HITS_ATTR, metrics.cache_hits as i64),
        KeyValue::new(AGENT_TOOL_CALLS_ATTR, metrics.tool_calls as i64),
        KeyValue::new(AGENT_COMPACTIONS_ATTR, metrics.compactions as i64),
        KeyValue::new(AGENT_DURATION_MS_ATTR, metrics.duration.as_millis() as i64),
    ];
    if let Some(trace_id) = &metrics.trace_id {
        attrs.push(KeyValue::new(TRACE_ID_ATTR, trace_id.clone()));
    }
    if let Some(tokens) = metrics.total_tokens {
        attrs.push(KeyValue::new(AGENT_TOTAL_TOKENS_ATTR, tokens as i64));
    }
    attrs
}

/// Parses the `OTEL_EXPORTER_OTLP_HEADERS` value: comma-separated `k=v` pairs
/// with percent-decoding. Malformed pairs (no `=`) are skipped.
fn parse_headers(raw: &str) -> HashMap<String, String> {
    raw.split(',')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?.trim();
            let value = parts.next()?;
            if key.is_empty() {
                return None;
            }
            Some((key.to_string(), percent_decode(value.trim())))
        })
        .collect()
}

/// Minimal RFC 3986 percent-decoding (UTF-8 aware); leaves non-`%` bytes
/// untouched. Invalid escape sequences are kept verbatim.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::language_models::TokenUsage;
    use lc_core::observability::CostEvent;
    use opentelemetry::Value;

    fn token_event() -> ObsEvent {
        ObsEvent::TokenUsage(TokenUsage {
            prompt_tokens: 11,
            completion_tokens: 3,
            total_tokens: 14,
        })
    }

    fn attr<'a>(attrs: &'a [KeyValue], key: &str) -> Option<&'a KeyValue> {
        attrs.iter().find(|kv| kv.key.as_str() == key)
    }

    #[test]
    fn span_name_maps_each_event_kind() {
        assert_eq!(span_name_for(&token_event()), "llm");
        let cost = ObsEvent::Cost(CostEvent {
            scope: None,
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 1000,
            cost_usd: 0.75,
        });
        assert_eq!(span_name_for(&cost), "cost");
        assert_eq!(
            span_name_for(&ObsEvent::AgentMetrics(AgentMetrics::default())),
            "agent"
        );
    }

    #[test]
    fn token_usage_maps_usage_semconv() {
        let attrs = span_attributes_for(&token_event());
        assert_eq!(attr(&attrs, GEN_AI_OPERATION_NAME).unwrap().value, Value::from("chat"));
        assert_eq!(
            attr(&attrs, GEN_AI_USAGE_INPUT_TOKENS).unwrap().value,
            Value::from(11i64)
        );
        assert_eq!(
            attr(&attrs, GEN_AI_USAGE_OUTPUT_TOKENS).unwrap().value,
            Value::from(3i64)
        );
        // No provider / model on a bare token-usage event.
        assert!(attr(&attrs, GEN_AI_PROVIDER_NAME).is_none());
        assert!(attr(&attrs, GEN_AI_REQUEST_MODEL).is_none());
    }

    #[test]
    fn cost_maps_provider_model_usage_and_price() {
        let attrs = span_attributes_for(&ObsEvent::Cost(CostEvent {
            scope: Some("run-7".to_string()),
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 500,
            cost_usd: 0.375,
        }));
        assert_eq!(
            attr(&attrs, GEN_AI_PROVIDER_NAME).unwrap().value,
            Value::from("openai")
        );
        assert_eq!(
            attr(&attrs, GEN_AI_REQUEST_MODEL).unwrap().value,
            Value::from("gpt-4o-mini")
        );
        assert_eq!(
            attr(&attrs, GEN_AI_USAGE_INPUT_TOKENS).unwrap().value,
            Value::from(1000i64)
        );
        assert_eq!(
            attr(&attrs, GEN_AI_USAGE_OUTPUT_TOKENS).unwrap().value,
            Value::from(500i64)
        );
        assert!(matches!(
            attr(&attrs, COST_USD_ATTR).unwrap().value,
            Value::F64(v) if v == 0.375
        ));
        assert_eq!(
            attr(&attrs, COST_SCOPE_ATTR).unwrap().value,
            Value::from("run-7")
        );
    }

    #[test]
    fn cost_with_undeclared_provider_omits_provider_attr() {
        let attrs = span_attributes_for(&ObsEvent::Cost(CostEvent {
            scope: None,
            provider: None,
            model: "local-model".to_string(),
            prompt_tokens: 1,
            completion_tokens: 1,
            cost_usd: 0.0,
        }));
        assert!(attr(&attrs, GEN_AI_PROVIDER_NAME).is_none());
    }

    #[test]
    fn agent_metrics_maps_counters_and_duration() {
        let attrs = span_attributes_for(&ObsEvent::AgentMetrics(AgentMetrics {
            trace_id: Some("trace-x".to_string()),
            llm_calls: 2,
            cache_hits: 1,
            tool_calls: 4,
            compactions: 1,
            total_tokens: Some(30),
            duration: Duration::from_millis(1500),
        }));
        assert_eq!(
            attr(&attrs, GEN_AI_OPERATION_NAME).unwrap().value,
            Value::from("invoke_agent")
        );
        assert_eq!(attr(&attrs, TRACE_ID_ATTR).unwrap().value, Value::from("trace-x"));
        assert_eq!(attr(&attrs, AGENT_LLM_CALLS_ATTR).unwrap().value, Value::from(2i64));
        assert_eq!(attr(&attrs, AGENT_CACHE_HITS_ATTR).unwrap().value, Value::from(1i64));
        assert_eq!(attr(&attrs, AGENT_TOOL_CALLS_ATTR).unwrap().value, Value::from(4i64));
        assert_eq!(attr(&attrs, AGENT_COMPACTIONS_ATTR).unwrap().value, Value::from(1i64));
        assert_eq!(attr(&attrs, AGENT_TOTAL_TOKENS_ATTR).unwrap().value, Value::from(30i64));
        assert_eq!(
            attr(&attrs, AGENT_DURATION_MS_ATTR).unwrap().value,
            Value::from(1500i64)
        );
    }

    #[test]
    fn agent_metrics_without_tokens_omits_total_tokens_attr() {
        let attrs = span_attributes_for(&ObsEvent::AgentMetrics(AgentMetrics {
            trace_id: None,
            ..Default::default()
        }));
        assert!(attr(&attrs, TRACE_ID_ATTR).is_none());
        assert!(attr(&attrs, AGENT_TOTAL_TOKENS_ATTR).is_none());
    }

    #[test]
    fn parses_comma_separated_headers() {
        let map = parse_headers("authorization=Bearer abc,x-custom=v");
        assert_eq!(map.get("authorization").unwrap(), "Bearer abc");
        assert_eq!(map.get("x-custom").unwrap(), "v");
    }

    #[test]
    fn decodes_percent_encoded_values() {
        assert_eq!(percent_decode("a%2Cb"), "a,b");
        assert_eq!(percent_decode("%41%42%43"), "ABC");
    }

    #[test]
    fn default_config_falls_back_when_env_absent() {
        env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        env::remove_var("OTEL_EXPORTER_OTLP_TIMEOUT");
        let config = OtlpConfig::default();
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }

    // A multi-threaded runtime is required: the batch exporter's worker is
    // spawned on it, and the synchronous drop above must not stall that worker
    // (on a current-thread runtime the shutdown would deadlock).
    #[tokio::test(flavor = "multi_thread")]
    async fn export_records_event_without_network() {
        // A down collector is absorbed: export() records the span into the batch
        // queue and returns Ok; delivery failures are logged internally.
        env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        env::remove_var("OTEL_EXPORTER_OTLP_TIMEOUT");
        let sink = OtlpSink::install(OtlpConfig::default()).expect("install without collector");
        sink.export(&token_event()).await.expect("record does not fail");
        sink.export(&ObsEvent::AgentMetrics(AgentMetrics::default()))
            .await
            .expect("record does not fail");
        drop(sink); // shutdown flushes (connection-refused is logged, not panicky)
    }
}