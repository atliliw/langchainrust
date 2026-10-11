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

use lc_core::observability::{AgentMetrics, MetricsSink, ObsError, ObsEvent, RunEvent};
use opentelemetry::metrics::{Counter, Meter, MeterProvider};
use opentelemetry::trace::{Span as _, Status, Tracer as _, TracerProvider as _};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{
    MetricExporter, Protocol, SpanExporter, WithExportConfig, WithHttpConfig,
};
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::{Tracer, TracerProvider};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

const DEFAULT_ENDPOINT: &str = "http://localhost:4318";
const DEFAULT_SERVICE_NAME: &str = "langchainrust";
const DEFAULT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_METRIC_EXPORT_INTERVAL_SECS: u64 = 60;
const INSTRUMENTATION_NAME: &str = "lc-observability";

// GenAI semconv keys — same spellings as `lc-callbacks::semconv`.
const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
const TRACE_ID_ATTR: &str = "langchainrust.trace_id";
// GenAI token-usage counter (T2 §5: cost-aggregation KPI). Name and unit follow
// the OTel GenAI semantic convention so Prometheus/collector dashboards key off
// the standard metric; the value is the per-event token total.
const GEN_AI_METRIC_TOKEN_USAGE: &str = "gen_ai.client.token.usage";
const GEN_AI_METRIC_TOKEN_USAGE_UNIT: &str = "{token}";
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
    /// Enable the `gen_ai.client.token.usage` Meter pipeline (T2 §5). When `true`,
    /// [`OtlpSink::install`] also builds an OTLP metric exporter and exposes a
    /// token-usage counter on the same endpoint/headers. Default off — keeps the
    /// sink trace-only unless the caller opts into metrics.
    pub metrics: bool,
    /// Metric export interval for the token-usage counter. Only read when
    /// [`OtlpConfig::metrics`] is enabled.
    pub metric_export_interval: Duration,
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
            metrics: false,
            metric_export_interval: env::var("OTEL_METRIC_EXPORT_INTERVAL")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(DEFAULT_METRIC_EXPORT_INTERVAL_SECS)),
        }
    }
}

/// Sinks each [`ObsEvent`] as one OTLP span with GenAI semconv attributes.
///
/// Holds a shared [`TracerProvider`] (clonable — every clone shares the same
/// batch exporter); the config (endpoint/headers) is consumed at `install` and
/// never stored, so [`Debug`] prints nothing sensitive.
///
/// # Known limitation — cross-event trace lineage (reported §3.1/§3.2/§3.3)
///
/// [`ObsEvent::AgentMetrics`], [`ObsEvent::Cost`], and [`ObsEvent::RunError`] all
/// carry a run `trace_id`, so the cost and failure spans join the agent span into one
/// OTel trace. [`ObsEvent::TokenUsage`] (the highest-frequency `llm` span) still has no
/// trace linkage and no `model`, because it shares the workspace-wide
/// `lc_core::language_models::TokenUsage` type that embeddings/rag also emit — extending
/// it is a broad cross-crate API change deliberately deferred.
#[derive(Clone)]
pub struct OtlpSink {
    provider: TracerProvider,
    /// Optional token-usage Meter pipeline (T2 §5). `None` keeps the sink
    /// trace-only; `Some` carries the `gen_ai.client.token.usage` counter.
    meter: Option<SinkMeter>,
}

/// Clonable handle to the token-usage meter + its counter.
#[derive(Clone)]
struct SinkMeter {
    counter: Counter<u64>,
    /// Held so the provider's background metric reader stays alive for the sink's
    /// lifetime and is shut down when the last clone drops.
    _provider: SdkMeterProvider,
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
        // §4: the batch exporter schedules on a Tokio runtime (`runtime::Tokio`), so
        // calling this from a non-Tokio thread deadlocks or panics. Probe up front and
        // return a clear error instead of a confusing panic deep in the SDK.
        if tokio::runtime::Handle::try_current().is_err() {
            return Err(ObsError::Transport(
                "otlp: OtlpSink::install requires a Tokio runtime (call it from within \
                 #[tokio::main] / tokio::spawn); no runtime was active"
                    .to_string(),
            ));
        }
        // 0.27 config is moved into the exporter builders, so `.clone()` the
        // fields the span exporter consumes; `config` stays whole for `build_meter`.
        let exporter: SpanExporter = SpanExporter::builder()
            .with_http()
            .with_endpoint(config.endpoint.clone())
            // Wire format is fixed to HTTP/JSON (no native TLS toolchain needed).
            .with_protocol(Protocol::HttpJson)
            .with_timeout(config.timeout)
            .with_headers(config.headers.clone())
            .build()
            .map_err(|e| ObsError::Transport(format!("otlp: build exporter: {e}")))?;

        // 0.27 `Resource` takes ownership; clone so `config` stays whole.
        let resource = Resource::new([KeyValue::new("service.name", config.service_name.clone())]);

        let provider = TracerProvider::builder()
            .with_batch_exporter(exporter, runtime::Tokio)
            .with_resource(resource.clone())
            .build();

        // T2 §5: optionally build the token-usage meter on the same endpoint. The
        // metric exporter and tracer share `config.endpoint`/`headers`/`timeout`.
        let meter = if config.metrics {
            Some(build_meter(&config, resource)?)
        } else {
            None
        };

        Ok(Self { provider, meter })
    }

    /// The instrumentation tracer used by [`MetricsSink::export`].
    fn tracer(&self) -> Tracer {
        self.provider.tracer(INSTRUMENTATION_NAME)
    }
}

/// Builds the token-usage Meter pipeline (T2 §5): an OTLP metric exporter on the
/// same endpoint/headers/timeout as the tracer and a `gen_ai.client.token.usage`
/// monotonic counter. The `SdkMeterProvider` owns a background periodic reader;
/// it is dropped (and shut down) with the sink.
fn build_meter(config: &OtlpConfig, resource: Resource) -> Result<SinkMeter, ObsError> {
    let exporter: MetricExporter = MetricExporter::builder()
        .with_http()
        .with_endpoint(config.endpoint.clone())
        .with_protocol(Protocol::HttpJson)
        .with_timeout(config.timeout)
        .with_headers(config.headers.clone())
        .build()
        .map_err(|e| ObsError::Transport(format!("otlp: build metric exporter: {e}")))?;

    let reader = PeriodicReader::builder(exporter, runtime::Tokio)
        .with_interval(config.metric_export_interval)
        .build();

    let provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();
    let meter: Meter = provider.meter(INSTRUMENTATION_NAME);
    let counter: Counter<u64> = meter
        .u64_counter(GEN_AI_METRIC_TOKEN_USAGE)
        .with_description("Total tokens passed to/returned from a GenAI model call")
        .with_unit(GEN_AI_METRIC_TOKEN_USAGE_UNIT)
        .build();
    Ok(SinkMeter {
        counter,
        _provider: provider,
    })
}

#[async_trait::async_trait]
impl MetricsSink for OtlpSink {
    async fn export(&self, event: &ObsEvent) -> Result<(), ObsError> {
        let mut span = self.tracer().start(span_name_for(event));
        for kv in span_attributes_for(event) {
            span.set_attribute(kv);
        }
        // A failed run is exported as an error span so roll-ups that only look
        // at failed spans still surface it (see §3.4).
        if let ObsEvent::RunError(run) = event {
            span.set_status(Status::error(run.error.clone()));
        }
        // Recording is fire-and-forget: the batch processor ships it and, on a
        // down collector, logs the export failure internally. Never panics.
        span.end();

        // T2 §5: bump the token-usage KPI when a meter is configured. Token-cloud
        // events (TokenUsage, Cost) add their total; agent/error events are not
        // token-bearing and leave the counter untouched.
        if let Some(meter) = &self.meter {
            if let Some((tokens, attrs)) = token_usage_record(event) {
                meter.counter.add(tokens, &attrs);
            }
        }
        Ok(())
    }
}

/// Yields the `(total_tokens, attributes)` recorded for a token-bearing event, or
/// `None` for events that carry no token count (AgentMetrics / RunError).
fn token_usage_record(event: &ObsEvent) -> Option<(u64, Vec<KeyValue>)> {
    match event {
        ObsEvent::TokenUsage(usage) => {
            let mut attrs = vec![KeyValue::new(GEN_AI_OPERATION_NAME, "chat")];
            attrs.push(KeyValue::new(GEN_AI_PROVIDER_NAME, "unknown"));
            Some((usage.total_tokens as u64, attrs))
        }
        ObsEvent::Cost(cost) => {
            let mut attrs = vec![
                KeyValue::new(GEN_AI_OPERATION_NAME, "chat"),
                KeyValue::new(GEN_AI_REQUEST_MODEL, cost.model.clone()),
            ];
            if let Some(provider) = &cost.provider {
                attrs.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider.clone()));
            }
            let tokens = (cost.prompt_tokens + cost.completion_tokens) as u64;
            Some((tokens, attrs))
        }
        _ => None,
    }
}

/// Semconv span name for an event, templated as `{op} {provider}` (T2 §4).
///
/// OTel GenAI splits the span name into an operation and the provider that served
/// it (`llm openai`, `cost anthropic`), which is what roll-ups key on. When the
/// event carries no provider the bare op is used (`llm`, `agent`, `agent.error`),
/// so events whose provider is unknown still produce a clean, stable name.
pub fn span_name_for(event: &ObsEvent) -> String {
    let (op, provider) = match event {
        ObsEvent::TokenUsage(_) => ("llm", None),
        ObsEvent::Cost(cost) => ("cost", cost.provider.as_deref()),
        ObsEvent::AgentMetrics(_) => ("agent", None),
        ObsEvent::RunError(run) => ("agent.error", run.provider.as_deref()),
    };
    template_span_name(op, provider)
}

/// Templates `{op} {provider}`, falling back to `{op}` when the provider is
/// absent or empty. Kept separate so the templating rule is testable on its own.
pub fn template_span_name(op: &str, provider: Option<&str>) -> String {
    match provider.filter(|p| !p.is_empty()) {
        Some(p) => format!("{op} {p}"),
        None => op.to_string(),
    }
}

/// GenAI semconv attributes for an event (pure mapping, testable offline).
pub fn span_attributes_for(event: &ObsEvent) -> Vec<KeyValue> {
    match event {
        ObsEvent::TokenUsage(usage) => {
            usage_attributes(usage.prompt_tokens, usage.completion_tokens)
        }
        ObsEvent::Cost(cost) => {
            let mut attrs = usage_attributes(cost.prompt_tokens, cost.completion_tokens)
                .into_iter()
                .chain(Some(KeyValue::new(
                    GEN_AI_REQUEST_MODEL,
                    cost.model.clone(),
                )))
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
        ObsEvent::RunError(run) => run_error_attributes(run),
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

/// A failed run as an error-span attribute set. Carries the same run
/// `trace_id` join key as `AgentMetrics`/`Cost` so the failure rolls up with the
/// run's other events, plus provider/model when the failing stage had them.
fn run_error_attributes(run: &RunEvent) -> Vec<KeyValue> {
    let mut attrs = vec![KeyValue::new(GEN_AI_OPERATION_NAME, "invoke_agent")];
    if let Some(trace_id) = &run.trace_id {
        attrs.push(KeyValue::new(TRACE_ID_ATTR, trace_id.clone()));
    }
    if let Some(provider) = &run.provider {
        attrs.push(KeyValue::new(GEN_AI_PROVIDER_NAME, provider.clone()));
    }
    if let Some(model) = &run.model {
        attrs.push(KeyValue::new(GEN_AI_REQUEST_MODEL, model.clone()));
    }
    attrs.push(KeyValue::new("error.message", run.error.clone()));
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
        // Bare token/agent events have no provider -> op only.
        assert_eq!(span_name_for(&token_event()), "llm");
        assert_eq!(
            span_name_for(&ObsEvent::AgentMetrics(AgentMetrics::default())),
            "agent"
        );
        // Provider-carrying events template as `{op} {provider}` (T2 §4).
        let cost = ObsEvent::Cost(CostEvent {
            trace_id: None,
            scope: None,
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 1000,
            cost_usd: 0.75,
        });
        assert_eq!(span_name_for(&cost), "cost openai");
    }

    #[test]
    fn span_name_templates_provider_onto_run_error() {
        let evt = ObsEvent::RunError(RunEvent {
            trace_id: None,
            provider: Some("anthropic".to_string()),
            model: None,
            error: "boom".to_string(),
        });
        assert_eq!(span_name_for(&evt), "agent.error anthropic");
    }

    #[test]
    fn template_span_name_omits_blank_provider() {
        assert_eq!(template_span_name("llm", None), "llm");
        assert_eq!(template_span_name("llm", Some("openai")), "llm openai");
        assert_eq!(template_span_name("cost", Some("")), "cost");
    }

    #[test]
    fn token_usage_record_yields_total_tokens() {
        let (tokens, attrs) = token_usage_record(&token_event()).unwrap();
        assert_eq!(tokens, 14);
        // A bare token event tags the provider as unknown so the counter row is
        // still filterable without claiming a real provider.
        assert_eq!(
            attr(&attrs, GEN_AI_PROVIDER_NAME).unwrap().value,
            Value::from("unknown")
        );
    }

    #[test]
    fn cost_record_yields_prompt_plus_completion_with_provider() {
        let (tokens, attrs) = token_usage_record(&ObsEvent::Cost(CostEvent {
            trace_id: None,
            scope: None,
            provider: Some("openai".to_string()),
            model: "gpt-4o-mini".to_string(),
            prompt_tokens: 1000,
            completion_tokens: 500,
            cost_usd: 0.375,
        }))
        .unwrap();
        assert_eq!(tokens, 1500);
        assert_eq!(
            attr(&attrs, GEN_AI_PROVIDER_NAME).unwrap().value,
            Value::from("openai")
        );
    }

    #[test]
    fn non_token_events_record_no_usage() {
        assert!(token_usage_record(&ObsEvent::AgentMetrics(AgentMetrics::default())).is_none());
        assert!(token_usage_record(&ObsEvent::RunError(RunEvent {
            trace_id: None,
            provider: None,
            model: None,
            error: "x".to_string(),
        }))
        .is_none());
    }

    #[test]
    fn token_usage_maps_usage_semconv() {
        let attrs = span_attributes_for(&token_event());
        assert_eq!(
            attr(&attrs, GEN_AI_OPERATION_NAME).unwrap().value,
            Value::from("chat")
        );
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
            trace_id: None,
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
            trace_id: None,
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
    fn run_error_maps_to_error_span() {
        let evt = ObsEvent::RunError(RunEvent {
            trace_id: Some("trace-x".to_string()),
            provider: Some("openai".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            error: "tool failed".to_string(),
        });
        assert_eq!(span_name_for(&evt), "agent.error openai");
        let attrs = span_attributes_for(&evt);
        assert_eq!(
            attr(&attrs, TRACE_ID_ATTR).unwrap().value,
            Value::from("trace-x")
        );
        assert_eq!(
            attr(&attrs, GEN_AI_PROVIDER_NAME).unwrap().value,
            Value::from("openai")
        );
        assert_eq!(
            attr(&attrs, GEN_AI_REQUEST_MODEL).unwrap().value,
            Value::from("gpt-4o-mini")
        );
        assert_eq!(
            attr(&attrs, "error.message").unwrap().value,
            Value::from("tool failed")
        );
    }

    #[test]
    fn run_error_sans_provider_model_omits_those_attrs() {
        let evt = ObsEvent::RunError(RunEvent {
            trace_id: None,
            provider: None,
            model: None,
            error: "boom".to_string(),
        });
        let attrs = span_attributes_for(&evt);
        assert!(attr(&attrs, GEN_AI_PROVIDER_NAME).is_none());
        assert!(attr(&attrs, GEN_AI_REQUEST_MODEL).is_none());
        assert!(attr(&attrs, TRACE_ID_ATTR).is_none());
        assert_eq!(
            attr(&attrs, "error.message").unwrap().value,
            Value::from("boom")
        );
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
        assert_eq!(
            attr(&attrs, TRACE_ID_ATTR).unwrap().value,
            Value::from("trace-x")
        );
        assert_eq!(
            attr(&attrs, AGENT_LLM_CALLS_ATTR).unwrap().value,
            Value::from(2i64)
        );
        assert_eq!(
            attr(&attrs, AGENT_CACHE_HITS_ATTR).unwrap().value,
            Value::from(1i64)
        );
        assert_eq!(
            attr(&attrs, AGENT_TOOL_CALLS_ATTR).unwrap().value,
            Value::from(4i64)
        );
        assert_eq!(
            attr(&attrs, AGENT_COMPACTIONS_ATTR).unwrap().value,
            Value::from(1i64)
        );
        assert_eq!(
            attr(&attrs, AGENT_TOTAL_TOKENS_ATTR).unwrap().value,
            Value::from(30i64)
        );
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
    fn install_outside_tokio_runtime_returns_error() {
        // §4: a sync (non-Tokio) thread must get a clear error, not a panic/deadlock
        // from the batch exporter's runtime::Tokio scheduling.
        let err = OtlpSink::install(OtlpConfig::default()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Tokio runtime"), "got: {msg}");
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
        sink.export(&token_event())
            .await
            .expect("record does not fail");
        sink.export(&ObsEvent::AgentMetrics(AgentMetrics::default()))
            .await
            .expect("record does not fail");
        drop(sink); // shutdown flushes (connection-refused is logged, not panicky)
    }
}
