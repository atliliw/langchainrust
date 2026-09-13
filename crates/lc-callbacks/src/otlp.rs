//! OTLP exporter pipeline (feature = "otlp", B9 v0.22.4)
//!
//! A batteries-included setup for shipping the spans produced by
//! [`crate::OtelHandler`] to an OpenTelemetry collector:
//!
//! - exporter: OTLP over **HTTP/JSON** (reqwest, no native TLS toolchain
//!   required)
//! - processor: batching, on the Tokio runtime
//! - context propagation: W3C trace context installed globally
//! - resource: `service.name` (from `OTEL_SERVICE_NAME`)
//!
//! The SDK provider is optional by default — the plain `opentelemetry`
//! feature keeps the API-only dependency footprint for in-process
//! recording. Enable `otlp` for actual export.
//!
//! # Environment variables
//!
//! - `OTEL_EXPORTER_OTLP_ENDPOINT` — collector base URL
//!   (default `http://localhost:4318`; the exporter appends `/v1/traces`)
//! - `OTEL_EXPORTER_OTLP_HEADERS` — `k1=v1,k2=v2`, values
//!   percent-decoded (commas/equals inside values must be encoded)
//! - `OTEL_EXPORTER_OTLP_TIMEOUT` — request timeout in seconds
//!   (default 10)
//! - `OTEL_SERVICE_NAME` — resource service name
//!   (default `langchainrust`)
//!
//! # Example
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use lc_callbacks::{OtelHandler, otlp::install_otlp_pipeline};
//! use std::sync::Arc;
//!
//! // Keep the guard alive for the whole process; dropping it flushes and
//! // shuts the pipeline down.
//! let _otlp = install_otlp_pipeline()?;
//! let handler = Arc::new(OtelHandler::from_global("langchainrust"));
//! let _ = handler;
//! # Ok(()) }
//! ```

use opentelemetry::global;
use opentelemetry::trace::TraceError;
use opentelemetry::{KeyValue, Value};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::runtime;
use opentelemetry_sdk::trace::TracerProvider;
use std::collections::HashMap;
use std::env;
use std::time::Duration;

const DEFAULT_ENDPOINT: &str = "http://localhost:4318";
const DEFAULT_SERVICE_NAME: &str = "langchainrust";
const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// Configuration for the OTLP pipeline. [`Default`] reads the standard
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

/// Owns the global OTLP tracer provider. Dropping the guard shuts the
/// provider down, forcing a final flush of batched spans.
pub struct OtlpGuard {
    provider: TracerProvider,
}

impl Drop for OtlpGuard {
    fn drop(&mut self) {
        // Best-effort flush; nothing to do with an error at shutdown.
        let _ = self.provider.shutdown();
    }
}

/// Install an OTLP/HTTP-JSON tracer provider from process environment
/// configuration and make it the global provider (W3C trace-context
/// propagation included).
///
/// Keep the returned guard alive until all spans have been emitted.
pub fn install_otlp_pipeline() -> Result<OtlpGuard, TraceError> {
    install_otlp_pipeline_with(OtlpConfig::default())
}

/// Install an OTLP/HTTP-JSON tracer provider with explicit configuration.
pub fn install_otlp_pipeline_with(config: OtlpConfig) -> Result<OtlpGuard, TraceError> {
    let exporter: SpanExporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(config.endpoint)
        // The crate builds one protocol per set of enabled features; setting
        // it explicitly documents the wire format and keeps tooling honest.
        .with_protocol(Protocol::HttpJson)
        .with_timeout(config.timeout)
        .with_headers(config.headers)
        .build()?;

    let resource = Resource::new([KeyValue::new(
        "service.name",
        Value::from(config.service_name),
    )]);

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(resource)
        .build();

    // Global propagation is needed for cross-process parent linkage.
    global::set_text_map_propagator(TraceContextPropagator::new());
    global::set_tracer_provider(provider.clone());

    Ok(OtlpGuard { provider })
}

/// Parses the `OTEL_EXPORTER_OTLP_HEADERS` value: comma-separated `k=v`
/// pairs with percent-decoding, matching the OTLP spec. Malformed pairs
/// (no `=`) are skipped.
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

    #[test]
    fn parses_comma_separated_headers() {
        let map = parse_headers("authorization=Bearer abc,x-custom=v");
        assert_eq!(map.get("authorization").unwrap(), "Bearer abc");
        assert_eq!(map.get("x-custom").unwrap(), "v");
    }

    #[test]
    fn skips_malformed_pairs_and_blank_keys() {
        let map = parse_headers("no-equals, ,=novalue,k=v");
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("k").unwrap(), "v");
    }

    #[test]
    fn decodes_percent_encoded_values() {
        assert_eq!(percent_decode("a%2Cb"), "a,b");
        assert_eq!(percent_decode("a%3Db"), "a=b");
        assert_eq!(percent_decode("%41%42%43"), "ABC");
        assert_eq!(percent_decode("plain"), "plain");
        // Invalid escapes are preserved rather than panicking.
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn default_config_falls_back_when_env_absent() {
        // The test process must not export anywhere; unset the endpoint if a
        // developer happens to have it set in their shell.
        env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        let config = OtlpConfig::default();
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));
    }
}
