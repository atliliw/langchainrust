//! Provider-scoped construction of the unified [`lc_core::HttpClient`].
//!
//! 0.25.0: the chat model paths (OpenAI / Azure / Cohere / Responses) moved
//! from per-provider reqwest clients + the local `retry` shim to the unified
//! HTTP layer in lc-core, so retriable-status classification, `Retry-After`
//! handling, body bounds and POST retry safety are defined once.
//!
//! Provider calls get longer budgets than the generic core profiles: a
//! non-streaming reasoning-model generation can legitimately take minutes,
//! while the core 60 s default targets ordinary API calls.

use std::time::Duration;

use lc_core::http::{HttpClient, Profile, RequestOptions};

/// TCP connect budget for provider calls.
pub(crate) const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Total budget for a buffered provider call (covers a slow long generation).
pub(crate) const PROVIDER_API_TOTAL_TIMEOUT: Duration = Duration::from_secs(180);
/// Budget for establishing a provider SSE stream; the stream itself is
/// unbounded — only its establishment may fail/retry.
pub(crate) const PROVIDER_SSE_ESTABLISH_TIMEOUT: Duration = Duration::from_secs(300);

/// Buffered JSON client for provider API calls (10 MiB response cap, retries
/// per the core default policy).
pub(crate) fn provider_api_client() -> HttpClient {
    HttpClient::builder(Profile::Api)
        .timeouts(PROVIDER_CONNECT_TIMEOUT, PROVIDER_API_TOTAL_TIMEOUT)
        .build()
        .expect("provider API HTTP client builder cannot fail with constant settings")
}

/// Streaming client for provider SSE calls (response body never buffered).
pub(crate) fn provider_sse_client() -> HttpClient {
    HttpClient::builder(Profile::Sse)
        .timeouts(PROVIDER_CONNECT_TIMEOUT, PROVIDER_SSE_ESTABLISH_TIMEOUT)
        .build()
        .expect("provider SSE HTTP client builder cannot fail with constant settings")
}

/// Assembles per-request auth/headers for a provider config.
///
/// `send_auth=false` covers keyless local OpenAI-compatible endpoints
/// (LM Studio, vLLM, the Ollama `/v1` shim). Malformed caller-supplied
/// extra headers are skipped with a warning rather than failing the whole
/// request — they originate from user configuration, not wire data.
pub(crate) fn provider_request_options(
    send_auth: bool,
    api_key: &str,
    extra_headers: &[(String, String)],
) -> RequestOptions {
    let mut opts = RequestOptions::new();
    if send_auth && !api_key.is_empty() {
        opts = opts.bearer(api_key);
    }
    for (name, value) in extra_headers {
        let (Ok(header_name), Ok(header_value)) = (
            reqwest::header::HeaderName::try_from(name.as_str()),
            reqwest::header::HeaderValue::try_from(value.as_str()),
        ) else {
            log::warn!("Ignoring malformed extra HTTP header {name:?}");
            continue;
        };
        opts = opts.header(header_name, header_value);
    }
    opts
}
