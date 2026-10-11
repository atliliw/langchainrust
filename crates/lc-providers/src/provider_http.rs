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

/// Idle (silence) budget for an established SSE stream. Some providers drive
/// streaming through a bare `client.post(...).send()` + `bytes_stream()`
/// (Gemini, Anthropic, Ollama) that never arms a total-timeout, so a peer that
/// stops sending and never closes would hang forever. Wrapping each stream
/// `next()` in this timeout turns "no bytes for IDLE" into a
/// `StreamInterrupted` error instead of an unbounded park. The timer resets on
/// every received chunk, so it stays a true idle guard, not a wall-clock cap.
pub(crate) const PROVIDER_SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

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
/// (LM Studio, vLLM, the Ollama `/v1` shim). When `send_auth=true` the caller
/// is a key-required cloud provider: an empty key is a configuration error and
/// fails fast with a clear message instead of silently sending a bearer-less
/// request that would 401 over the wire. Malformed caller-supplied extra
/// headers are skipped with a warning rather than failing the whole request —
/// they originate from user configuration, not wire data.
///
/// # Errors
/// Returns an `Err(message)` when `send_auth` is true but `api_key` is empty.
pub(crate) fn provider_request_options(
    send_auth: bool,
    api_key: &str,
    extra_headers: &[(String, String)],
) -> Result<RequestOptions, String> {
    if send_auth && api_key.is_empty() {
        return Err(
            "cannot authenticate: API key is empty; set one via the provider config \
             (e.g. from_env) to fail fast instead of a 401 at request time"
                .to_string(),
        );
    }
    let mut opts = RequestOptions::new();
    if send_auth {
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
    Ok(opts)
}

/// Fail-fast guard shared by providers that attach their auth header directly
/// (Anthropic `x-api-key`, Gemini `x-goog-api-key`) rather than through
/// [`provider_request_options`]. A key-required provider with an empty key is a
/// configuration error: return a clear message before building the request
/// instead of silently shipping a keyless call that 401s over the wire.
///
/// This closes the validation-timing gap (§4 lc-providers): the OpenAI-family /
/// Azure / Cohere / Responses fail fast in `provider_request_options`, and these
/// two direct-header providers now fail fast identically at the same stage.
///
/// # Errors
/// Returns an `Err(message)` when `api_key` is empty, naming the env var to set.
pub(crate) fn ensure_provider_api_key(api_key: &str, env_var: &str) -> Result<(), String> {
    if api_key.is_empty() {
        return Err(format!(
            "cannot authenticate: {env_var} is empty; set it (or construct the \
             provider via from_env) to fail fast instead of a 401 at request time"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T1: a key-required (`send_auth=true`) provider with an empty key fails
    /// fast at request-option assembly instead of silently sending no bearer.
    #[test]
    fn empty_key_with_auth_fails_fast() {
        let err = provider_request_options(true, "", &[]).unwrap_err();
        assert!(err.contains("API key is empty"), "got: {err}");
    }

    /// Keyless local endpoints (`send_auth=false`) never fail, even with an
    /// empty key.
    #[test]
    fn keyless_send_auth_false_is_ok() {
        assert!(provider_request_options(false, "", &[]).is_ok());
    }

    /// A valid key keeps working.
    #[test]
    fn non_empty_key_is_ok() {
        assert!(provider_request_options(true, "sk-test", &[]).is_ok());
    }

    /// §4: a direct-header provider with an empty key fails fast and names the
    /// env var, so Anthropic/Gemini match the OpenAI-family behaviour.
    #[test]
    fn ensure_provider_api_key_fails_fast_on_empty() {
        let err = ensure_provider_api_key("", "ANTHROPIC_API_KEY").unwrap_err();
        assert!(err.contains("ANTHROPIC_API_KEY is empty"), "got: {err}");
    }

    #[test]
    fn ensure_provider_api_key_ok_when_present() {
        assert!(ensure_provider_api_key("sk-test", "GEMINI_API_KEY").is_ok());
    }
}
