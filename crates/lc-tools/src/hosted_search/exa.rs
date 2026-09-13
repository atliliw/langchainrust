// lc-tools/src/hosted_search/exa.rs
//! Exa neural search backend (B6, v0.22.4).
//!
//! API reference: `POST https://api.exa.ai/search` with `x-api-key`, JSON
//! body `{query, numResults, contents: {text: true}}`. Hits live under
//! `data.results` with native `score` in `0..=1`, `publishedDate`, `author`
//! and the excerpt in `text`.

use async_trait::async_trait;
use serde_json::json;

use super::{require_env, trim_trailing_slash, BackendResponse, SearchBackend, SearchResult};
use lc_core::tools::ToolError;

/// Exa API endpoint.
pub const EXA_BASE_URL: &str = "https://api.exa.ai/search";
/// Backend label used in tool names, output and logs.
pub const EXA_LABEL: &str = "exa";
/// Maximum characters of page text requested per hit.
const EXA_TEXT_MAX_CHARS: u32 = 1_000;

/// Exa neural search backend.
#[derive(Clone)]
pub struct ExaBackend {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for ExaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never log the API key (traces / error reports may capture Debug).
        f.debug_struct("ExaBackend")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ExaBackend {
    /// Creates the backend with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: EXA_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .user_agent("LangChainRust/0.22 (Exa Search)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates the backend from `EXA_API_KEY`.
    pub fn from_env() -> Result<Self, ToolError> {
        Ok(Self::new(require_env("EXA_API_KEY")?))
    }

    /// Overrides the endpoint (tests / proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = trim_trailing_slash(base_url.into());
        self
    }

    /// Pure response parser (`data.results` envelope).
    pub(crate) fn parse(body: &serde_json::Value, top_k: usize) -> BackendResponse {
        let mut results = Vec::new();
        if let Some(items) = body
            .get("data")
            .and_then(|d| d.get("results"))
            .and_then(|v| v.as_array())
        {
            for item in items {
                let Some(url) = item.get("url").and_then(|v| v.as_str()) else {
                    continue;
                };
                let score = item
                    .get("score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                results.push(SearchResult {
                    title: item
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    url: url.to_string(),
                    snippet: item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    score,
                    published_date: item
                        .get("publishedDate")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    author: item.get("author").and_then(|v| match v {
                        serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
                        _ => None,
                    }),
                    provider: EXA_LABEL,
                });
                if results.len() >= top_k {
                    break;
                }
            }
        }
        BackendResponse {
            results,
            answer: None,
        }
    }
}

#[async_trait]
impl SearchBackend for ExaBackend {
    fn label(&self) -> &'static str {
        EXA_LABEL
    }

    async fn search(
        &self,
        query: &str,
        top_k: usize,
        _include_answer: bool,
    ) -> Result<BackendResponse, ToolError> {
        // contents.text asks Exa to return the cleaned page excerpt alongside
        // metadata; without it snippets would always be empty.
        let payload = json!({
            "query": query,
            "numResults": top_k,
            "contents": {"text": {"maxCharacters": EXA_TEXT_MAX_CHARS}},
        });
        let response = self
            .client
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("exa request failed: {e}")))?;
        let status = response.status();
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("exa response parse failed: {e}")))?;
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "exa returned HTTP {}: {}",
                status.as_u16(),
                body.to_string().chars().take(300).collect::<String>()
            )));
        }
        Ok(Self::parse(&body, top_k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosted_search::test_support::{spawn_one_shot_json, ENV_LOCK};
    use serde_json::json;

    #[test]
    fn parse_reads_data_envelope_and_metadata() {
        let body = json!({"data": {"results": [
            {"title":"A","url":"https://a.example","text":"excerpt","score":0.77,"publishedDate":"2026-02-10","author":"Lee"},
            {"title":"B","url":"https://b.example","text":"","score":null},
            {"title":"no-url"}
        ]}});
        let parsed = ExaBackend::parse(&body, 10);
        assert_eq!(parsed.results.len(), 2);
        assert_eq!(parsed.results[0].snippet, "excerpt");
        assert!((parsed.results[0].score - 0.77).abs() < 1e-9);
        assert_eq!(
            parsed.results[0].published_date.as_deref(),
            Some("2026-02-10")
        );
        assert_eq!(parsed.results[0].author.as_deref(), Some("Lee"));
        assert_eq!(parsed.results[0].provider, EXA_LABEL);
        // Missing/None score normalizes to 0 rather than NaN.
        assert_eq!(parsed.results[1].score, 0.0);
        // Empty-string author normalizes to None.
        assert!(parsed.results[1].author.is_none());
    }

    #[test]
    fn parse_handles_envelope_without_results() {
        assert!(ExaBackend::parse(&json!({"data": {}}), 5)
            .results
            .is_empty());
        assert!(ExaBackend::parse(&json!({}), 5).results.is_empty());
    }

    #[tokio::test]
    async fn http_call_requests_text_contents_and_sends_key() {
        let reply = json!({"data": {"results": [
            {"title":"A","url":"https://a.example","text":"t","score":0.5}
        ]}});
        let (base, request_rx) = spawn_one_shot_json(reply).await;
        let backend = ExaBackend::new("exa-secret").with_base_url(base);
        let out = backend
            .search("long-context memory survey", 2, false)
            .await
            .unwrap();
        assert_eq!(out.results[0].url, "https://a.example");

        let request = String::from_utf8(request_rx.await.unwrap()).unwrap();
        // HTTP header names are case-insensitive; normalize before asserting.
        let head = request.split("\r\n\r\n").next().unwrap().to_lowercase();
        assert!(head.contains("post / http/1.1"));
        assert!(head.contains("x-api-key: exa-secret"), "{head}");
        let body = request.split("\r\n\r\n").nth(1).unwrap();
        let sent: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(sent["numResults"], 2);
        assert_eq!(
            sent["contents"]["text"]["maxCharacters"],
            EXA_TEXT_MAX_CHARS
        );
    }

    #[test]
    fn from_env_requires_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved = std::env::var("EXA_API_KEY").ok();
        std::env::remove_var("EXA_API_KEY");
        let err = ExaBackend::from_env().unwrap_err();
        assert!(err.to_string().contains("EXA_API_KEY"));
        if let Some(value) = saved {
            std::env::set_var("EXA_API_KEY", value);
        }
    }
}
