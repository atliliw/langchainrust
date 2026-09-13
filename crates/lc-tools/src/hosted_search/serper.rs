// lc-tools/src/hosted_search/serper.rs
//! Serper hosted search backend (B6, v0.22.4).
//!
//! API reference: `POST https://google.serper.dev/search` with `X-API-KEY`,
//! JSON body `{q, num}`. Serper returns Google-style `organic` hits (title,
//! link, snippet, 1-based `position`, optional `date`) and has no synthesized
//! answer; rank position is converted into a `0..=1` score.

use async_trait::async_trait;
use serde_json::json;

use super::{require_env, trim_trailing_slash, BackendResponse, SearchBackend, SearchResult};
use lc_core::tools::ToolError;

/// Serper API endpoint.
pub const SERPER_BASE_URL: &str = "https://google.serper.dev/search";
/// Backend label used in tool names, output and logs.
pub const SERPER_LABEL: &str = "serper";

/// Serper (Google results) search backend.
#[derive(Clone)]
pub struct SerperBackend {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for SerperBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never log the API key (traces / error reports may capture Debug).
        f.debug_struct("SerperBackend")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl SerperBackend {
    /// Creates the backend with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: SERPER_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .user_agent("LangChainRust/0.22 (Serper Search)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates the backend from `SERPER_API_KEY`.
    pub fn from_env() -> Result<Self, ToolError> {
        Ok(Self::new(require_env("SERPER_API_KEY")?))
    }

    /// Overrides the endpoint (tests / proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = trim_trailing_slash(base_url.into());
        self
    }

    /// Pure response parser.
    ///
    /// Position `p` (1-based) among `n` organic hits maps to
    /// `1 - (p-1)/n`, so the first hit scores 1.0 and the last approaches 0.
    pub(crate) fn parse(body: &serde_json::Value, top_k: usize) -> BackendResponse {
        let mut results = Vec::new();
        if let Some(items) = body.get("organic").and_then(|v| v.as_array()) {
            let n = items.len().max(1) as f64;
            for item in items {
                let Some(link) = item.get("link").and_then(|v| v.as_str()) else {
                    continue;
                };
                let position =
                    item.get("position")
                        .and_then(|v| v.as_u64())
                        .unwrap_or_else(|| results.len() as u64 + 1) as f64;
                let score = (1.0 - (position - 1.0) / n).clamp(0.0, 1.0);
                results.push(SearchResult {
                    title: item
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    url: link.to_string(),
                    snippet: item
                        .get("snippet")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    score,
                    published_date: item
                        .get("date")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    author: None,
                    provider: SERPER_LABEL,
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
impl SearchBackend for SerperBackend {
    fn label(&self) -> &'static str {
        SERPER_LABEL
    }

    async fn search(
        &self,
        query: &str,
        top_k: usize,
        _include_answer: bool,
    ) -> Result<BackendResponse, ToolError> {
        let payload = json!({"q": query, "num": top_k});
        let response = self
            .client
            .post(&self.base_url)
            .header("X-API-KEY", &self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("serper request failed: {e}")))?;
        let status = response.status();
        let body: serde_json::Value = response.json().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("serper response parse failed: {e}"))
        })?;
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "serper returned HTTP {}: {}",
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
    fn parse_converts_positions_to_descending_scores() {
        let body = json!({"organic": [
            {"title":"A","link":"https://a.example","snippet":"sa","position":1,"date":"3 days ago"},
            {"title":"B","link":"https://b.example","snippet":"sb","position":2},
            {"title":"no-link","snippet":"skip"},
            {"title":"C","link":"https://c.example","snippet":"sc","position":4}
        ]});
        let parsed = SerperBackend::parse(&body, 10);
        assert_eq!(parsed.results.len(), 3);
        assert_eq!(parsed.results[0].url, "https://a.example");
        assert_eq!(parsed.results[0].score, 1.0);
        // n = 4 rows in the organic array (including the skipped row): 1 - 1/4.
        assert!((parsed.results[1].score - 0.75).abs() < 1e-9);
        assert_eq!(
            parsed.results[0].published_date.as_deref(),
            Some("3 days ago")
        );
        assert_eq!(parsed.results[2].provider, SERPER_LABEL);
        assert!(parsed.answer.is_none());

        let capped = SerperBackend::parse(&body, 2);
        assert_eq!(capped.results.len(), 2);
    }

    #[test]
    fn parse_handles_missing_organic() {
        let parsed = SerperBackend::parse(&json!({"searchParameters": {"q": "x"}}), 5);
        assert!(parsed.results.is_empty());
    }

    #[tokio::test]
    async fn http_call_sends_api_key_header_and_q_num() {
        let reply = json!({"organic": [
            {"title":"A","link":"https://a.example","snippet":"s","position":1}
        ]});
        let (base, request_rx) = spawn_one_shot_json(reply).await;
        let backend = SerperBackend::new("serper-secret").with_base_url(base);
        let out = backend.search("rust lang", 5, true).await.unwrap();
        assert_eq!(out.results[0].title, "A");
        // include_answer is irrelevant to Serper; answer is always None.
        assert!(out.answer.is_none());

        let request = String::from_utf8(request_rx.await.unwrap()).unwrap();
        // HTTP header names are case-insensitive; normalize before asserting.
        let head = request.split("\r\n\r\n").next().unwrap().to_lowercase();
        assert!(head.contains("post / http/1.1"));
        assert!(
            head.contains("x-api-key: serper-secret"),
            "missing X-API-KEY: {head}"
        );
        let body = request.split("\r\n\r\n").nth(1).unwrap();
        let sent: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(sent["q"], "rust lang");
        assert_eq!(sent["num"], 5);
    }

    #[test]
    fn from_env_requires_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved = std::env::var("SERPER_API_KEY").ok();
        std::env::remove_var("SERPER_API_KEY");
        let err = SerperBackend::from_env().unwrap_err();
        assert!(err.to_string().contains("SERPER_API_KEY"));
        if let Some(value) = saved {
            std::env::set_var("SERPER_API_KEY", value);
        }
    }
}
