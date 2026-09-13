// lc-tools/src/hosted_search/tavily.rs
//! Tavily hosted search backend (B6, v0.22.4).
//!
//! API reference: `POST https://api.tavily.com/search` with a bearer key,
//! JSON body `{query, max_results, search_depth, include_answer}`. The
//! response carries RAG-tuned snippets (`content`), per-result `score` in
//! `0..=1`, and an optional synthesized `answer`.

use async_trait::async_trait;
use serde_json::json;

use super::{require_env, trim_trailing_slash, BackendResponse, SearchBackend, SearchResult};
use lc_core::tools::ToolError;

/// Tavily API endpoint.
pub const TAVILY_BASE_URL: &str = "https://api.tavily.com/search";
/// Backend label used in tool names, output and logs.
pub const TAVILY_LABEL: &str = "tavily";

/// Tavily search backend.
#[derive(Clone)]
pub struct TavilyBackend {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for TavilyBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never log the bearer key (defense in depth: it may appear in agent
        // traces, error reports or Debug-derived logs).
        f.debug_struct("TavilyBackend")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl TavilyBackend {
    /// Creates the backend with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: TAVILY_BASE_URL.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .user_agent("LangChainRust/0.22 (Tavily Search)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Creates the backend from `TAVILY_API_KEY`.
    pub fn from_env() -> Result<Self, ToolError> {
        Ok(Self::new(require_env("TAVILY_API_KEY")?))
    }

    /// Overrides the endpoint (tests / proxies / self-hosted gateways).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = trim_trailing_slash(base_url.into());
        self
    }

    /// Pure response parser; `top_k` truncates provider-side ordering.
    pub(crate) fn parse(
        body: &serde_json::Value,
        top_k: usize,
    ) -> Result<BackendResponse, ToolError> {
        let answer = body.get("answer").and_then(|v| match v {
            serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
            _ => None,
        });
        let mut results = Vec::new();
        if let Some(items) = body.get("results").and_then(|v| v.as_array()) {
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
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    score,
                    published_date: item
                        .get("published_date")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    author: item
                        .get("author")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    provider: TAVILY_LABEL,
                });
                if results.len() >= top_k {
                    break;
                }
            }
        }
        Ok(BackendResponse { results, answer })
    }
}

#[async_trait]
impl SearchBackend for TavilyBackend {
    fn label(&self) -> &'static str {
        TAVILY_LABEL
    }

    async fn search(
        &self,
        query: &str,
        top_k: usize,
        include_answer: bool,
    ) -> Result<BackendResponse, ToolError> {
        let payload = json!({
            "query": query,
            "max_results": top_k,
            "search_depth": "basic",
            "include_answer": include_answer,
        });
        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("tavily request failed: {e}")))?;
        let status = response.status();
        let body: serde_json::Value = response.json().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("tavily response parse failed: {e}"))
        })?;
        if !status.is_success() {
            return Err(ToolError::ExecutionFailed(format!(
                "tavily returned HTTP {}: {}",
                status.as_u16(),
                body.to_string().chars().take(300).collect::<String>()
            )));
        }
        Self::parse(&body, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hosted_search::test_support::{spawn_one_shot_json, ENV_LOCK};
    use serde_json::json;

    #[test]
    fn parse_maps_results_answer_and_bad_rows() {
        let body = json!({
            "answer": "short synthesis",
            "results": [
                {"title":"A","url":"https://a.example","content":"snip-a","score":0.91,"published_date":"2026-03-01","author":"Ada"},
                {"title":"B","url":"https://b.example","content":"snip-b","score":0.4},
                {"title":"no-url"},
            ]
        });
        let parsed = TavilyBackend::parse(&body, 10).unwrap();
        assert_eq!(parsed.answer.as_deref(), Some("short synthesis"));
        assert_eq!(parsed.results.len(), 2, "row without url must be skipped");
        assert_eq!(parsed.results[0].url, "https://a.example");
        assert!((parsed.results[0].score - 0.91).abs() < 1e-9);
        assert_eq!(
            parsed.results[0].published_date.as_deref(),
            Some("2026-03-01")
        );
        assert_eq!(parsed.results[0].author.as_deref(), Some("Ada"));
        assert_eq!(parsed.results[1].provider, TAVILY_LABEL);

        let capped = TavilyBackend::parse(&body, 1).unwrap();
        assert_eq!(capped.results.len(), 1);

        // Empty/absent answer is normalized to None.
        let body_no_answer = json!({"answer": "", "results": []});
        assert!(TavilyBackend::parse(&body_no_answer, 5)
            .unwrap()
            .answer
            .is_none());
        // Out-of-range score clamps into 0..=1.
        let weird = json!({"results":[{"url":"u","score":7.0}]});
        assert_eq!(
            TavilyBackend::parse(&weird, 5).unwrap().results[0].score,
            1.0
        );
    }

    #[tokio::test]
    async fn http_call_sends_bearer_and_payload_and_parses_response() {
        let reply = json!({"answer": null, "results": [
            {"title":"A","url":"https://a.example","content":"x","score":0.8}
        ]});
        let (base, request_rx) = spawn_one_shot_json(reply).await;
        let backend = TavilyBackend::new("tvly-secret").with_base_url(base);
        let out = backend.search("rust async", 3, false).await.unwrap();
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.results[0].url, "https://a.example");

        let request = String::from_utf8(request_rx.await.unwrap()).unwrap();
        // Header names are case-insensitive (HTTP/1.1); reqwest sends
        // `Authorization`, so compare the head lowercased.
        let head = request.split("\r\n\r\n").next().unwrap().to_lowercase();
        assert!(head.contains("post / http/1.1"), "{head}");
        assert!(
            head.contains("authorization: bearer tvly-secret"),
            "auth header missing: {head}"
        );
        let body = request.split("\r\n\r\n").nth(1).unwrap();
        let sent: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(sent["query"], "rust async");
        assert_eq!(sent["max_results"], 3);
        assert_eq!(sent["include_answer"], false);
    }

    #[tokio::test]
    async fn connection_failure_is_surfaced() {
        // Accept then immediately drop the TCP connection without writing an
        // HTTP response. A plain closed port is unreliable here: some Windows
        // machines run transparent proxies that answer every loopback port.
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((socket, _)) = listener.accept().await {
                drop(socket);
            }
        });
        let dead = TavilyBackend::new("k")
            .with_base_url(format!("http://{addr}/v1"))
            .search("q", 1, false)
            .await;
        let err = dead.unwrap_err().to_string();
        assert!(err.starts_with("Execution failed: tavily "), "{err}");
        assert!(!err.contains("HTTP 2"), "{err}");
    }

    #[tokio::test]
    async fn non_2xx_status_includes_body_excerpt() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let body = r#"{"detail":"invalid api key"}"#;
            let response = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        let err = TavilyBackend::new("bad")
            .with_base_url(format!("http://{addr}"))
            .search("q", 1, false)
            .await
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("HTTP 401"), "{text}");
        assert!(text.contains("invalid api key"), "{text}");
    }

    #[test]
    fn from_env_requires_key() {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved = std::env::var("TAVILY_API_KEY").ok();
        std::env::remove_var("TAVILY_API_KEY");
        let err = TavilyBackend::from_env().unwrap_err();
        assert!(err.to_string().contains("TAVILY_API_KEY"));
        if let Some(value) = saved {
            std::env::set_var("TAVILY_API_KEY", value);
        }
    }
}
