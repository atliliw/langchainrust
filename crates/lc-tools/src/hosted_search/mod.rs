// lc-tools/src/hosted_search/mod.rs
//! Hosted web-search backends behind one unified result shape (B6, v0.22.4).
//!
//! The framework previously shipped only the keyless DuckDuckGo instant-answer
//! tool ([`crate::DuckDuckGoSearchTool`]). Agentic web research in 2026 mostly
//! runs through paid hosted search APIs with cleaner result quality and
//! optional synthesized answers:
//!
//! - [Tavily](https://tavily.com) — `tavily_search` (RAG-tuned snippets + answer)
//! - [Serper](https://serper.dev) — `serper_search` (Google result pages)
//! - [Exa](https://exa.ai) — `exa_search` (neural/embedding search)
//!
//! All three share a single [`HostedSearchTool`] and a single result schema;
//! only the [`SearchBackend`] implementations differ. Provider relevance
//! signals are normalized into `0..=1`, results are re-ranked and URL-deduped
//! before being handed back to the agent.
//!
//! ```no_run
//! # async fn demo() -> Result<(), lc_tools::ToolError> {
//! use lc_tools::hosted_search::HostedSearchTool;
//! use lc_tools::BaseTool;
//! let tool = HostedSearchTool::tavily_from_env()?;
//! let out = tool
//!     .run(serde_json::json!({"query": "Rust 1.85 release notes"}).to_string())
//!     .await?;
//! # let _ = out;
//! # Ok(()) }
//! ```

pub mod exa;
pub mod serper;
pub mod tavily;

pub use exa::ExaBackend;
pub use serper::SerperBackend;
pub use tavily::TavilyBackend;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use lc_core::tools::{BaseTool, ToolError};

/// Default number of hits returned when the caller omits `top_k`.
pub const DEFAULT_TOP_K: usize = 5;
/// Maximum number of hits callers may request in one tool call.
pub const MAX_TOP_K: usize = 20;

/// One ranked web hit, independent of the hosting provider.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SearchResult {
    /// Result title.
    pub title: String,
    /// Canonicalized result URL (fragment stripped).
    pub url: String,
    /// Provider text snippet / page excerpt.
    pub snippet: String,
    /// Relevance normalized into `0..=1`; higher is better.
    ///
    /// Providers that ship a native score (Tavily, Exa) keep it; position-based
    /// providers (Serper) get a rank-derived score. The tool re-ranks by this
    /// field, so callers never need provider-specific comparisons.
    pub score: f64,
    /// Publisher-reported date, when the API returns one.
    pub published_date: Option<String>,
    /// Publisher-reported author, when the API returns one.
    pub author: Option<String>,
    /// Backend label (`"tavily"`, `"serper"`, `"exa"`, …).
    pub provider: &'static str,
}

/// Unified output of a hosted search call.
#[derive(Debug, Clone, Serialize)]
pub struct SearchOutput {
    /// The query that was executed.
    pub query: String,
    /// Ranked, URL-deduped hits.
    pub results: Vec<SearchResult>,
    /// Synthesized answer, for backends/requests that produce one (Tavily).
    pub answer: Option<String>,
    /// Backend label.
    pub provider: &'static str,
}

/// Parsed backend response before cross-provider ranking.
#[derive(Debug, Clone, Default)]
pub struct BackendResponse {
    /// Raw hits in provider order; `score` already normalized into `0..=1`.
    pub results: Vec<SearchResult>,
    /// Synthesized answer if the backend produced one.
    pub answer: Option<String>,
}

/// A hosted search provider: maps the common query shape to the provider API.
#[async_trait]
pub trait SearchBackend: Send + Sync {
    /// Stable backend/tool label (`"tavily"`, …); the tool name is `{label}_search`.
    fn label(&self) -> &'static str;

    /// Runs the provider-specific HTTP call and parses the response.
    async fn search(
        &self,
        query: &str,
        top_k: usize,
        include_answer: bool,
    ) -> Result<BackendResponse, ToolError>;
}

/// Tool input for every hosted backend.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HostedSearchInput {
    /// The search query.
    pub query: String,
    /// Number of results to return (default 5, capped at 20).
    pub top_k: Option<usize>,
    /// Whether to request a synthesized answer when the backend supports one (default: true).
    pub include_answer: Option<bool>,
}

/// Hosted web-search tool parameterized by a [`SearchBackend`].
///
/// Construct with the provider constructors:
/// [`HostedSearchTool::tavily`], [`HostedSearchTool::serper`],
/// [`HostedSearchTool::exa`] (or their `*_from_env` variants).
pub struct HostedSearchTool {
    backend: Arc<dyn SearchBackend>,
    tool_name: String,
}

impl HostedSearchTool {
    /// Wraps an arbitrary backend (custom gateway, mock, …).
    pub fn new(backend: impl SearchBackend + 'static) -> Self {
        Self::from_arc(Arc::new(backend))
    }

    /// Wraps an already-shared backend.
    pub fn from_arc(backend: Arc<dyn SearchBackend>) -> Self {
        let tool_name = format!("{}_search", backend.label());
        Self { backend, tool_name }
    }

    /// Backend label exposed by this tool instance.
    pub fn provider(&self) -> &'static str {
        self.backend.label()
    }

    /// Tavily with an explicit API key.
    pub fn tavily(api_key: impl Into<String>) -> Self {
        Self::new(TavilyBackend::new(api_key))
    }

    /// Tavily from `TAVILY_API_KEY`.
    pub fn tavily_from_env() -> Result<Self, ToolError> {
        Ok(Self::new(TavilyBackend::from_env()?))
    }

    /// Serper with an explicit API key.
    pub fn serper(api_key: impl Into<String>) -> Self {
        Self::new(SerperBackend::new(api_key))
    }

    /// Serper from `SERPER_API_KEY`.
    pub fn serper_from_env() -> Result<Self, ToolError> {
        Ok(Self::new(SerperBackend::from_env()?))
    }

    /// Exa with an explicit API key.
    pub fn exa(api_key: impl Into<String>) -> Self {
        Self::new(ExaBackend::new(api_key))
    }

    /// Exa from `EXA_API_KEY`.
    pub fn exa_from_env() -> Result<Self, ToolError> {
        Ok(Self::new(ExaBackend::from_env()?))
    }

    /// Typed entry point used by the [`lc_core::tools::Tool`] impls of backends.
    pub async fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        include_answer: Option<bool>,
    ) -> Result<SearchOutput, ToolError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(ToolError::InvalidInput(
                "search query must not be empty".to_string(),
            ));
        }
        let top_k = top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
        let include_answer = include_answer.unwrap_or(true);

        let mut response = self.backend.search(query, top_k, include_answer).await?;
        rank_results(&mut response.results);
        response.results.truncate(top_k);

        Ok(SearchOutput {
            query: query.to_string(),
            results: response.results,
            answer: response.answer,
            provider: self.backend.label(),
        })
    }
}

/// Canonicalizes a URL for cross-provider dedup: parse, drop fragment,
/// lowercase host, strip a trailing `/` from the path. Unparseable inputs are
/// returned trimmed and lowercased so dedup still degrades gracefully.
pub(crate) fn canonical_url(raw: &str) -> String {
    let raw = raw.trim();
    match url::Url::parse(raw) {
        Ok(mut parsed) => {
            parsed.set_fragment(None);
            if let Some(host) = parsed.host_str().map(str::to_lowercase) {
                let _ = parsed.set_host(Some(&host));
            }
            let path = parsed.path().to_string();
            if path.len() > 1 && path.ends_with('/') {
                parsed.set_path(path.trim_end_matches('/'));
            }
            parsed.to_string()
        }
        Err(_) => raw.to_lowercase(),
    }
}

/// Sorts by normalized score descending (stable) and removes duplicate URLs,
/// keeping the highest-ranked occurrence.
pub(crate) fn rank_results(results: &mut Vec<SearchResult>) {
    // sort_by is stable; NaN guards (score is produced by our parsers, but never
    // trust a float enough to let NaN poison ordering).
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut seen = std::collections::HashSet::new();
    results.retain(|r| seen.insert(canonical_url(&r.url)));
}

/// Reads an API key from the environment with a provider-named error.
pub(crate) fn require_env(key: &str) -> Result<String, ToolError> {
    let value = std::env::var(key).map_err(|_| {
        ToolError::InvalidInput(format!("{key} environment variable not set or empty"))
    })?;
    let value = value.trim();
    if value.is_empty() {
        return Err(ToolError::InvalidInput(format!(
            "{key} environment variable not set or empty"
        )));
    }
    Ok(value.to_string())
}

/// Strips the trailing slashes from a configured base URL.
pub(crate) fn trim_trailing_slash(mut base: String) -> String {
    while base.len() > 1 && base.ends_with('/') {
        base.pop();
    }
    base
}

fn render_text(output: &SearchOutput) -> String {
    let mut text = format!("{} 搜索结果(查询: {})\n\n", output.provider, output.query);
    if let Some(answer) = output.answer.as_ref().filter(|a| !a.is_empty()) {
        text.push_str(&format!("综合答案: {answer}\n\n"));
    }
    for (i, result) in output.results.iter().enumerate() {
        text.push_str(&format!("{}. {}\n", i + 1, result.title));
        text.push_str(&format!("   {}\n", result.snippet));
        text.push_str(&format!("   URL: {}\n", result.url));
        text.push_str(&format!("   相关度: {:.2}\n", result.score));
        match (result.published_date.as_ref(), result.author.as_ref()) {
            (Some(date), Some(author)) => {
                text.push_str(&format!("   发布: {date} · {author}\n"));
            }
            (Some(date), None) => text.push_str(&format!("   发布: {date}\n")),
            (None, Some(author)) => text.push_str(&format!("   作者: {author}\n")),
            (None, None) => {}
        }
        text.push('\n');
    }
    if output.results.is_empty() {
        text.push_str("未找到相关结果");
    } else {
        text.push_str(&format!("共 {} 条结果", output.results.len()));
    }
    text
}

#[async_trait]
impl BaseTool for HostedSearchTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        match self.backend.label() {
            "tavily" => "Tavily 托管网页搜索工具(为 RAG/agent 优化的正文片段,可选综合答案)。\n\n参数:\n- query: 搜索关键词\n- top_k: 返回结果数量(默认 5,上限 20)\n- include_answer: 是否请求综合答案(默认 true)\n\n需设置 TAVILY_API_KEY。\n示例: {\"query\": \"Rust 1.85 async closure\", \"top_k\": 5}",
            "serper" => "Serper 托管网页搜索工具(Google 结果页)。\n\n参数:\n- query: 搜索关键词\n- top_k: 返回结果数量(默认 5,上限 20)\n- include_answer: 此后端不支持综合答案,字段被忽略\n\n需设置 SERPER_API_KEY。\n示例: {\"query\": \"tokio tungstenite connect_async\", \"top_k\": 5}",
            "exa" => "Exa 神经托管网页搜索工具(语义检索,适合研究类查询)。\n\n参数:\n- query: 搜索查询(自然语言描述)\n- top_k: 返回结果数量(默认 5,上限 20)\n- include_answer: 此后端不支持综合答案,字段被忽略\n\n需设置 EXA_API_KEY。\n示例: {\"query\": \"papers about long-context transformer memory\", \"top_k\": 5}",
            _ => "Hosted web search tool.\n\n参数:\n- query: 搜索关键词\n- top_k: 返回结果数量(默认 5,上限 20)",
        }
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: HostedSearchInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse failed: {e}")))?;
        let output = self
            .search(&parsed.query, parsed.top_k, parsed.include_answer)
            .await?;
        Ok(render_text(&output))
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        serde_json::to_value(schemars::schema_for!(HostedSearchInput)).ok()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Env-var serialization for the per-backend `*_from_env` tests and a tiny
    //! loopback JSON HTTP server so request construction is exercised without
    //! hitting the real APIs.
    use std::sync::Mutex;

    /// All hosted-search env tests mutate process environment; serialize them.
    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Starts a one-shot HTTP server on an ephemeral port that responds to the
    /// first request with `reply` and returns the raw request bytes through the
    /// oneshot so tests can assert method, headers and body.
    pub(crate) async fn spawn_one_shot_json(
        reply: serde_json::Value,
    ) -> (String, tokio::sync::oneshot::Receiver<Vec<u8>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            // Read until end of headers.
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0, "client closed request early");
                request.extend_from_slice(&buf[..n]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            // Read the rest of the declared body.
            let header_end = request
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| p + 4)
                .unwrap();
            let content_length = String::from_utf8_lossy(&request[..header_end])
                .lines()
                .find_map(|line| {
                    let line = line.to_ascii_lowercase();
                    line.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while request.len() < header_end + content_length {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                request.extend_from_slice(&buf[..n]);
            }
            let body = serde_json::to_vec(&reply).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
            socket.flush().await.unwrap();
            let _ = tx.send(request);
        });
        (format!("http://{addr}"), rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_debug_never_exposes_api_keys() {
        let secret = "supersecret-key-DEBUG-LEAK";
        for rendered in [
            format!("{:?}", TavilyBackend::new(secret)),
            format!("{:?}", SerperBackend::new(secret)),
            format!("{:?}", ExaBackend::new(secret)),
        ] {
            assert!(
                !rendered.contains(secret),
                "key leaked via Debug: {rendered}"
            );
            assert!(rendered.contains("<redacted>"), "{rendered}");
        }
    }

    #[test]
    fn canonical_url_strips_fragment_and_trailing_slash() {
        assert_eq!(
            canonical_url("HTTPS://Example.COM/path/#section"),
            "https://example.com/path"
        );
        assert_eq!(
            canonical_url("https://a.example/x?b=1#frag"),
            "https://a.example/x?b=1"
        );
        // Root path keeps its slash (url crate normalizes "" back to "/").
        assert_eq!(canonical_url("http://b.example/"), "http://b.example/");
        // Garbage still lowercases, so dedup is case-insensitive.
        assert_eq!(canonical_url("  NOT-A-URL  "), "not-a-url");
    }

    #[test]
    fn rank_orders_by_score_and_dedupes_url() {
        let provider = "test";
        let mk = |url: &str, score: f64| SearchResult {
            title: url.to_string(),
            url: url.to_string(),
            snippet: String::new(),
            score,
            published_date: None,
            author: None,
            provider,
        };
        let mut results = vec![
            mk("https://a.example/p", 0.2),
            mk("https://a.example/p#frag", 0.9), // same canonical URL, higher score
            mk("https://b.example/", 0.5),
        ];
        rank_results(&mut results);
        assert_eq!(results.len(), 2, "fragment dup must collapse");
        assert_eq!(results[0].url, "https://a.example/p#frag");
        assert!(results[0].score > results[1].score);
    }

    /// Records the `top_k` the tool dispatched, so validation/clamping is proven
    /// without any HTTP.
    struct RecordingBackend {
        seen_k: tokio::sync::Mutex<Option<usize>>,
    }
    #[async_trait]
    impl SearchBackend for RecordingBackend {
        fn label(&self) -> &'static str {
            "recording"
        }
        async fn search(
            &self,
            _query: &str,
            top_k: usize,
            _include_answer: bool,
        ) -> Result<BackendResponse, ToolError> {
            *self.seen_k.lock().await = Some(top_k);
            Ok(BackendResponse::default())
        }
    }

    /// Minimal current-thread block_on (the crate pulls tokio full anyway).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(fut)
    }

    #[test]
    fn empty_query_is_rejected_before_backend() {
        let tool = HostedSearchTool::new(RecordingBackend {
            seen_k: tokio::sync::Mutex::new(None),
        });
        let err = block_on(tool.search("   ", None, None)).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn top_k_is_clamped_before_dispatch() {
        let backend = Arc::new(RecordingBackend {
            seen_k: tokio::sync::Mutex::new(None),
        });
        let tool = HostedSearchTool::from_arc(backend.clone() as Arc<dyn SearchBackend>);
        block_on(tool.search("q", Some(999), None)).unwrap();
        assert_eq!(block_on(backend.seen_k.lock()).as_ref(), Some(&MAX_TOP_K));

        let backend2 = Arc::new(RecordingBackend {
            seen_k: tokio::sync::Mutex::new(None),
        });
        let tool2 = HostedSearchTool::from_arc(backend2.clone() as Arc<dyn SearchBackend>);
        block_on(tool2.search("q", Some(0), None)).unwrap();
        assert_eq!(block_on(backend2.seen_k.lock()).as_ref(), Some(&1));
    }

    #[test]
    fn tool_metadata_is_provider_named() {
        let tool = HostedSearchTool::tavily("tvly-test");
        assert_eq!(BaseTool::name(&tool), "tavily_search");
        assert!(tool.description().contains("Tavily"));
        assert!(BaseTool::args_schema(&tool).is_some());
        assert_eq!(tool.provider(), "tavily");
    }

    #[test]
    fn render_includes_answer_scores_and_empty_state() {
        let output = SearchOutput {
            query: "q".into(),
            results: vec![SearchResult {
                title: "T".into(),
                url: "https://x.example".into(),
                snippet: "S".into(),
                score: 0.91,
                published_date: Some("2026-01-02".into()),
                author: Some("A".into()),
                provider: "tavily",
            }],
            answer: Some("synth".into()),
            provider: "tavily",
        };
        let text = render_text(&output);
        assert!(text.contains("综合答案: synth"));
        assert!(text.contains("相关度: 0.91"));
        assert!(text.contains("2026-01-02 · A"));

        let empty = SearchOutput {
            query: "q".into(),
            results: vec![],
            answer: None,
            provider: "serper",
        };
        assert!(render_text(&empty).contains("未找到相关结果"));
    }
}
