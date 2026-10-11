// lc-callbacks/src/langsmith_client.rs
//! LangSmith API client

use reqwest::Client;
use std::env;
use std::time::Duration;

/// M-18: a trace POST/PATCH must never hang the agent forever. Observability
/// is a side channel — a slow or dead endpoint degrades to a logged error
/// instead of blocking the run indefinitely. The caller still sees the error
/// and, if the handler runs in the sync (default) mode, is only delayed by
/// this bound rather than stalled.
const LANGSMITH_TIMEOUT: Duration = Duration::from_secs(15);

/// §4 lc-callbacks: bounded retry attempts for a trace submission (first try +
/// retries) so a transient failure (429/5xx/connect) self-heals without turning
/// observability into an unbounded stall on a dead endpoint.
const LANGSMITH_MAX_ATTEMPTS: usize = 3;
/// Base backoff between attempts — doubled each attempt (150ms → 300ms) for
/// exponential spacing, still far under the LANGSMITH_TIMEOUT budget per send.
const LANGSMITH_BACKOFF_BASE_MS: u64 = 150;

use super::run_tree::{RunCreate, RunTree, RunUpdate};

/// LangSmith configuration
#[derive(Debug, Clone)]
pub struct LangSmithConfig {
    /// API key (starts with "ls_")
    pub api_key: String,

    /// API endpoint URL
    pub api_url: String,

    /// Workspace ID (required for org accounts)
    pub workspace_id: Option<String>,

    /// Project name
    pub project_name: String,

    /// Whether tracing is enabled
    pub tracing_enabled: bool,
}

impl LangSmithConfig {
    /// Create config from environment variables
    pub fn from_env() -> Result<Self, LangSmithError> {
        let api_key = env::var("LANGSMITH_API_KEY").map_err(|_| {
            LangSmithError::Config("LANGSMITH_API_KEY environment variable not set".to_string())
        })?;

        let tracing_enabled = env::var("LANGSMITH_TRACING")
            .map(|v| v == "true")
            .unwrap_or(true);

        let project_name = env::var("LANGSMITH_PROJECT").unwrap_or_else(|_| "default".to_string());

        let api_url = env::var("LANGSMITH_ENDPOINT")
            .unwrap_or_else(|_| "https://api.smith.langchain.com".to_string());

        let workspace_id = env::var("LANGSMITH_WORKSPACE_ID").ok();

        Ok(Self {
            api_key,
            api_url,
            workspace_id,
            project_name,
            tracing_enabled,
        })
    }

    /// Create config with API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_url: "https://api.smith.langchain.com".to_string(),
            workspace_id: None,
            project_name: "default".to_string(),
            tracing_enabled: true,
        }
    }

    /// Set project name
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project_name = project.into();
        self
    }

    /// Set workspace ID
    pub fn with_workspace(mut self, workspace_id: impl Into<String>) -> Self {
        self.workspace_id = Some(workspace_id.into());
        self
    }

    /// Set API endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.api_url = endpoint.into();
        self
    }

    /// Enable or disable tracing
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }
}

/// LangSmith API client
pub struct LangSmithClient {
    /// Configuration
    pub config: LangSmithConfig,
    http_client: Client,
}

impl LangSmithClient {
    /// Create a new client
    pub fn new(config: LangSmithConfig) -> Self {
        // M-18: `Client::new()` sets no request timeout, so a hung endpoint
        // blocks the `.send().await` forever. Bound the whole request cycle
        // (connect + send + response body) so the agent degrades to an error.
        let http_client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(LANGSMITH_TIMEOUT)
            .build()
            // A builder failure here is a programming error (invalid TLS / too
            // many redirects config), not a runtime condition — a plain reqwest
            // client from valid static settings never fails to build. Falling
            // back keeps `new()` infallible.
            .unwrap_or_else(|_| Client::new());
        Self {
            config,
            http_client,
        }
    }

    /// Create client from environment variables
    pub fn from_env() -> Result<Self, LangSmithError> {
        let config = LangSmithConfig::from_env()?;
        Ok(Self::new(config))
    }

    /// Check if tracing is enabled
    pub fn is_tracing_enabled(&self) -> bool {
        self.config.tracing_enabled
    }

    /// Get the project name
    pub fn project_name(&self) -> &str {
        &self.config.project_name
    }

    /// Create a run (POST /runs)
    pub async fn create_run(&self, run: &RunTree) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled {
            return Ok(());
        }

        let url = format!("{}/runs", self.config.api_url);
        let mut run_create = RunCreate::from(run);
        if run_create.session_name.is_none() {
            run_create.session_name = Some(self.config.project_name.clone());
        }
        let body = run_create;

        send_with_retry(
            || self.http_client.post(&url).json(&body),
            &self.config.api_key,
            self.config.workspace_id.as_deref(),
        )
        .await
    }

    /// Update a run (PATCH /runs/{run_id})
    pub async fn update_run(&self, run: &RunTree) -> Result<(), LangSmithError> {
        if !self.config.tracing_enabled {
            return Ok(());
        }

        let url = format!("{}/runs/{}", self.config.api_url, run.id);
        let body = RunUpdate::from(run);

        send_with_retry(
            || self.http_client.patch(&url).json(&body),
            &self.config.api_key,
            self.config.workspace_id.as_deref(),
        )
        .await
    }
}

/// Sends a LangSmith trace HTTP request with bounded exponential-backoff retry
/// on transient failures.
///
/// Only a subset of failures is retried: a connection/timeout transport error or
/// a 429/5xx response may have been transient (rate limit, blip, upstream), so a
/// brief backoff followed by a re-send is worthwhile. A 4xx is a contract error
/// (bad key, bad payload) and is returned immediately — retrying it would only
/// repeat a guaranteed failure. Every attempt carries the full
/// `LANGSMITH_TIMEOUT` budget, so the capped attempt count bounds total stall.
async fn send_with_retry(
    build: impl Fn() -> reqwest::RequestBuilder + Send + Sync,
    api_key: &str,
    workspace_id: Option<&str>,
) -> Result<(), LangSmithError> {
    let mut last: Option<LangSmithError> = None;
    for attempt in 0..LANGSMITH_MAX_ATTEMPTS {
        let mut request = build();
        request = request.header("x-api-key", api_key);
        if let Some(wid) = workspace_id {
            request = request.header("x-tenant-id", wid);
        }

        match request.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(());
                }
                let retryable = status.as_u16() == 429 || status.is_server_error();
                if retryable && attempt + 1 < LANGSMITH_MAX_ATTEMPTS {
                    last = Some(LangSmithError::Api(format!("HTTP {status}")));
                    retry_backoff(attempt).await;
                    continue;
                }
                let body = resp.text().await.unwrap_or_default();
                return Err(LangSmithError::Api(format!("HTTP {}: {}", status, body)));
            }
            Err(err) if err.is_timeout() || err.is_connect() => {
                last = Some(LangSmithError::Http(err.to_string()));
                if attempt + 1 < LANGSMITH_MAX_ATTEMPTS {
                    retry_backoff(attempt).await;
                    continue;
                }
                return Err(LangSmithError::Http(err.to_string()));
            }
            Err(err) => return Err(LangSmithError::Http(err.to_string())),
        }
    }
    Err(last.unwrap_or_else(|| LangSmithError::Api("retry exhausted".to_string())))
}

/// Exponential backoff: `150ms << attempt` (150ms, 300ms).
async fn retry_backoff(attempt: usize) {
    tokio::time::sleep(Duration::from_millis(LANGSMITH_BACKOFF_BASE_MS << attempt)).await;
}

/// LangSmith error type
#[derive(Debug)]
#[non_exhaustive]
pub enum LangSmithError {
    /// HTTP request error.
    Http(String),
    /// API returned an error.
    Api(String),
    /// Configuration error.
    Config(String),
}

impl std::fmt::Display for LangSmithError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(msg) => write!(f, "HTTP error: {}", msg),
            Self::Api(msg) => write!(f, "API error: {}", msg),
            Self::Config(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for LangSmithError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Spawns a tiny raw-HTTP server that answers each accepted connection with
    /// one status from `statuses` (in order), counting requests. Each response
    /// carries `Connection: close`, so reqwest opens a fresh socket per attempt —
    /// request count == connection count, which lets a test assert how many times
    /// `send_with_retry` actually hit the wire.
    async fn spawn_status_server(statuses: Vec<u16>) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let count2 = Arc::clone(&count);
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            for status in statuses {
                let (mut sock, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await; // drain the request headers+body
                count2.fetch_add(1, Ordering::SeqCst);
                let reason = if status >= 500 {
                    "Internal Server Error"
                } else {
                    "OK"
                };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                let _ = sock.write_all(resp.as_bytes()).await;
            }
        });
        (format!("http://{addr}"), count)
    }

    fn client(base_url: &str) -> LangSmithClient {
        LangSmithClient::new(
            LangSmithConfig::new("ls_test")
                .with_endpoint(base_url.to_string())
                .with_project("p"),
        )
    }

    fn run() -> RunTree {
        RunTree::new("r", crate::RunType::Llm, serde_json::json!({}))
    }

    #[tokio::test]
    async fn retries_transient_5xx_then_succeeds() {
        // §4: 5xx then 5xx then 200 → three wire hits, final call succeeds.
        let (base, count) = spawn_status_server(vec![503, 503, 200]).await;
        let client = client(&base);
        client
            .create_run(&run())
            .await
            .expect("succeeds after retries");
        assert_eq!(
            count.load(Ordering::SeqCst),
            3,
            "5xx should retry up to the cap"
        );
    }

    #[tokio::test]
    async fn retries_raced_rate_limit_until_success() {
        // 429 then 200 → two hits.
        let (base, count) = spawn_status_server(vec![429, 200]).await;
        let client = client(&base);
        client
            .update_run(&run())
            .await
            .expect("429 retried then succeeded");
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_a_4xx_contract_error() {
        // §4: a 4xx is a contract error (bad key/payload) — no blind retry.
        let (base, count) = spawn_status_server(vec![400]).await;
        let client = client(&base);
        let err = client
            .create_run(&run())
            .await
            .expect_err("HTTP 400 must be an error");
        assert!(err.to_string().contains("400"), "got: {err}");
        assert_eq!(count.load(Ordering::SeqCst), 1, "4xx must not be retried");
    }

    #[tokio::test]
    async fn caps_attempts_on_persistent_5xx() {
        // §4: the attempt cap bounds total stall — 5xx never reaching success
        // still returns an error after LANGSMITH_MAX_ATTEMPTS wire hits.
        let (base, count) = spawn_status_server(vec![500, 500, 500]).await;
        let client = client(&base);
        assert!(client.create_run(&run()).await.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 3, "attempts capped");
    }
}
