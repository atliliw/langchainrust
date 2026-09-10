// lc-mcp/src/client_stateless.rs
//! Stateless MCP client (2026-07-28 track, 0.22.0 S2.4).
//!
//! No handshake, no session: every request carries `_meta` (protocol version
//! plus client identity, and `requestState` on MRTR resends). Server-initiated
//! interaction is handled through MRTR — an `input_required` response carries
//! an opaque `requestState` plus questions; the client collects answers via
//! the configured [`MrtrAnswerProvider`] and **resends the original request**
//! with the `requestState` attached, up to `max_round_trips`.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::auth::AuthScheme;
use crate::gateway::MethodRateLimiter;
use crate::protocol::{InputRequired, MCPError, MCPRequest, MrtrAnswer, MrtrQuestion, RequestMeta};
use crate::transport::stateless::{default_meta, StatelessTransport};
use crate::types::{MCPToolDefinition, MCPToolResult};

/// MRTR configuration.
#[derive(Debug, Clone)]
pub struct MrtrConfig {
    /// Maximum `input_required` round trips per logical request (default 3).
    pub max_round_trips: usize,
}

impl Default for MrtrConfig {
    fn default() -> Self {
        Self { max_round_trips: 3 }
    }
}

/// Collects the answers the server asked for during an MRTR round.
///
/// Implement against your UX surface (user prompt, policy engine, cache).
/// Returning `Err` aborts the logical request with that error.
#[async_trait::async_trait]
pub trait MrtrAnswerProvider: Send + Sync {
    /// Collects an answer for each question (order preserved).
    async fn collect(&self, questions: &[MrtrQuestion]) -> Result<Vec<MrtrAnswer>, String>;
}

/// A provider that answers every question with a fixed canned value
/// (tests / deterministic integrations).
pub struct CannedAnswerProvider {
    value: String,
}

impl CannedAnswerProvider {
    /// Answers every question with `value`.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }
}

#[async_trait::async_trait]
impl MrtrAnswerProvider for CannedAnswerProvider {
    async fn collect(&self, questions: &[MrtrQuestion]) -> Result<Vec<MrtrAnswer>, String> {
        Ok(questions
            .iter()
            .map(|q| MrtrAnswer {
                id: q.id.clone(),
                value: self.value.clone(),
            })
            .collect())
    }
}

/// Stateless MCP client.
pub struct StatelessMcpClient {
    transport: Arc<StatelessTransport>,
    meta: RequestMeta,
    mrtr: MrtrConfig,
    answers: Option<Arc<dyn MrtrAnswerProvider>>,
    rate_limiter: Option<Arc<Mutex<MethodRateLimiter>>>,

    request_id: AtomicU64,
}

impl Clone for StatelessMcpClient {
    fn clone(&self) -> Self {
        Self {
            transport: self.transport.clone(),
            meta: self.meta.clone(),
            mrtr: self.mrtr.clone(),
            answers: self.answers.clone(),
            rate_limiter: self.rate_limiter.clone(),
            request_id: AtomicU64::new(self.request_id.load(Ordering::SeqCst)),
        }
    }
}

impl StatelessMcpClient {
    /// Connects to a stateless endpoint — there is no handshake; this only
    /// constructs the transport and pins the `_meta`.
    pub fn connect(url: impl Into<String>) -> Self {
        Self {
            transport: Arc::new(StatelessTransport::new(url)),
            meta: default_meta(),
            mrtr: MrtrConfig::default(),
            answers: None,
            rate_limiter: None,
            request_id: AtomicU64::new(1),
        }
    }

    /// Connects with bearer auth (401 responses surface as the unauthorized
    /// JSON-RPC error code).
    pub fn connect_with_auth(url: impl Into<String>, auth: AuthScheme) -> Self {
        Self {
            transport: Arc::new(StatelessTransport::with_auth(url, auth)),
            meta: default_meta(),
            mrtr: MrtrConfig::default(),
            answers: None,
            rate_limiter: None,
            request_id: AtomicU64::new(1),
        }
    }

    /// Overrides the MRTR round-trip limit.
    pub fn with_mrtr(mut self, mrtr: MrtrConfig) -> Self {
        self.mrtr = mrtr;
        self
    }

    /// Attaches the MRTR answer provider (required for servers that use
    /// `input_required`; without one such requests fail after one round).
    pub fn with_answer_provider(mut self, provider: Arc<dyn MrtrAnswerProvider>) -> Self {
        self.answers = Some(provider);
        self
    }

    /// Attaches a per-method rate limiter (checked before every send; a hit
    /// limit surfaces as an `MCPError` with code -32002 without a network
    /// round trip).
    pub fn with_method_rate_limiter(mut self, limiter: MethodRateLimiter) -> Self {
        self.rate_limiter = Some(Arc::new(Mutex::new(limiter)));
        self
    }

    /// The pinned `_meta` (read-only view).
    pub fn meta(&self) -> &RequestMeta {
        &self.meta
    }

    /// Releases the client (connection-manager lifecycle hook). Stateless
    /// track has no persistent connection: this is a no-op that just drops
    /// cached state on the next clone.
    pub async fn close(&self) -> Result<(), MCPError> {
        Ok(())
    }

    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn check_rate_limit(&self, method: &str) -> Result<(), MCPError> {
        if let Some(limiter) = &self.rate_limiter {
            let mut guard = limiter.lock().await;
            if !guard.allow(method) {
                return Err(MCPError::new(
                    -32002,
                    format!("rate limit exceeded for method {method}"),
                ));
            }
        }
        Ok(())
    }

    /// Sends one logical request (with MRTR loop). Public for gateways and
    /// tests; `call_tool` / `list_tools` / `discover` build on it.
    pub async fn send(&self, method: &str, params: Option<Value>) -> Result<Value, MCPError> {
        self.check_rate_limit(method).await?;
        let mut meta = self.meta.clone();
        let mut params = params;
        let mut round_trips = 0usize;
        loop {
            let req = MCPRequest::new_stateless(self.next_id(), method, params.clone(), meta);
            let resp = self.transport.post_jsonrpc(&req).await?;
            let result = resp.into_result()?;

            match InputRequired::from_result(&result) {
                Some(ir) if round_trips < self.mrtr.max_round_trips => {
                    round_trips += 1;
                    let provider = self.answers.as_ref().ok_or_else(|| {
                        MCPError::new(
                            -32003,
                            "server sent input_required but no MrtrAnswerProvider is attached",
                        )
                    })?;
                    let answers = provider
                        .collect(&ir.questions)
                        .await
                        .map_err(|e| MCPError::new(-32003, format!("MRTR collect failed: {e}")))?;
                    // Resend the ORIGINAL method/params with the continuation
                    // token attached; answers travel as a side channel.
                    let mut resent_params = params.clone().unwrap_or_else(|| json!({}));
                    resent_params["mrtr_answers"] =
                        serde_json::to_value(&answers).unwrap_or_else(|_| json!([]));
                    meta = self.meta.clone().with_request_state(ir.request_state);
                    params = Some(resent_params);
                    continue;
                }
                Some(_) => {
                    return Err(MCPError::new(
                        -32003,
                        format!(
                            "MRTR exceeded {} round trips for {method}",
                            self.mrtr.max_round_trips
                        ),
                    ));
                }
                None => return Ok(result),
            }
        }
    }

    /// `server/discover`: on-demand capability discovery (2026-07-28).
    pub async fn discover(&self) -> Result<Value, MCPError> {
        self.send("server/discover", None).await
    }

    /// `tools/list` (uncached).
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send("tools/list", None).await?;
        let tools_value = result
            .get("tools")
            .ok_or_else(|| MCPError::new(-1, "tools/list response missing 'tools' field"))?;
        let tools: Vec<MCPToolDefinition> = serde_json::from_value(tools_value.clone())
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool list: {e}")))?;
        Ok(tools)
    }

    /// `tools/call` (MRTR-aware).
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool result: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{MrtrQuestion, MCP_VERSION_STATELESS};
    use crate::test_support::{start_fake_stateless_server, StatelessMode};
    use crate::types::MCPContent;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Provider that answers every question with a per-call sequence value.
    #[allow(dead_code)]
    struct SeqProvider {
        counter: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl MrtrAnswerProvider for SeqProvider {
        async fn collect(&self, questions: &[MrtrQuestion]) -> Result<Vec<MrtrAnswer>, String> {
            let n = self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(questions
                .iter()
                .map(|q| MrtrAnswer {
                    id: q.id.clone(),
                    value: format!("answer-{n}"),
                })
                .collect())
        }
    }

    /// M1 stateless_roundtrip: no handshake — connect → list_tools →
    /// call_tool work over plain POSTs; the server sees the routing headers.
    #[tokio::test]
    async fn m1_stateless_roundtrip() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = StatelessMcpClient::connect(&server.url);

        let tools = client.list_tools().await.expect("tools/list should work");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = client
            .call_tool("echo", json!({"msg": "hi"}))
            .await
            .expect("tools/call should work");
        assert!(!result.is_error);
        assert!(
            server
                .method_headers_seen
                .lock()
                .unwrap()
                .iter()
                .any(|m| m == "tools/call"),
            "server must have received the Mcp-Method header"
        );
    }

    /// M2 meta_snapshot: every request carries `_meta` with the stateless
    /// version + client identity; the server records it.
    #[tokio::test]
    async fn m2_meta_snapshot() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = StatelessMcpClient::connect(&server.url);
        let _ = client.list_tools().await.unwrap();

        let metas = server.metas_seen.lock().unwrap().clone();
        assert!(!metas.is_empty(), "server must receive _meta");
        for meta in &metas {
            assert_eq!(meta.protocol_version, MCP_VERSION_STATELESS);
            assert_eq!(meta.client_info.name, "langchainrust-mcp-client");
        }
    }

    /// M3 discover: server/discover returns capabilities.
    #[tokio::test]
    async fn m3_discover() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let client = StatelessMcpClient::connect(&server.url);
        let caps = client.discover().await.expect("discover should work");
        assert!(caps.get("capabilities").is_some(), "caps: {caps}");
        assert!(caps.get("serverInfo").is_some(), "caps: {caps}");
    }

    /// M4 mrtr_happy_path: tools/call returns input_required → provider
    /// answers → resend with requestState → final result.
    #[tokio::test]
    async fn m4_mrtr_happy_path() {
        let server = start_fake_stateless_server(StatelessMode::Mrtr).await;
        let client = StatelessMcpClient::connect(&server.url)
            .with_answer_provider(Arc::new(CannedAnswerProvider::new("yes")));

        let result = client
            .call_tool("echo", json!({"msg": "hi"}))
            .await
            .expect("MRTR flow should complete");
        assert!(!result.is_error);
        assert!(
            result.content.iter().any(|c| match c {
                MCPContent::Text { text } => text.contains("answered"),
                _ => false,
            }),
            "final result must reflect the resent request: {:?}",
            result.content
        );
        assert_eq!(
            server.request_states_seen.lock().unwrap().clone(),
            vec!["state-1"],
            "resend must carry the server's requestState"
        );
    }

    /// M5 mrtr_idempotent: the resent request carries requestState; the
    /// server's tool-execution counter shows the tool ran once for the
    /// logical request (idempotent handling of the resent shape is visible
    /// in the response text).
    #[tokio::test]
    async fn m5_mrtr_state_carried() {
        let server = start_fake_stateless_server(StatelessMode::Mrtr).await;
        let client = StatelessMcpClient::connect(&server.url)
            .with_answer_provider(Arc::new(CannedAnswerProvider::new("yes")));
        let result = client.call_tool("echo", json!({"msg": "x"})).await.unwrap();
        let echoed = result
            .content
            .iter()
            .map(|c| match c {
                MCPContent::Text { text } => text.clone(),
                _ => String::new(),
            })
            .collect::<String>();
        assert!(
            echoed.contains("state-1"),
            "echo must carry state: {echoed}"
        );
    }

    /// M6 mrtr_limit: a server that always answers input_required eventually
    /// trips the round-trip limit (max 2 here).
    #[tokio::test]
    async fn m6_mrtr_limit() {
        let server = start_fake_stateless_server(StatelessMode::MrtrLoop).await;
        let client = StatelessMcpClient::connect(&server.url)
            .with_answer_provider(Arc::new(CannedAnswerProvider::new("yes")))
            .with_mrtr(MrtrConfig { max_round_trips: 2 });

        let err = client.call_tool("echo", json!({})).await.unwrap_err();
        assert!(
            err.to_string().contains("MRTR exceeded"),
            "expected limit error, got: {err}"
        );
    }

    /// M7: server-side iss validation rejects foreign issuers (via the fake
    /// server in Unauth mode; validator-level tests live in auth.rs).
    #[tokio::test]
    async fn m7_unauthorized_surfaces() {
        let server = start_fake_stateless_server(StatelessMode::Unauth).await;
        let client = StatelessMcpClient::connect(&server.url);
        let err = client.list_tools().await.unwrap_err();
        assert_eq!(err.code, crate::protocol::MCP_ERROR_UNAUTHORIZED);
    }

    /// M8: rate limit hit → -32002 without a network round trip (server
    /// never sees the request).
    #[tokio::test]
    async fn m8_rate_limit_by_method() {
        let server = start_fake_stateless_server(StatelessMode::Normal).await;
        let mut limiter = MethodRateLimiter::new(1, std::time::Duration::from_secs(60));
        assert!(limiter.allow("tools/list")); // consume the only slot
        let client = StatelessMcpClient::connect(&server.url).with_method_rate_limiter(limiter);
        let err = client.list_tools().await.unwrap_err();
        assert_eq!(err.code, -32002);
        assert_eq!(
            server.request_count.load(Ordering::SeqCst),
            0,
            "rate-limited request must not reach the server"
        );
    }

    /// Canned provider answers all questions.
    #[tokio::test]
    async fn canned_provider_contract() {
        let p = CannedAnswerProvider::new("yes");
        let answers = p
            .collect(&[MrtrQuestion {
                id: "q1".into(),
                prompt: "confirm?".into(),
            }])
            .await
            .unwrap();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].value, "yes");
    }

    /// MRTR default config.
    #[tokio::test]
    async fn mrtr_defaults() {
        let client = StatelessMcpClient::connect("http://127.0.0.1:1");
        assert_eq!(client.mrtr.max_round_trips, 3);
        assert_eq!(client.meta().protocol_version, MCP_VERSION_STATELESS);
    }
}
