// lc-tools/src/browser/mod.rs
//! Browser tool over the Chrome DevTools Protocol (B6, v0.22.4).
//!
//! Unlike plain HTTP fetching ([`crate::URLFetchTool`]), a real browser renders
//! JavaScript, waits for SPA navigation and exposes the post-load DOM. This
//! module drives a **local** Chromium/Chrome debugger (also used by
//! Playwright-managed browsers) over CDP:
//!
//! 1. Start Chrome with `--remote-debugging-port=9222
//!    --remote-allow-origins=*` (the allow-origins flag is required by Chrome
//!    ≥111 for non-browser CDP clients), or launch a Playwright browser and
//!    point at its per-run `webSocketDebuggerUrl` HTTP endpoint.
//! 2. [`CdpBrowserTool::connect("http://127.0.0.1:9222")`](CdpBrowserTool::connect)
//!    opens a fresh tab (`PUT /json/new`) and connects to its CDP WebSocket.
//!
//! Only plain `ws://` is used (the debugger is local), so enabling the
//! `browser-cdp` feature pulls in `tokio-tungstenite` without any TLS stack.
//!
//! ```no_run
//! # async fn demo() -> Result<(), lc_tools::ToolError> {
//! use lc_tools::browser::CdpBrowserTool;
//! use lc_tools::BaseTool;
//! let tool = CdpBrowserTool::connect("http://127.0.0.1:9222").await?;
//! let page = tool
//!     .run(r#"{"operation":"extract_text","url":"https://example.com"}"#.to_string())
//!     .await?;
//! # let _ = page;
//! # Ok(()) }
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::ssrf::url_points_to_private_ip;
use lc_core::tools::{BaseTool, Tool, ToolError};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Per-command CDP deadline.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(15);
/// Deadline for `Page.loadEventFired` after `Page.navigate`.
const LOAD_TIMEOUT: Duration = Duration::from_secs(20);
/// Caller-supplied extra settle time is capped to keep tool runs bounded.
const MAX_WAIT_MS: u64 = 10_000;
/// Extracted page text is truncated to this many characters.
const TEXT_LIMIT: usize = 20_000;
/// At most this many links are returned per page.
const LINK_LIMIT: usize = 200;

/// A CDP connection attached to one browser tab.
///
/// Cheaply cloneable via `Arc` via [`CdpBrowserTool::from_browser`]; all calls
/// are serialized through an internal mutex (CDP sessions multiplex by command
/// id, but serial use keeps event correlation trivially correct for tools).
pub struct CdpBrowser {
    ws: tokio::sync::Mutex<Ws>,
    next_id: AtomicU64,
    /// Reads as the SSRF default during per-hop interception (V0.27, #200).
    allow_private_urls: AtomicBool,
}

impl std::fmt::Debug for CdpBrowser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CdpBrowser").finish_non_exhaustive()
    }
}

impl CdpBrowser {
    /// Connects to a Chrome debugger HTTP endpoint and opens a fresh tab.
    ///
    /// `debugger_url` is e.g. `http://127.0.0.1:9222` (no trailing slash
    /// required). Tries `PUT /json/new?about:blank`; when the build forbids tab
    /// creation it falls back to the first existing page target from
    /// `GET /json`.
    pub async fn connect(debugger_url: impl AsRef<str>) -> Result<Self, ToolError> {
        let base = debugger_url.as_ref().trim().trim_end_matches('/');
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(ToolError::InvalidInput(
                "debugger URL must start with http:// or https://".to_string(),
            ));
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            // SSRF hardening: never follow redirects on the CDP control endpoints —
            // a redirect can bounce to an internal address and leak into the browser's
            // navigation. Same policy as lc-rag's guarded fetch path.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| ToolError::ExecutionFailed(format!("CDP HTTP client error: {e}")))?;

        // Chrome ≥136 requires PUT; older builds accepted POST. Try PUT, then
        // fall back to attaching to an existing page.
        let target = http
            .put(format!(
                "{base}/json/new?{}",
                urlencoding::encode("about:blank")
            ))
            .send()
            .await
            .ok();
        let ws_url = match target {
            Some(resp) if resp.status().is_success() => {
                let body: serde_json::Value = resp.json().await.map_err(|e| {
                    ToolError::ExecutionFailed(format!("CDP /json/new parse failed: {e}"))
                })?;
                body.get("webSocketDebuggerUrl")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ToolError::ExecutionFailed(
                            "CDP /json/new response lacks webSocketDebuggerUrl".to_string(),
                        )
                    })?
                    .to_string()
            }
            _ => {
                // Fallback: reuse an existing page target.
                let resp = http.get(format!("{base}/json")).send().await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "CDP discovery failed (PUT /json/new unavailable): {e}"
                    ))
                })?;
                let targets: serde_json::Value = resp.json().await.map_err(|e| {
                    ToolError::ExecutionFailed(format!("CDP /json parse failed: {e}"))
                })?;
                targets
                    .as_array()
                    .and_then(|arr| {
                        arr.iter().find(|t| {
                            t.get("type").and_then(|v| v.as_str()) == Some("page")
                                && t.get("webSocketDebuggerUrl")
                                    .and_then(|v| v.as_str())
                                    .is_some()
                        })
                    })
                    .and_then(|t| {
                        t.get("webSocketDebuggerUrl")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    })
                    .ok_or_else(|| {
                        ToolError::ExecutionFailed(
                            "CDP endpoint exposes no usable page target (start Chrome with \
                             --remote-debugging-port=... --remote-allow-origins=*)"
                                .to_string(),
                        )
                    })?
            }
        };

        let (ws, _) = connect_async(ws_url).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("CDP websocket connect failed: {e}"))
        })?;
        let browser = Self {
            ws: tokio::sync::Mutex::new(ws),
            next_id: AtomicU64::new(0),
            allow_private_urls: AtomicBool::new(false),
        };
        browser.enable_domains().await?;
        Ok(browser)
    }

    /// Opt in/out of navigating to private/loopback addresses (V0.27, #200).
    pub fn set_allow_private_urls(&self, allow: bool) {
        self.allow_private_urls.store(allow, Ordering::Relaxed);
    }

    /// Whether this browser currently permits private addresses.
    pub fn allow_private(&self) -> bool {
        self.allow_private_urls.load(Ordering::Relaxed)
    }

    async fn enable_domains(&self) -> Result<(), ToolError> {
        self.command("Page.enable", serde_json::json!({})).await?;
        self.command("Runtime.enable", serde_json::json!({}))
            .await?;
        Ok(())
    }

    /// Sends a command and waits for its matching response, discarding events.
    async fn command(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let fut = async {
            let mut ws = self.ws.lock().await;
            let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
            let envelope = serde_json::json!({"id": id, "method": method, "params": params});
            ws.send(Message::Text(envelope.to_string().into()))
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("CDP send failed: {e}")))?;
            loop {
                let msg = ws
                    .next()
                    .await
                    .ok_or_else(|| ToolError::ExecutionFailed("CDP websocket closed".to_string()))?
                    .map_err(|e| ToolError::ExecutionFailed(format!("CDP read failed: {e}")))?;
                let Message::Text(text) = msg else { continue };
                let value: serde_json::Value =
                    serde_json::from_str(text.as_str()).map_err(|e| {
                        ToolError::ExecutionFailed(format!("CDP frame parse failed: {e}"))
                    })?;
                if value.get("id").and_then(|v| v.as_u64()) != Some(id) {
                    continue; // event for an earlier command
                }
                if let Some(error) = value.get("error") {
                    return Err(ToolError::ExecutionFailed(format!(
                        "CDP {method} error: {}",
                        error
                    )));
                }
                return Ok(value
                    .get("result")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null));
            }
        };
        tokio::time::timeout(COMMAND_TIMEOUT, fut)
            .await
            .map_err(|_| ToolError::ExecutionFailed(format!("CDP {method} timed out")))?
    }

    /// Navigates and waits for the load event (bounded by [`LOAD_TIMEOUT`]),
    /// then optionally settles for `wait_ms` more.
    ///
    /// While navigating, per-hop SSRF interception is armed (V0.27, #200): the
    /// `Fetch` CDP domain pauses every request before it leaves, and each
    /// `Fetch.requestPaused` is validated — private hops are failed, public ones
    /// continued — so a top-level check that passed cannot be followed later by
    /// a redirect, DNS-rebound, or subresource request to an internal host.
    ///
    /// With `wait_ms`, interception stays armed through that settle window too
    /// (R-7) and every hop fired inside it is validated, so a `setTimeout`-scheduled
    /// navigation that lands post-load is still caught. It is disarmed once the
    /// load *and* settle window are over (or on any failure) so a later
    /// `Runtime.evaluate` is never stranded on an unanswered pause.
    async fn goto(&self, url: &str, wait_ms: Option<u64>) -> Result<(), ToolError> {
        self.command(
            "Fetch.enable",
            serde_json::json!({"patterns": [{"urlPattern": "*"}]}),
        )
        .await?;
        let fut = async {
            let mut ws = self.ws.lock().await;
            let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
            let envelope = serde_json::json!({
                "id": id,
                "method": "Page.navigate",
                "params": {"url": url},
            });
            ws.send(Message::Text(envelope.to_string().into()))
                .await
                .map_err(|e| ToolError::ExecutionFailed(format!("CDP send failed: {e}")))?;

            let mut got_response = false;
            loop {
                let msg = ws
                    .next()
                    .await
                    .ok_or_else(|| ToolError::ExecutionFailed("CDP websocket closed".to_string()))?
                    .map_err(|e| ToolError::ExecutionFailed(format!("CDP read failed: {e}")))?;
                let Message::Text(text) = msg else { continue };
                let value: serde_json::Value =
                    serde_json::from_str(text.as_str()).map_err(|e| {
                        ToolError::ExecutionFailed(format!("CDP frame parse failed: {e}"))
                    })?;
                if let Some(method) = value.get("method").and_then(|v| v.as_str()) {
                    // V0.27 (#200): a request hop was paused *before* it goes
                    // out — validate its URL now and continue or fail it on the
                    // same connection.
                    if method == "Fetch.requestPaused" {
                        self.resolve_hop(&mut *ws, &value).await?;
                        continue;
                    }
                    if method == "Page.loadEventFired" && got_response {
                        return Ok(());
                    }
                }
                if value.get("id").and_then(|v| v.as_u64()) == Some(id) {
                    if let Some(error) = value.get("error") {
                        return Err(ToolError::ExecutionFailed(format!(
                            "CDP Page.navigate error: {error}"
                        )));
                    }
                    // A loader-error result (DNS fail, bad scheme) reports errorText.
                    if let Some(error_text) =
                        value.pointer("/result/errorText").and_then(|v| v.as_str())
                    {
                        return Err(ToolError::ExecutionFailed(format!(
                            "navigation failed: {error_text}"
                        )));
                    }
                    got_response = true;
                }
            }
        };
        let result = match tokio::time::timeout(LOAD_TIMEOUT, fut).await {
            Ok(result) => result,
            Err(_) => {
                // Some pages keep long-polling open and never fire "load": accept
                // a committed navigation once the deadline passes.
                Ok(())
            }
        };
        // R-7: keep `Fetch` interception armed through the caller's settle window
        // and actively validate every hop fired inside it (a `setTimeout`-scheduled
        // navigation that lands post-load used to fall outside the SSRF net). We
        // cannot leave interception on beyond `goto` — no reader would answer the
        // pauses — so this closes the window the tool itself observes, then the
        // disarm below restores the documented outer boundary.
        if let Some(ms) = wait_ms {
            let settle_ms = ms.min(MAX_WAIT_MS);
            let _ = tokio::time::timeout(Duration::from_millis(settle_ms + 1_000), async {
                let mut ws = self.ws.lock().await;
                let deadline = Instant::now() + Duration::from_millis(settle_ms);
                while Instant::now() < deadline {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    let step = tokio::time::timeout(remaining, ws.next())
                        .await
                        // The settle window elapsed with no further traffic.
                        .unwrap_or(None);
                    let Some(Ok(msg)) = step else {
                        // Stream ended / read error — traversal is over.
                        break;
                    };
                    let Message::Text(text) = msg else { continue };
                    let Ok(value) =
                        serde_json::from_str::<serde_json::Value>(text.as_str())
                    else {
                        continue;
                    };
                    if value.get("method").and_then(|v| v.as_str())
                        == Some("Fetch.requestPaused")
                    {
                        // Ignore a failed hop reply here: the only realistic failure is the
                        // websocket send itself (i.e. the connection is already gone), and the
                        // settle window is hard-bounded above regardless.
                        let _ = self.resolve_hop(&mut *ws, &value).await;
                    }
                }
            })
            .await;
        }
        // Disarm interception on every exit path — including timeout and error —
        // so a later `Runtime.evaluate` is not stalled on pending pauses.
        let _ = self
            .command("Fetch.disable", serde_json::json!({}))
            .await;
        result?;
        Ok(())
    }

    /// Validate one `Fetch.requestPaused` hop against the SSRF policy and continue or fail it
    /// in place (V0.27 #200). Shared by the top-level navigation loop and the post-load settle
    /// drain (R-7) so both use the same decision rule.
    async fn resolve_hop(
        &self,
        ws: &mut Ws,
        value: &serde_json::Value,
    ) -> Result<(), ToolError> {
        let request_id = value
            .pointer("/params/requestId")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let request_url = value
            .pointer("/params/request/url")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let allow_private = self.allow_private_urls.load(Ordering::Relaxed);
        let block = block_request_hop(request_url, allow_private).await?;
        let (hop_method, hop_params) = if block {
            // Deny the request before any packet reaches the net.
            (
                "Fetch.failRequest",
                serde_json::json!({
                    "requestId": request_id,
                    "errorReason": "BlockedByClient",
                }),
            )
        } else {
            ("Fetch.continueRequest", serde_json::json!({ "requestId": request_id }))
        };
        let hop_id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        ws.send(Message::Text(
            serde_json::json!({
                "id": hop_id,
                "method": hop_method,
                "params": hop_params,
            })
            .to_string()
            .into(),
        ))
        .await
        .map_err(|e| ToolError::ExecutionFailed(format!("CDP send failed: {e}")))?;
        Ok(())
    }

    /// Runs a JS expression, requiring a JSON-serializable by-value result.
    async fn evaluate(&self, expression: &str) -> Result<serde_json::Value, ToolError> {
        let result = self
            .command(
                "Runtime.evaluate",
                serde_json::json!({
                    "expression": expression,
                    "returnByValue": true,
                }),
            )
            .await?;
        if let Some(details) = result.get("exceptionDetails") {
            return Err(ToolError::ExecutionFailed(format!(
                "page script failed: {}",
                details
                    .pointer("/exception/description")
                    .or_else(|| details.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown exception>")
            )));
        }
        Ok(result
            .pointer("/result/value")
            .cloned()
            .unwrap_or(serde_json::Value::Null))
    }

    async fn current_title(&self) -> Result<String, ToolError> {
        Ok(self
            .evaluate("document.title")
            .await?
            .as_str()
            .unwrap_or_default()
            .to_string())
    }

    /// Loads `url`, returning the final page title.
    pub async fn navigate(
        &self,
        url: &str,
        wait_ms: Option<u64>,
    ) -> Result<NavigationInfo, ToolError> {
        self.goto(url, wait_ms).await?;
        Ok(NavigationInfo {
            url: url.to_string(),
            title: self.current_title().await?,
        })
    }

    /// Loads `url` and extracts visible text (`document.body.innerText`).
    pub async fn extract_text(&self, url: &str, wait_ms: Option<u64>) -> Result<String, ToolError> {
        self.goto(url, wait_ms).await?;
        let text = self
            .evaluate("(document.body && document.body.innerText || '')")
            .await?;
        let text = text.as_str().unwrap_or_default();
        if text.chars().count() > TEXT_LIMIT {
            Ok(format!(
                "{}\n... [内容已截断]",
                text.chars().take(TEXT_LIMIT).collect::<String>()
            ))
        } else {
            Ok(text.to_string())
        }
    }

    /// Loads `url` and extracts anchor links (text + absolute href).
    pub async fn extract_links(
        &self,
        url: &str,
        wait_ms: Option<u64>,
    ) -> Result<Vec<PageLink>, ToolError> {
        self.goto(url, wait_ms).await?;
        let expression = format!(
            "JSON.stringify(Array.from(document.querySelectorAll('a[href]'))\
             .slice(0,{LINK_LIMIT})\
             .map(a => ({{text: (a.innerText || a.textContent || '').trim().slice(0,200), \
             href: a.href}})))"
        );
        let value = self.evaluate(&expression).await?;
        parse_jsonish(value).map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to parse page links JSON: {e}"))
        })
    }

    /// Loads `url` and returns title / meta description / final URL.
    pub async fn metadata(
        &self,
        url: &str,
        wait_ms: Option<u64>,
    ) -> Result<PageMetadata, ToolError> {
        self.goto(url, wait_ms).await?;
        // Unquoted attribute value is a valid CSS ident selector, which keeps
        // the Rust/JS quoting trivial.
        let expression = r#"JSON.stringify({title: document.title, description: (document.querySelector('meta[name=description]') || {}).content || '', url: location.href})"#;
        let value = self.evaluate(expression).await?;
        parse_jsonish(value).map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to parse page metadata JSON: {e}"))
        })
    }

    /// The URL the page actually landed on (after HTTP / client-side redirects
    /// and load-time meta-refresh), read via `location.href`.
    async fn current_url(&self) -> Result<String, ToolError> {
        let value = self.evaluate("location.href").await?;
        Ok(value.as_str().map(str::to_string).unwrap_or_default())
    }
}

/// Deserializes a by-value CDP result. Pages return these as JSON strings
/// (`JSON.stringify(...)`); accept both the stringified form and a direct value
/// so non-browser fakes/CDP shims work too.
fn parse_jsonish<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
) -> Result<T, serde_json::Error> {
    match value {
        serde_json::Value::String(s) => serde_json::from_str(&s),
        other => serde_json::from_value(other),
    }
}

/// Result of the `navigate` operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationInfo {
    /// Requested URL.
    pub url: String,
    /// Page title after load.
    pub title: String,
}

/// One extracted anchor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PageLink {
    /// Link text (trimmed, max 200 chars).
    pub text: String,
    /// Absolute href as resolved by the browser.
    pub href: String,
}

/// Page metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageMetadata {
    /// `<title>`.
    pub title: String,
    /// Meta description content, empty when absent.
    pub description: String,
    /// Final URL after client-side/HTTP redirects.
    pub url: String,
}

/// Tool input for the CDP browser.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BrowserInput {
    /// Operation: `navigate`, `extract_text`, `extract_links`, or `metadata`.
    pub operation: String,
    /// The page URL (http/https). Intranet addresses are blocked unless the
    /// tool was built with `with_allow_private_urls(true)`.
    pub url: String,
    /// Extra settle time after the load event, in milliseconds (max 10000).
    pub wait_ms: Option<u64>,
}

/// Typed output of the CDP browser tool.
#[derive(Debug, Serialize)]
pub struct BrowserOutput {
    /// Operation that ran.
    pub operation: String,
    /// Page URL.
    pub url: String,
    /// Title (navigate/metadata), if the operation produced one.
    pub title: Option<String>,
    /// Extracted text (`extract_text`).
    pub text: Option<String>,
    /// Extracted links (`extract_links`).
    pub links: Option<Vec<PageLink>>,
    /// Meta description (`metadata`).
    pub description: Option<String>,
}

/// Browser tool driving a local Chrome via CDP.
///
/// Private/intranet navigation targets are blocked by default (same SSRF
/// posture as [`crate::URLFetchTool`]); opt in with
/// [`CdpBrowserTool::with_allow_private_urls`].
pub struct CdpBrowserTool {
    browser: Arc<CdpBrowser>,
    allow_private_urls: bool,
}

impl std::fmt::Debug for CdpBrowserTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CdpBrowserTool")
            .field("allow_private_urls", &self.allow_private_urls)
            .finish_non_exhaustive()
    }
}

impl CdpBrowserTool {
    /// Opens a fresh tab against the given Chrome debugger endpoint.
    pub async fn connect(debugger_url: impl AsRef<str>) -> Result<Self, ToolError> {
        let browser = Arc::new(CdpBrowser::connect(debugger_url).await?);
        let allow_private_urls = browser.allow_private();
        Ok(Self {
            browser,
            allow_private_urls,
        })
    }

    /// Wraps an already-connected browser (shares one tab across callers).
    pub fn from_browser(browser: Arc<CdpBrowser>) -> Self {
        let allow_private_urls = browser.allow_private();
        Self {
            browser,
            allow_private_urls,
        }
    }

    /// Allows navigating to private/loopback/link-local addresses (SSRF opt-in).
    ///
    /// Mirrors the decision onto the shared [`CdpBrowser`] so per-hop
    /// interception (V0.27, #200) honors the same setting: public-by-default,
    /// and every request Chrome issues is validated against it.
    pub fn with_allow_private_urls(mut self, allow: bool) -> Self {
        self.browser.set_allow_private_urls(allow);
        self.allow_private_urls = allow;
        self
    }

    async fn validate_target(&self, url: &str) -> Result<(), ToolError> {
        validate_navigation_url(url, self.allow_private_urls).await
    }
}

/// Decide whether one request hop must be blocked by per-hop SSRF interception
/// (V0.27, #200).
///
/// Returns `true` when `allow_private` is off and `url` resolves to a
/// private/loopback/link-local address. A private opt-in, or a non-network URL
/// (data:/blob:/about:), is never blocked here — such requests carry no SSRF
/// vector. Shared by the `Fetch.requestPaused` handler and its tests.
async fn block_request_hop(url: &str, allow_private: bool) -> Result<bool, ToolError> {
    if allow_private
        || url.is_empty()
        || !(url.starts_with("http://") || url.starts_with("https://"))
    {
        return Ok(false);
    }
    url_points_to_private_ip(url).await
}

/// Scheme + SSRF validation shared by the tool and tests.
async fn validate_navigation_url(url: &str, allow_private: bool) -> Result<(), ToolError> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ToolError::InvalidInput(
            "URL must start with http:// or https://".to_string(),
        ));
    }
    if !allow_private && url_points_to_private_ip(url).await? {
        return Err(ToolError::ExecutionFailed(format!(
            "SSRF protection blocked navigation to private/internal address: {url}"
        )));
    }
    Ok(())
}

#[async_trait]
impl Tool for CdpBrowserTool {
    type Input = BrowserInput;
    type Output = BrowserOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        self.validate_target(&input.url).await?;
        let output = match input.operation.as_str() {
            "navigate" => {
                let info = self.browser.navigate(&input.url, input.wait_ms).await?;
                BrowserOutput {
                    operation: "navigate".into(),
                    url: info.url,
                    title: Some(info.title),
                    text: None,
                    links: None,
                    description: None,
                }
            }
            "extract_text" => {
                let text = self.browser.extract_text(&input.url, input.wait_ms).await?;
                BrowserOutput {
                    operation: "extract_text".into(),
                    url: input.url,
                    title: None,
                    text: Some(text),
                    links: None,
                    description: None,
                }
            }
            "extract_links" => {
                let links = self.browser.extract_links(&input.url, input.wait_ms).await?;
                BrowserOutput {
                    operation: "extract_links".into(),
                    url: input.url,
                    title: None,
                    text: None,
                    links: Some(links),
                    description: None,
                }
            }
            "metadata" => {
                let meta = self.browser.metadata(&input.url, input.wait_ms).await?;
                BrowserOutput {
                    operation: "metadata".into(),
                    url: meta.url,
                    title: Some(meta.title),
                    text: None,
                    links: None,
                    description: Some(meta.description),
                }
            }
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "unsupported operation: {other}, use: navigate, extract_text, extract_links, metadata"
                )))
            }
        };

        // H5 (per-hop SSRF recheck): the initial URL was validated above, but
        // Chrome may follow HTTP / client-side redirects or a load-time
        // meta-refresh to a private address that the one-time check never saw.
        // Re-read the landed URL and reject it if private. Non-http final URLs
        // (about:blank, data:, blob:) are not a network SSRF vector, so skip.
        let landed = self.browser.current_url().await?;
        if landed.starts_with("http://") || landed.starts_with("https://") {
            validate_navigation_url(&landed, self.allow_private_urls).await?;
        }
        Ok(output)
    }
}

#[async_trait]
impl BaseTool for CdpBrowserTool {
    fn name(&self) -> &str {
        "browser_cdp"
    }

    fn description(&self) -> &str {
        "浏览器工具(Chrome DevTools Protocol,驱动本机 Chrome/Playwright 浏览器,渲染 JS)。\n\n操作类型:\n- navigate: 打开页面,返回加载后的标题\n- extract_text: 打开页面并提取可见正文文本(innerText,最多 20000 字符)\n- extract_links: 打开页面并提取链接(text + 绝对 href,最多 200 条)\n- metadata: 标题 / meta description / 最终 URL\n\n参数:\n- operation: 操作类型\n- url: 页面地址(必须 http/https;默认禁止内网地址)\n- wait_ms: load 事件后额外等待毫秒数(可选,上限 10000)\n\n示例: {\"operation\": \"extract_text\", \"url\": \"https://example.com\", \"wait_ms\": 500}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: BrowserInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse failed: {e}")))?;
        let output = self.invoke(parsed).await?;
        let mut text = format!("URL: {}\n操作: {}\n", output.url, output.operation);
        if let Some(title) = output.title {
            text.push_str(&format!("标题: {title}\n"));
        }
        if let Some(description) = output.description {
            text.push_str(&format!("描述: {description}\n"));
        }
        if let Some(page_text) = output.text {
            text.push_str("\n正文:\n");
            text.push_str(&page_text);
        }
        if let Some(links) = output.links {
            text.push_str(&format!("\n链接(共 {} 条):\n", links.len()));
            for (i, link) in links.iter().enumerate() {
                text.push_str(&format!("{}. {} → {}\n", i + 1, link.text, link.href));
            }
        }
        Ok(text)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        serde_json::to_value(schemars::schema_for!(BrowserInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serves the CDP HTTP discovery endpoints; the WebSocket side is handled
    /// by [`spawn_fake_cdp_ws`].
    async fn spawn_fake_cdp_http(ws_url: String) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 4096];
                let Ok(Ok(n)) =
                    tokio::time::timeout(Duration::from_secs(2), socket.read(&mut buf)).await
                else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let body = if request.starts_with("PUT /json/new") {
                    format!(
                        r#"{{"targetId":"tab1","type":"page","webSocketDebuggerUrl":"{}"}}"#,
                        ws_url.replace('"', "\\\"")
                    )
                } else {
                    format!(
                        r#"[{{"id":"tab1","type":"page","webSocketDebuggerUrl":"{}"}}]"#,
                        ws_url.replace('"', "\\\"")
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }

    /// Minimal fake CDP peer: acks enables/navigate, fires loadEventFired, and
    /// canned Runtime.evaluate results.
    async fn spawn_fake_cdp_ws() -> String {
        use tokio::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            while let Some(Ok(Message::Text(raw))) = ws.next().await {
                let msg: serde_json::Value = match serde_json::from_str(raw.as_str()) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let id = msg.get("id").cloned().unwrap_or(serde_json::Value::Null);
                let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let reply = match method {
                    "Page.enable" | "Runtime.enable" => {
                        serde_json::json!({"id": id, "result": {}})
                    }
                    "Page.navigate" => {
                        let _ = ws
                            .send(Message::Text(
                                serde_json::json!({"id": id, "result": {"frameId": "f1"}})
                                    .to_string()
                                    .into(),
                            ))
                            .await;
                        let _ = ws
                            .send(Message::Text(
                                serde_json::json!({"method": "Page.loadEventFired", "params": {}})
                                    .to_string()
                                    .into(),
                            ))
                            .await;
                        continue;
                    }
                    "Runtime.evaluate" => {
                        let expr = msg
                            .pointer("/params/expression")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let value = if expr.contains("querySelectorAll") {
                            serde_json::json!(r#"[{"text":"A","href":"https://a.example/"}]"#)
                        } else if expr.trim() == "location.href" {
                            // Bare `location.href` (used by the per-hop SSRF
                            // recheck): land on a private address so the
                            // public-initial → private-landed case is exercised.
                            serde_json::json!("http://127.0.0.1/")
                        } else if expr.contains("location.href") {
                            serde_json::json!(
                                r#"{"title":"Meta Title","description":"desc","url":"https://example.test/final"}"#
                            )
                        } else if expr.contains("innerText") {
                            serde_json::json!("hello page text")
                        } else {
                            serde_json::json!("Fake Title")
                        };
                        serde_json::json!({"id": id, "result": {"result": {"type": "string", "value": value}}})
                    }
                    _ => serde_json::json!({"id": id, "result": {}}),
                };
                let _ = ws.send(Message::Text(reply.to_string().into())).await;
            }
        });
        format!("ws://{addr}")
    }

    async fn connect_fake() -> CdpBrowser {
        let ws_url = spawn_fake_cdp_ws().await;
        let http_url = spawn_fake_cdp_http(ws_url).await;
        CdpBrowser::connect(http_url).await.unwrap()
    }

    #[tokio::test]
    async fn navigate_extract_and_metadata_against_fake_cdp() {
        let browser = connect_fake().await;

        let nav = browser
            .navigate("https://example.test/", None)
            .await
            .unwrap();
        assert_eq!(nav.title, "Fake Title");

        let text = browser
            .extract_text("https://example.test/", None)
            .await
            .unwrap();
        assert_eq!(text, "hello page text");

        let links = browser
            .extract_links("https://example.test/", None)
            .await
            .unwrap();
        assert_eq!(
            links,
            vec![PageLink {
                text: "A".into(),
                href: "https://a.example/".into(),
            }]
        );

        let meta = browser
            .metadata("https://example.test/", Some(1))
            .await
            .unwrap();
        assert_eq!(meta.title, "Meta Title");
        assert_eq!(meta.description, "desc");
        assert_eq!(meta.url, "https://example.test/final");
    }

    #[tokio::test]
    async fn public_initial_but_private_landing_is_blocked() {
        // H5 per-hop recheck: the requested URL is public and passes the initial
        // check, but the fake page lands on a loopback address (simulating an
        // HTTP redirect / load-time meta-refresh to an internal host). The
        // post-navigation re-read must block it even though the first check passed.
        // Initial URL is a public IP literal (`1.1.1.1`) so the one-time check
        // passes without a DNS round-trip; the fake then lands on 127.0.0.1.
        let tool = CdpBrowserTool::from_browser(Arc::new(connect_fake().await));
        let input = BrowserInput {
            operation: "navigate".into(),
            url: "http://1.1.1.1:1/".into(),
            wait_ms: None,
        };
        let err = tool.invoke(input).await.unwrap_err();
        assert!(
            err.to_string().contains("SSRF"),
            "expected SSRF block on private landing, got: {err}"
        );

        // Opting into private URLs lets the same (initial public, landed
        // private) navigation through.
        let tool = CdpBrowserTool::from_browser(Arc::new(connect_fake().await))
            .with_allow_private_urls(true);
        let input = BrowserInput {
            operation: "navigate".into(),
            url: "http://1.1.1.1:1/".into(),
            wait_ms: None,
        };
        assert!(tool.invoke(input).await.is_ok());
    }

    #[tokio::test]
    async fn tool_runs_all_operations_and_rejects_unknown() {
        // Loopback target + explicit SSRF opt-in keeps the test offline (the
        // fake CDP peer never actually opens the URL).
        let target = "http://127.0.0.1:1/page";
        let tool = CdpBrowserTool::from_browser(Arc::new(connect_fake().await))
            .with_allow_private_urls(true);
        let out = tool
            .run(format!(
                r#"{{"operation":"extract_links","url":"{target}"}}"#
            ))
            .await
            .unwrap();
        assert!(out.contains("链接(共 1 条)"));
        assert!(out.contains("https://a.example/"));

        let bad_op = tool
            .run(format!(r#"{{"operation":"click","url":"{target}"}}"#))
            .await
            .unwrap_err();
        assert!(bad_op.to_string().contains("unsupported operation"));
    }

    #[tokio::test]
    async fn tool_metadata_is_complete() {
        let tool = CdpBrowserTool::from_browser(Arc::new(connect_fake().await));
        assert_eq!(tool.name(), "browser_cdp");
        assert!(tool.description().contains("extract_text"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }

    #[tokio::test]
    async fn rejects_non_http_scheme_and_private_targets_by_default() {
        let err = validate_navigation_url("file:///etc/passwd", false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("http://"));

        let err = validate_navigation_url("http://127.0.0.1:9222/json", false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SSRF"), "{}", err);

        // Explicit opt-in passes the SSRF gate (scheme still must be http).
        validate_navigation_url("http://127.0.0.1:9222/json", true)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn per_hop_block_request_private_vs_public() {
        use std::net::{IpAddr, Ipv4Addr};
        // Private IP literals are rejected (no DNS round-trip, stays offline).
        assert!(
            block_request_hop("http://127.0.0.1:1/x", false).await.unwrap(),
            "loopback must be blocked"
        );
        assert!(
            block_request_hop("http://192.168.1.10:1/x", false)
                .await
                .unwrap(),
            "private RFC1918 must be blocked"
        );
        assert!(
            block_request_hop("http://10.0.0.5:1/x", false).await.unwrap(),
            "private 10/8 must be blocked"
        );
        // Non-network URLs carry no SSRF vector and are never blocked.
        assert!(!block_request_hop("data:text/plain,hi", false).await.unwrap());
        assert!(!block_request_hop("about:blank", false).await.unwrap());
        assert!(!block_request_hop("", false).await.unwrap());
        // Private opt-in lets a loopback hop through.
        assert!(!block_request_hop("http://127.0.0.1:1/x", true).await.unwrap());
        // A public IP literal is allowed (lc-core considers non-private).
        let ip = IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1));
        assert!(!lc_core::ssrf::is_private_ip(&ip));
    }

    /// A fake CDP ws that, during `Page.navigate`, pauses a **private** hop
/// (`Fetch.requestPaused` for a loopback URL) before firing the load event, and
/// records how the client answered it. The client must `Fetch.failRequest` the
/// hop and never `Fetch.continueRequest`.
async fn spawn_intercepting_cdp_ws() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    use std::sync::Mutex;
    use tokio::net::TcpListener;
    let decisions: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = decisions.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(Message::Text(raw))) = ws.next().await {
            let msg: serde_json::Value = match serde_json::from_str(raw.as_str()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let id = msg.get("id").cloned().unwrap_or(serde_json::Value::Null);
            let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
            let ack = || {
                serde_json::json!({"id": id, "result": {}}).to_string()
            };
            match method {
                "Fetch.enable" | "Fetch.disable" => {
                    let _ = ws.send(Message::Text(ack().into())).await;
                }
                "Fetch.failRequest" | "Fetch.continueRequest" => {
                    recorder.lock().unwrap().push(method.to_string());
                    let _ = ws.send(Message::Text(ack().into())).await;
                }
                "Page.navigate" => {
                    let _ = ws.send(Message::Text(ack().into())).await;
                    // A loopback hop is paused while navigating.
                    let _ = ws
                        .send(Message::Text(
                            serde_json::json!({
                                "method": "Fetch.requestPaused",
                                "params": {
                                    "requestId": "req-private-1",
                                    "request": {"url": "http://127.0.0.1:1/"},
                                }
                            })
                            .to_string()
                            .into(),
                        ))
                        .await;
                    let _ = ws
                        .send(Message::Text(
                            serde_json::json!({"method": "Page.loadEventFired", "params": {}})
                                .to_string()
                                .into(),
                        ))
                        .await;
                }
                _ => {
                    let _ = ws.send(Message::Text(ack().into())).await;
                }
            }
        }
    });
    (format!("ws://{addr}"), decisions)
}

#[tokio::test]
async fn per_hop_interception_sends_failrequest_for_private() {
    // End-to-end over a mock CDP exchange: connect through the fake HTTP
    // discovery, then navigate. The paused loopback hop must be answered with
    // Fetch.failRequest (default SSRF posture), not continueRequest.
    let (ws_url, decisions) = spawn_intercepting_cdp_ws().await;
    let http_url = spawn_fake_cdp_http(ws_url).await;
    let browser = CdpBrowser::connect(http_url).await.unwrap();

    browser.goto("http://1.1.1.1:1/", None).await.unwrap();

    let got = decisions.lock().unwrap();
    assert_eq!(got.as_slice(), &["Fetch.failRequest".to_string()]);

    // Opting into private URLs lets the same loopback hop through.
    let (ws_url, decisions) = spawn_intercepting_cdp_ws().await;
    let http_url = spawn_fake_cdp_http(ws_url).await;
    let browser = CdpBrowser::connect(http_url).await.unwrap();
    browser.set_allow_private_urls(true);

    browser.goto("http://1.1.1.1:1/", None).await.unwrap();

    let got = decisions.lock().unwrap();
    assert_eq!(got.as_slice(), &["Fetch.continueRequest".to_string()]);
}

#[tokio::test]
async fn connect_failure_is_surfaced() {
        let err = CdpBrowser::connect("http://127.0.0.1:1").await.unwrap_err();
        // Either discovery transport failure or a CDP-labeled message.
        let text = err.to_string();
        assert!(
            text.contains("CDP") || text.contains("connection") || text.contains("refused"),
            "{text}"
        );
    }

    #[test]
    fn connect_rejects_non_http_debugger_url() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt.block_on(CdpBrowser::connect("not-a-url")).unwrap_err();
        assert!(err.to_string().contains("http://"));
    }

    /// Live Chrome smoke test; requires a running debugger:
    /// `chrome --remote-debugging-port=9222 --remote-allow-origins=*`.
    /// Enable with `LANGCHAINRUST_TEST_CDP_URL=http://127.0.0.1:9222`.
    #[tokio::test]
    #[ignore = "needs a local Chrome with --remote-debugging-port; set LANGCHAINRUST_TEST_CDP_URL"]
    async fn live_browser_navigation_smoke() {
        let base = std::env::var("LANGCHAINRUST_TEST_CDP_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:9222".to_string());
        let tool = CdpBrowserTool::connect(base).await.unwrap();
        let out = tool
            .run(r#"{"operation":"metadata","url":"https://example.com"}"#.to_string())
            .await
            .unwrap();
        assert!(out.contains("Example Domain"), "{out}");
    }
}
