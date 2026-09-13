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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
        };
        browser.enable_domains().await?;
        Ok(browser)
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
    async fn goto(&self, url: &str, wait_ms: Option<u64>) -> Result<(), ToolError> {
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
                } else if value.get("method").and_then(|v| v.as_str())
                    == Some("Page.loadEventFired")
                    && got_response
                {
                    return Ok(());
                }
            }
        };
        match tokio::time::timeout(LOAD_TIMEOUT, fut).await {
            Ok(result) => result?,
            Err(_) => {
                // Some pages keep long-polling open and never fire "load": accept
                // a committed navigation once the deadline passes.
            }
        }
        if let Some(ms) = wait_ms {
            tokio::time::sleep(Duration::from_millis(ms.min(MAX_WAIT_MS))).await;
        }
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
        Ok(Self {
            browser: Arc::new(CdpBrowser::connect(debugger_url).await?),
            allow_private_urls: false,
        })
    }

    /// Wraps an already-connected browser (shares one tab across callers).
    pub fn from_browser(browser: Arc<CdpBrowser>) -> Self {
        Self {
            browser,
            allow_private_urls: false,
        }
    }

    /// Allows navigating to private/loopback/link-local addresses (SSRF opt-in).
    pub fn with_allow_private_urls(mut self, allow: bool) -> Self {
        self.allow_private_urls = allow;
        self
    }

    async fn validate_target(&self, url: &str) -> Result<(), ToolError> {
        validate_navigation_url(url, self.allow_private_urls).await
    }
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
        match input.operation.as_str() {
            "navigate" => {
                let info = self.browser.navigate(&input.url, input.wait_ms).await?;
                Ok(BrowserOutput {
                    operation: "navigate".into(),
                    url: info.url,
                    title: Some(info.title),
                    text: None,
                    links: None,
                    description: None,
                })
            }
            "extract_text" => {
                let text = self.browser.extract_text(&input.url, input.wait_ms).await?;
                Ok(BrowserOutput {
                    operation: "extract_text".into(),
                    url: input.url,
                    title: None,
                    text: Some(text),
                    links: None,
                    description: None,
                })
            }
            "extract_links" => {
                let links = self.browser.extract_links(&input.url, input.wait_ms).await?;
                Ok(BrowserOutput {
                    operation: "extract_links".into(),
                    url: input.url,
                    title: None,
                    text: None,
                    links: Some(links),
                    description: None,
                })
            }
            "metadata" => {
                let meta = self.browser.metadata(&input.url, input.wait_ms).await?;
                Ok(BrowserOutput {
                    operation: "metadata".into(),
                    url: meta.url,
                    title: Some(meta.title),
                    text: None,
                    links: None,
                    description: Some(meta.description),
                })
            }
            other => Err(ToolError::InvalidInput(format!(
                "unsupported operation: {other}, use: navigate, extract_text, extract_links, metadata"
            ))),
        }
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
