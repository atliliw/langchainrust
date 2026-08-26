//! MCP Client - 连接 MCP Server,获取并调用工具

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::{broadcast, Mutex};
use tokio::time::{timeout, Duration};

use super::protocol::{
    MCPError, MCPRequest, ProtocolInfo, VersionPolicy, MCP_VERSION, SUPPORTED_PROTOCOL_VERSIONS,
};
use super::stream::{parse_partial_notification, PartialContent, ToolStream};
use super::tool_adapter::MCPToolAdapter;
use super::transport::{MCPEvent, MCPTransport, SseTransport, StdioTransport};
use super::types::{MCPConfig, MCPToolDefinition, MCPToolResult};
use lc_core::BaseTool;

/// Default timeout for MCP client connection and initialization (30 seconds).
const MCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// 工具列表缓存 TTL:超过该时长视为过期,下一次 `as_tools` 重新拉取(P1-8)。
const TOOLS_CACHE_TTL: Duration = Duration::from_secs(60);

struct MCPClientInner {
    transport: Box<dyn MCPTransport + Send + Sync>,
    tools: Mutex<Vec<MCPToolDefinition>>,
    /// 最近一次成功拉取工具列表的时间;`None` = 从未拉取或已失效(P1-8)。
    tools_fetched_at: Mutex<Option<Instant>>,
    request_id: AtomicU64,
    /// 流式工具输出广播(P2-9):后台事件监听把 `notifications/tool_partial`
    /// 转成 [`PartialContent`] 广播进来,`subscribe_tool_stream` 按工具名过滤。
    partial_tx: broadcast::Sender<PartialContent>,
    /// 协议版本协商策略(P2-10):Server 版本不支持时降级或拒绝。
    version_policy: VersionPolicy,
    /// 握手锁定后的协议版本协商结果(P2-10);未握手为 `None`。
    protocol_info: RwLock<Option<ProtocolInfo>>,
}

/// MCP Client - 连接 MCP Server 获取工具能力
pub struct MCPClient {
    inner: Arc<MCPClientInner>,
}

impl Clone for MCPClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl MCPClient {
    /// 连接 MCP Server
    ///
    /// 建立传输层连接后,发送 MCP `initialize` 握手请求,
    /// 再发送 `notifications/initialized` 通知,完成协议握手。
    /// 协议版本协商采用默认的降级策略([`VersionPolicy::Degrade`]):Server
    /// 版本不受支持时降级到本实现版本继续用。严格模式见 `connect_with_policy`。
    pub async fn connect(config: MCPConfig) -> Result<Self, MCPError> {
        Self::connect_with_policy(config, VersionPolicy::Degrade).await
    }

    /// 指定版本协商策略连接(P2-10)。
    ///
    /// `VersionPolicy::Reject` 时,Server 声明版本不在支持列表内直接握手失败、
    /// 拒绝连接,不静默降级。
    pub async fn connect_with_policy(
        config: MCPConfig,
        policy: VersionPolicy,
    ) -> Result<Self, MCPError> {
        // Wrap the entire connect + initialize flow in a timeout
        timeout(MCP_CONNECT_TIMEOUT, Self::connect_inner(config, policy))
            .await
            .map_err(|_| {
                MCPError::new(
                    -1,
                    "MCP connect timeout: handshake not completed within 30 seconds",
                )
            })?
    }

    async fn connect_inner(config: MCPConfig, policy: VersionPolicy) -> Result<Self, MCPError> {
        let transport: Box<dyn MCPTransport + Send + Sync> = match &config {
            MCPConfig::Stdio { .. } => Box::new(StdioTransport::new(&config).await?),
            MCPConfig::Sse { .. } => Box::new(SseTransport::new(&config)?),
        };
        Self::with_transport_policy(transport, policy).await
    }

    /// 用自定义传输构造客户端,并完成握手(P2-6)。
    ///
    /// 进程内嵌入场景用:比如把 [`InMemoryTransport`](crate::InMemoryTransport)
    /// 直接接到本地 `MCPServer` 上,不走子进程 / 网络;或注入自定义传输层。
    /// 与 `connect` 一样完成 `reconnect → initialize 握手 → 监听推送`,调用方
    /// 拿到即用。
    pub async fn with_transport(
        transport: Box<dyn MCPTransport + Send + Sync>,
    ) -> Result<Self, MCPError> {
        Self::with_transport_policy(transport, VersionPolicy::Degrade).await
    }

    /// 指定版本协商策略的自定义传输构造(P2-10)。
    ///
    /// 与 `with_transport` 相同,但用给定策略协商协议版本。
    pub async fn with_transport_policy(
        transport: Box<dyn MCPTransport + Send + Sync>,
        policy: VersionPolicy,
    ) -> Result<Self, MCPError> {
        let client = Self {
            inner: Arc::new(MCPClientInner {
                transport,
                tools: Mutex::new(Vec::new()),
                tools_fetched_at: Mutex::new(None),
                request_id: AtomicU64::new(1),
                partial_tx: broadcast::channel(64).0,
                version_policy: policy,
                protocol_info: RwLock::new(None),
            }),
        };

        // 先建立传输层连接(SSE 惰性启动后台读循环 / Stdio 子进程已就绪 /
        // 进程内传输已就绪),避免首次 initialize 请求与连接建立竞争。
        client.inner.transport.reconnect().await?;
        // MCP 协议握手: initialize + initialized 通知(协商协议版本)
        client.handshake().await?;
        // P1-8: 后台监听服务器推送,`tools/list_changed` 时失效工具缓存。
        client.spawn_event_listener();

        Ok(client)
    }

    fn next_id(&self) -> u64 {
        self.inner.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// MCP 协议握手: initialize + initialized 通知,并协商协议版本(P2-10)。
    ///
    /// 请求版本为本库当前实现的 [`MCP_VERSION`];Server 响应的版本在支持列表
    /// 内则采用它,否则按 [`VersionPolicy`](crate::VersionPolicy) 降级或拒绝。
    /// 协商结果锁定到 `protocol_info`,供 `protocol_info()` / `protocol_version()`
    /// 读取。供连接建立与断连重连(重新握手)复用。
    async fn handshake(&self) -> Result<(), MCPError> {
        let requested = MCP_VERSION.to_string();
        let init_params = json!({
            "protocolVersion": requested,
            "capabilities": {},
            "clientInfo": {
                "name": "langchainrust-mcp-client",
                "version": "0.3.0"
            }
        });
        let req = MCPRequest::new(self.next_id(), "initialize", Some(init_params));
        let resp = self.inner.transport.request(req).await?;
        let result = resp.into_result()?;

        // 版本协商:Server 声明版本 ∈ 支持列表 → 采用;否则按策略降级 / 拒绝。
        let server_version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(MCP_VERSION)
            .to_string();
        let supported = SUPPORTED_PROTOCOL_VERSIONS.contains(&server_version.as_str());
        let negotiated = if supported {
            server_version.clone()
        } else {
            match self.inner.version_policy {
                VersionPolicy::Degrade => MCP_VERSION.to_string(),
                VersionPolicy::Reject => {
                    return Err(MCPError::new(
                        -32600,
                        format!("unsupported MCP protocol version: {server_version}"),
                    ));
                }
            }
        };
        *self
            .inner
            .protocol_info
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(ProtocolInfo {
            requested,
            server_version,
            negotiated,
            supported,
        });

        // 发送 initialized 通知(无 id,不等响应)
        self.inner
            .transport
            .notify("notifications/initialized", None)
            .await?;
        Ok(())
    }

    /// 已锁定的协议版本协商结果(P2-10);握手前为 `None`。
    pub fn protocol_info(&self) -> Option<ProtocolInfo> {
        self.inner
            .protocol_info
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// 实际协商生效的协议版本(P2-10);握手前为 `None`。
    pub fn protocol_version(&self) -> Option<String> {
        self.protocol_info().map(|p| p.negotiated)
    }

    async fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value, MCPError> {
        let req = MCPRequest::new(self.next_id(), method, params);
        let resp = self.inner.transport.request(req.clone()).await;
        match resp {
            Ok(r) => r.into_result(),
            Err(e) if e.is_connection_lost() => {
                // P0-2: 连接断开 → 重连 + 重新握手 + 刷新工具缓存 + 重试一次
                log::warn!(
                    "MCP connection lost, reconnecting and re-handshaking before retry for {}",
                    req.method
                );
                self.reconnect_and_retry(req).await
            }
            Err(e) => Err(e),
        }
    }

    /// 断连恢复:等待传输层重连完成 → 重新握手 → 清空工具缓存(子进程
    /// 重启后工具列表可能变化) → 重试一次原请求。
    async fn reconnect_and_retry(&self, req: MCPRequest) -> Result<Value, MCPError> {
        self.inner.transport.reconnect().await?;
        self.handshake().await?;
        self.invalidate_tools_cache().await;
        let resp = self.inner.transport.request(req).await?;
        resp.into_result()
    }

    /// 失效工具缓存:清空列表并重置拉取时间(P1-8)。
    ///
    /// 下次 `as_tools` / `list_tools` 会重新拉取,不会继续使用过时列表。
    /// 调用方均为 async 上下文(tokio Mutex),await 持锁短暂完成。
    async fn invalidate_tools_cache(&self) {
        *self.inner.tools.lock().await = Vec::new();
        *self.inner.tools_fetched_at.lock().await = None;
    }

    /// P1-8 后台监听:订阅传输层推送事件,`tools/list_changed` 或断连时
    /// 失效工具缓存,避免 agent 使用过时的工具列表。
    fn spawn_event_listener(&self) {
        let client = self.clone();
        let mut rx = self.inner.transport.subscribe_events();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(MCPEvent::Message { method, params })
                        if method == "notifications/tool_partial" =>
                    {
                        // P2-9 流式工具输出:服务器边跑边推增量片段,转发给
                        // subscribe_tool_stream 的订阅者;畸形片段静默丢弃。
                        if let Some(partial) = parse_partial_notification(params) {
                            let _ = client.inner.partial_tx.send(partial);
                        }
                    }
                    Ok(MCPEvent::Message { method, .. })
                        if method == "notifications/tools/list_changed" =>
                    {
                        log::info!("received tools/list_changed, invalidating tool cache");
                        client.invalidate_tools_cache().await;
                    }
                    Ok(MCPEvent::Disconnected) => {
                        // 断连后工具列表可能已变化,交由重连握手路径重新拉取
                        client.invalidate_tools_cache().await;
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // 事件积压被丢弃:保守失效,避免使用过时列表
                        client.invalidate_tools_cache().await;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    /// 获取可用工具列表(`tools/list`)
    pub async fn list_tools(&self) -> Result<Vec<MCPToolDefinition>, MCPError> {
        let result = self.send_request("tools/list", None).await?;
        let tools_value = result
            .get("tools")
            .ok_or_else(|| MCPError::new(-1, "tools/list response missing 'tools' field"))?;
        let tools: Vec<MCPToolDefinition> = serde_json::from_value(tools_value.clone())
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool list: {}", e)))?;
        *self.inner.tools.lock().await = tools.clone();
        // P1-8: 记录拉取时间,供 as_tools 判断缓存是否新鲜。
        *self.inner.tools_fetched_at.lock().await = Some(Instant::now());
        Ok(tools)
    }

    /// 调用工具(`tools/call`)
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<MCPToolResult, MCPError> {
        let params = json!({"name": name, "arguments": arguments});
        let result = self.send_request("tools/call", Some(params)).await?;
        serde_json::from_value(result)
            .map_err(|e| MCPError::new(-1, format!("failed to parse tool result: {}", e)))
    }

    /// 关闭连接
    pub async fn close(&self) -> Result<(), MCPError> {
        self.inner.transport.close().await
    }

    /// 订阅传输层推送事件(P2-4)。
    ///
    /// 供 per-tool 超时(P2-4)、流式工具输出(P2-9)等场景监听服务器推送:
    /// `notifications/progress` / 断连 / 工具变更等。
    pub fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
        self.inner.transport.subscribe_events()
    }

    /// 订阅某个工具的流式增量输出(P2-9)。
    ///
    /// 长任务工具"边跑边推":服务器把部分结果拆成多个片段,经
    /// `notifications/tool_partial` 推送。返回的 [`ToolStream`] 只投递
    /// 属于 `tool` 的增量,其他工具的推送被过滤;配合 P1-7,每个片段
    /// 携带独立 [`MCPContent`](crate::MCPContent)(文本 / 图片 / 资源)。
    ///
    /// 注意:广播通道只投递给"订阅时刻之后"的推送,应**先订阅再调用工具**;
    /// 推送过快导致丢帧时 [`ToolStream::next`] 返回
    /// [`ToolStreamError::Lagged`](crate::ToolStreamError::Lagged)。
    pub fn subscribe_tool_stream(&self, tool: &str) -> ToolStream {
        ToolStream::new(self.inner.partial_tx.subscribe(), tool.to_string())
    }

    /// 转换为 `BaseTool` 列表(供 Agent 使用)
    ///
    /// P0-3: 自动发现 —— 若工具缓存为空则自动调用 `list_tools`,
    /// 不再静默返回空列表;以 `Result` 显式暴露失败。
    ///
    /// P1-8: 缓存带 TTL —— 空缓存或超过 `TOOLS_CACHE_TTL` 未刷新时重新拉取;
    /// 收到 `tools/list_changed` 通知(后台监听)或断连后缓存即时失效。
    pub async fn as_tools(&self) -> Result<Vec<Arc<dyn BaseTool>>, MCPError> {
        let tools = {
            let cached = self.inner.tools.lock().await;
            let fetched = self.inner.tools_fetched_at.lock().await;
            let fresh = matches!(fetched.as_ref(), Some(t) if t.elapsed() < TOOLS_CACHE_TTL);
            if cached.is_empty() || !fresh {
                drop(cached);
                drop(fetched);
                self.list_tools().await?
            } else {
                cached.clone()
            }
        };
        Ok(tools
            .into_iter()
            .map(|def| Arc::new(MCPToolAdapter::new(self.clone(), def)) as Arc<dyn BaseTool>)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::MCPResponse;
    use crate::test_support::{start_fake_sse_server, PostMode};

    #[tokio::test]
    async fn test_connect_invalid_stdio_command() {
        let config = MCPConfig::stdio("nonexistent_command_xyz_zzz", vec![]);
        let result = MCPClient::connect(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_as_tools_uses_cache_within_ttl() {
        // P1-8:Quiet 模式服务器不发 list_changed → TTL 内第二次 as_tools 命中缓存,
        // 不再调用 tools/list(tools_list_count 保持 1)。
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let tools = client
            .as_tools()
            .await
            .expect("initial fetch should succeed");
        assert_eq!(tools.len(), 1);
        assert_eq!(server.tools_list_count.load(Ordering::SeqCst), 1);

        let tools2 = client.as_tools().await.expect("cache hit should succeed");
        assert_eq!(tools2.len(), 1);
        assert_eq!(
            server.tools_list_count.load(Ordering::SeqCst),
            1,
            "cache hit within TTL, tools/list should not be called again"
        );
    }

    #[tokio::test]
    async fn test_tools_cache_invalidated_on_list_changed() {
        // P1-8:服务器推送 tools/list_changed → 后台监听失效缓存 → 下一次
        // as_tools 重新拉取(tools_list_count >= 2)。轮询等待,避免时序抖动。
        let server = start_fake_sse_server(PostMode::NotifyChanged).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let tools = client
            .as_tools()
            .await
            .expect("initial fetch should succeed");
        assert_eq!(tools.len(), 1);

        timeout(Duration::from_secs(5), async {
            loop {
                // 每次轮询都会走 as_tools:缓存被失效后触发重新拉取
                let _ = client.as_tools().await;
                if server.tools_list_count.load(Ordering::SeqCst) >= 2 {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("list_changed should invalidate cache and refetch tools/list");
    }

    #[tokio::test]
    async fn test_connect_list_and_call_tool_via_sse_push_responses() {
        // F4 验收:服务器对 POST 回 202、JSON-RPC 响应经 SSE `event: message`
        // 推送 → connect(initialize)/list_tools/call_tool 全部走通。POST body
        // 恒为空,结果只能来自 SSE 推送,成功即证明按 id 关联的推送路径生效。
        let server = start_fake_sse_server(PostMode::PushResponse).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connect (initialize) should succeed via SSE-pushed response");

        let tools = client
            .list_tools()
            .await
            .expect("list_tools should succeed via SSE-pushed response");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = client
            .call_tool("echo", json!({"msg": "hi"}))
            .await
            .expect("call_tool should succeed via SSE-pushed response");
        assert!(!result.is_error, "server tool should not error");
    }

    // ---- P2-10 协议版本协商测试 ----

    /// 版本协商测试用 stub 传输:initialize 返回可配置的 `protocolVersion`,
    /// 其余方法空实现,避免引入 SSE 时序与真实服务器。
    struct StubVersionTransport {
        server_version: String,
    }

    #[async_trait::async_trait]
    impl MCPTransport for StubVersionTransport {
        async fn request(&self, req: MCPRequest) -> Result<MCPResponse, MCPError> {
            if req.method == "initialize" {
                return Ok(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(req.id),
                    result: Some(json!({
                        "protocolVersion": self.server_version,
                        "capabilities": {},
                        "serverInfo": { "name": "stub", "version": "0.0.1" }
                    })),
                    error: None,
                });
            }
            Ok(MCPResponse {
                jsonrpc: "2.0".to_string(),
                id: Some(req.id),
                result: Some(json!({})),
                error: None,
            })
        }
        async fn notify(&self, _m: &str, _p: Option<Value>) -> Result<(), MCPError> {
            Ok(())
        }
        async fn close(&self) -> Result<(), MCPError> {
            Ok(())
        }
        fn is_connected(&self) -> bool {
            true
        }
        async fn reconnect(&self) -> Result<(), MCPError> {
            Ok(())
        }
        fn subscribe_events(&self) -> broadcast::Receiver<MCPEvent> {
            broadcast::channel(4).1
        }
    }

    /// 支持列表内版本:协商成功,supported=true,锁定版本为请求版本。
    #[tokio::test]
    async fn test_handshake_negotiates_supported_version() {
        let client = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: MCP_VERSION.to_string(),
            }),
            VersionPolicy::Degrade,
        )
        .await
        .expect("supported version should handshake successfully");

        let info = client
            .protocol_info()
            .expect("negotiation result should be available after handshake");
        assert!(info.supported, "2024-11-05 should be in the supported list");
        assert_eq!(info.negotiated, MCP_VERSION);
        assert_eq!(info.server_version, MCP_VERSION);
        assert_eq!(client.protocol_version().as_deref(), Some(MCP_VERSION));
    }

    /// 降级策略:Server 声明未知版本 → supported=false,降级到本实现版本继续用。
    #[tokio::test]
    async fn test_handshake_degrades_unsupported_version() {
        let client = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: "2099-01-01".to_string(),
            }),
            VersionPolicy::Degrade,
        )
        .await
        .expect("degrade policy should tolerate unknown version");

        let info = client
            .protocol_info()
            .expect("negotiation result should be available after handshake");
        assert!(
            !info.supported,
            "2099-01-01 should not be in the supported list"
        );
        assert_eq!(info.server_version, "2099-01-01");
        assert_eq!(
            info.negotiated, MCP_VERSION,
            "should lock to this implementation version after degrade"
        );
        assert_eq!(client.protocol_version().as_deref(), Some(MCP_VERSION));
    }

    /// 拒绝策略:Server 声明未知版本 → 握手失败,拒绝连接。
    #[tokio::test]
    async fn test_handshake_rejects_unsupported_version() {
        let err = MCPClient::with_transport_policy(
            Box::new(StubVersionTransport {
                server_version: "2099-01-01".to_string(),
            }),
            VersionPolicy::Reject,
        )
        .await
        .err()
        .expect("reject policy should error on unknown version");

        assert!(
            err.to_string().contains("unsupported MCP protocol version"),
            "error message should state the version is unsupported: {err}"
        );
    }
}
