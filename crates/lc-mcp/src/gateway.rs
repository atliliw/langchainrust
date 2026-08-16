//! MCP Gateway(P2-8):统一入口,内部按需分发。
//!
//! 100+ Server 场景需要对外只暴露一个入口:调用方 `register` 声明所有 Server,
//! 之后按 `server:tool` 全名调用,内部自动路由到对应 Server。本模块把
//! P2-1~P2-6 的能力整合成一个统一工具注册表:
//!
//! - **P2-1** 惰性连接 / 空闲回收 / 连接池([`ConnectionManager`]);
//! - **P2-2** 工具命名空间 + 冲突策略([`ToolNamespace`]);
//! - **P2-3** 静态层 + 动态层工具发现([`ToolDiscovery`]);
//! - **P2-4** per-tool 超时 + Progress 重置([`ToolSpec`]);
//! - **P2-5** 健康检查 + 熔断([`crate::ServerHealth`]);
//! - **P2-6** per-Server 安全沙箱([`ServerSandbox`]);
//! - **速率限制**:每 Server 固定窗口限流([`RateLimiter`]);
//! - **统一审计**:Gateway 入口全量记录放行/拦截([`GatewayAuditRecord`])。
//!
//! # 统一注册表
//!
//! `register` 只登记 Server + 策略(惰性,不建连、不拉工具);`sync` / `sync_all`
//! 才真正连接 Server、拉取 `tools/list` 并命名空间化入注册表。`call` 未命中注册
//! 表时按 `server:tool` 前缀自动 `sync`(按需分发)。
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_mcp::{MCPGateway, GatewayServerSpec, MCPConfig, ToolConflict};
//! use std::time::Duration;
//!
//! let gw = MCPGateway::new();
//! gw.register(
//!     GatewayServerSpec::new("fs", MCPConfig::stdio("npx", vec!["@anthropic/mcp-server-filesystem".into(), "/tmp".into()]))
//!         .with_conflict(ToolConflict::Prefix)
//!         .with_rate_limit(60, Duration::from_secs(60)),
//! ).await?;
//! gw.sync("fs").await?;                      // 拉取工具进统一注册表
//! let out = gw.call("fs:read_file", serde_json::json!({"path": "/tmp/a.txt"})).await?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;
use tokio::sync::{Mutex, RwLock};

use crate::connection_manager::{ConnectionManager, ServerSpec};
use crate::health::ServerHealth;
use crate::protocol::MCPError;
use crate::sandbox::ServerSandbox;
use crate::tool_adapter::{from_mcp_error, result_to_string_or_error, MCPToolAdapter};
use crate::tool_discovery::ToolDiscovery;
use crate::tool_namespace::{NamespacedTool, ToolConflict, ToolNamespace};
use crate::tool_timeout::{call_tool_with_timeout, ToolSpec};
use crate::types::{MCPConfig, MCPToolDefinition};
use lc_core::tools::ToolError;
use lc_core::BaseTool;

/// 固定窗口速率限制器(P2-8):窗口内最多放行 `max_calls` 次,窗口过期重置。
///
/// 每 Server 一个实例,`call` 入口按 Server 名命中并 `allow()`。
#[derive(Debug, Clone)]
pub struct RateLimiter {
    max_calls: usize,
    window: Duration,
    window_start: Instant,
    count: usize,
}

impl RateLimiter {
    /// 创建窗口限流器:`window` 内最多 `max_calls` 次调用(至少 1)。
    pub fn new(max_calls: usize, window: Duration) -> Self {
        Self {
            max_calls: max_calls.max(1),
            window,
            window_start: Instant::now(),
            count: 0,
        }
    }

    /// 尝试放行一次;窗口已过则重置计数再判。
    pub fn allow(&mut self) -> bool {
        if self.window_start.elapsed() >= self.window {
            self.window_start = Instant::now();
            self.count = 0;
        }
        if self.count < self.max_calls {
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// 当前窗口内剩余可放行次数。
    pub fn remaining(&self) -> usize {
        if self.window_start.elapsed() >= self.window {
            self.max_calls
        } else {
            self.max_calls.saturating_sub(self.count)
        }
    }
}

/// Gateway 统一审计记录(P2-8):入口层一次放行/拦截。
#[derive(Debug, Clone)]
pub struct GatewayAuditRecord {
    /// 目标 Server。
    pub server: String,
    /// 调用的工具全名(`server:tool`)。
    pub tool: String,
    /// 是否放行(拦截 = 速率限制 / 沙箱 / 熔断 / 未同步)。
    pub allowed: bool,
    /// 拦截原因(`allowed` 为 false 时有值)。
    pub reason: Option<String>,
    /// 记录时间。
    pub at: SystemTime,
}

/// 单个 Server 的 Gateway 策略(冲突 / 超时 / 沙箱 / 静态层)。
///
/// 速率限制不在策略里:限流器运行时状态存 `rate_limiters`,策略只需在 register
/// 时据此建限流器即可,无需重复存储。
#[derive(Debug, Clone)]
struct ServerPolicy {
    conflict: ToolConflict,
    timeout: Option<ToolSpec>,
    sandbox: Option<Arc<ServerSandbox>>,
    /// 该 Server 全部工具自动进静态层(P2-3)。
    pin_all: bool,
}

/// Gateway 登记一个 Server 的完整声明(P2-8)。
#[derive(Debug, Clone)]
pub struct GatewayServerSpec {
    /// Server 名称(注册表 key / 工具命名空间前缀)。
    pub name: String,
    /// 连接配置(Stdio / SSE)。
    pub config: MCPConfig,
    /// 有状态 Server:空闲不回收(默认 false)。
    pub keep_alive: bool,
    /// 空闲回收阈值。
    pub max_idle: Duration,
    /// 健康熔断阈值(默认 3)。
    pub max_failures: u32,
    /// 工具命名冲突策略(默认 [`ToolConflict::Prefix`])。
    pub conflict: ToolConflict,
    /// per-tool 默认超时(P2-4):该 Server 所有工具统一挂。
    pub default_timeout: Option<ToolSpec>,
    /// per-Server 安全沙箱(P2-6)。
    pub sandbox: Option<Arc<ServerSandbox>>,
    /// 速率限制(P2-8):`(max_calls, window)`,`None` 不限流。
    pub rate_limit: Option<(usize, Duration)>,
    /// 该 Server 全部工具进静态层常驻注入(P2-3)。
    pub pin_all: bool,
}

impl GatewayServerSpec {
    /// 创建一个 Gateway Server 声明。
    pub fn new(name: impl Into<String>, config: MCPConfig) -> Self {
        Self {
            name: name.into(),
            config,
            keep_alive: false,
            max_idle: Duration::from_secs(300),
            max_failures: 3,
            conflict: ToolConflict::Prefix,
            default_timeout: None,
            sandbox: None,
            rate_limit: None,
            pin_all: false,
        }
    }

    /// 标记有状态 Server:空闲不回收。
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// 设置空闲回收阈值。
    pub fn with_max_idle(mut self, max_idle: Duration) -> Self {
        self.max_idle = max_idle;
        self
    }

    /// 设置健康熔断阈值。
    pub fn with_max_failures(mut self, max_failures: u32) -> Self {
        self.max_failures = max_failures.max(1);
        self
    }

    /// 设置工具命名冲突策略。
    pub fn with_conflict(mut self, conflict: ToolConflict) -> Self {
        self.conflict = conflict;
        self
    }

    /// 挂 per-tool 默认超时(P2-4),该 Server 所有工具生效。
    pub fn with_timeout(mut self, spec: ToolSpec) -> Self {
        self.default_timeout = Some(spec);
        self
    }

    /// 挂 per-Server 安全沙箱(P2-6)。
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// 挂固定窗口速率限制(P2-8):`window` 内最多 `max_calls` 次。
    pub fn with_rate_limit(mut self, max_calls: usize, window: Duration) -> Self {
        self.rate_limit = Some((max_calls, window));
        self
    }

    /// 该 Server 全部工具进静态层常驻注入(P2-3)。
    pub fn pin_all_tools(mut self) -> Self {
        self.pin_all = true;
        self
    }

    /// 转成底层连接管理器的 ServerSpec(借用字段,config 克隆)。
    fn to_server_spec(&self) -> ServerSpec {
        let mut spec = ServerSpec::new(&self.name, self.config.clone())
            .with_max_idle(self.max_idle)
            .with_max_failures(self.max_failures);
        if self.keep_alive {
            spec = spec.keep_alive();
        }
        spec
    }
}

/// MCP Gateway(P2-8):统一入口 + 统一工具注册表 + 按需分发。
///
/// 整合 P2-1~P2-6 的能力:
/// - 连接管理([`ConnectionManager`],惰性 / 空闲回收 / 熔断);
/// - 命名空间([`ToolNamespace`])与静态/动态发现([`ToolDiscovery`]);
/// - per-tool 超时([`ToolSpec`])、安全沙箱([`ServerSandbox`]);
/// - 每 Server 速率限制([`RateLimiter`]) + 统一审计([`GatewayAuditRecord`])。
///
/// 对外暴露统一工具注册表([`tools`](Self::tools)),并可直接 `call("server:tool")`
/// 或转成 [`BaseTool`](Self::as_base_tools) 挂 Agent。
pub struct MCPGateway {
    manager: ConnectionManager,
    /// Server 名 → 策略。
    policies: RwLock<HashMap<String, ServerPolicy>>,
    /// 统一工具注册表:full_name → (server, raw) 路由 + 定义。
    namespace: RwLock<ToolNamespace>,
    /// full_name → 工具定义(供 select / as_base_tools)。
    definitions: RwLock<HashMap<String, MCPToolDefinition>>,
    /// 静态层 + 动态层发现(按 full_name 建索引)。
    discovery: RwLock<ToolDiscovery>,
    /// 已同步过工具的 Server 名(幂等 sync)。
    synced: RwLock<HashSet<String>>,
    /// 每 Server 速率限制器。
    rate_limiters: Mutex<HashMap<String, RateLimiter>>,
    /// 统一审计环形缓冲。
    audit: Arc<StdMutex<VecDeque<GatewayAuditRecord>>>,
    max_audit: usize,
}

impl Default for MCPGateway {
    fn default() -> Self {
        Self::new()
    }
}

impl MCPGateway {
    /// 创建 Gateway(默认审计上限 1000 条)。
    pub fn new() -> Self {
        Self {
            manager: ConnectionManager::new(),
            policies: RwLock::new(HashMap::new()),
            namespace: RwLock::new(ToolNamespace::new()),
            definitions: RwLock::new(HashMap::new()),
            discovery: RwLock::new(ToolDiscovery::new()),
            synced: RwLock::new(HashSet::new()),
            rate_limiters: Mutex::new(HashMap::new()),
            audit: Arc::new(StdMutex::new(VecDeque::new())),
            max_audit: 1000,
        }
    }

    /// 自定义审计日志上限(至少 1)。
    pub fn with_max_audit(mut self, max_audit: usize) -> Self {
        self.max_audit = max_audit.max(1);
        self
    }

    /// 登记一个 Server(惰性:不建连、不拉工具,只存声明 + 策略)。
    pub async fn register(&self, spec: GatewayServerSpec) -> Result<(), MCPError> {
        let rate_limit = spec.rate_limit;
        self.manager.register(spec.to_server_spec()).await?;
        if let Some((max_calls, window)) = rate_limit {
            self.rate_limiters
                .lock()
                .await
                .insert(spec.name.clone(), RateLimiter::new(max_calls, window));
        }
        let policy = ServerPolicy {
            conflict: spec.conflict,
            timeout: spec.default_timeout,
            sandbox: spec.sandbox,
            pin_all: spec.pin_all,
        };
        self.policies.write().await.insert(spec.name, policy);
        Ok(())
    }

    /// 同步某个 Server 的工具进统一注册表(惰性建连;幂等)。
    ///
    /// 连接失败 / 冲突策略拒绝时返回错误,已注册表项不回滚。
    pub async fn sync(&self, server: &str) -> Result<Vec<NamespacedTool>, MCPError> {
        if self.synced.read().await.contains(server) {
            return Ok(self.tools_for_server(server).await);
        }
        let policy = self.policies.read().await.get(server).cloned();
        let conflict = policy
            .as_ref()
            .map(|p| p.conflict)
            .unwrap_or(ToolConflict::Prefix);
        let client = self.manager.client(server).await?;
        let tools = client.list_tools().await?;
        let namespaced = {
            let mut ns = self.namespace.write().await;
            ns.register(server, tools, conflict)?
        };
        {
            let mut defs = self.definitions.write().await;
            let mut disc = self.discovery.write().await;
            let pin_all = policy.as_ref().map(|p| p.pin_all).unwrap_or(false);
            for nt in &namespaced {
                defs.insert(nt.full_name.clone(), nt.definition.clone());
                // 发现层按全名建索引,select 返回的是 LLM 看到的 `server:tool`。
                let mut renamed = nt.definition.clone();
                renamed.name = nt.full_name.clone();
                disc.register(renamed);
                if pin_all {
                    disc.pin(&nt.full_name);
                }
            }
        }
        self.synced.write().await.insert(server.to_string());
        Ok(namespaced)
    }

    /// 同步所有已登记 Server 的工具,返回成功同步的 Server 数(单个失败不中断)。
    pub async fn sync_all(&self) -> Result<usize, MCPError> {
        let names: Vec<String> = self.policies.read().await.keys().cloned().collect();
        let mut ok = 0usize;
        for name in names {
            if self.sync(&name).await.is_ok() {
                ok += 1;
            }
        }
        Ok(ok)
    }

    /// 统一工具注册表(已同步的 Server 的命名空间化工具)。
    pub async fn tools(&self) -> Vec<NamespacedTool> {
        let ns = self.namespace.read().await;
        let defs = self.definitions.read().await;
        ns.names()
            .into_iter()
            .filter_map(|full| {
                let (server, _) = ns.resolve(&full)?;
                let definition = defs.get(&full)?.clone();
                Some(NamespacedTool {
                    full_name: full,
                    server: server.to_string(),
                    definition,
                })
            })
            .collect()
    }

    /// 静态层 + 动态层工具选择(P2-3):已同步工具中按 query 取 top-k。
    ///
    /// 返回的工具 `name` 为 `server:tool` 全名,直接喂给 LLM。
    pub async fn select(
        &self,
        query: &str,
        top_k: usize,
        static_limit: usize,
    ) -> Vec<MCPToolDefinition> {
        self.discovery
            .read()
            .await
            .select(query, top_k, static_limit)
    }

    /// 把某个全名工具 pin 进静态层(高频常驻);未同步则返回 false。
    pub async fn pin(&self, full_name: &str) -> bool {
        self.discovery.write().await.pin(full_name)
    }

    /// 统一调用入口:按 `server:tool` 全名分发到对应 Server。
    ///
    /// 内部顺序:解析(未同步则按前缀自动 `sync`)→ 速率限制 → 取客户端(惰性建连
    /// + 熔断门控)→ 沙箱参数校验 → 带超时调用。放行/拦截均记入统一审计。
    pub async fn call(&self, full_name: &str, arguments: Value) -> Result<String, ToolError> {
        let (server, raw) = match self.resolve(full_name).await {
            Some(x) => x,
            None => {
                // 已登记但未同步的 Server:按 server:tool 前缀自动 sync(按需分发)。
                let Some((s, _)) = ToolNamespace::parse(full_name) else {
                    return Err(ToolError::ToolNotFound(full_name.to_string()));
                };
                if !self.policies.read().await.contains_key(s) {
                    return Err(ToolError::ToolNotFound(full_name.to_string()));
                }
                self.sync(s).await.map_err(from_mcp_error)?;
                match self.resolve(full_name).await {
                    Some(x) => x,
                    None => return Err(ToolError::ToolNotFound(full_name.to_string())),
                }
            }
        };

        // 速率限制(固定窗口)。
        {
            let mut limiters = self.rate_limiters.lock().await;
            if let Some(limiter) = limiters.get_mut(&server) {
                if !limiter.allow() {
                    self.record(&server, full_name, false, Some("速率限制".to_string()));
                    return Err(ToolError::ExecutionFailed(format!(
                        "MCP server '{server}' 触发速率限制,请求被拒绝"
                    )));
                }
            }
        }

        // 取客户端:惰性建连 + 熔断门控(P2-1 / P2-5)。
        let client = self.manager.client(&server).await.map_err(from_mcp_error)?;

        // 沙箱:参数级最小权限(P2-6)。
        let sandbox = self
            .policies
            .read()
            .await
            .get(&server)
            .and_then(|p| p.sandbox.clone());
        if let Some(sb) = sandbox {
            if let Err(e) = sb.check_call(&raw, &arguments) {
                self.record(&server, full_name, false, Some(e.to_string()));
                return Err(ToolError::InvalidInput(e.to_string()));
            }
        }

        // 带超时调用(P2-4);失败记审计并保留 code/message。
        let timeout = self
            .policies
            .read()
            .await
            .get(&server)
            .and_then(|p| p.timeout.clone());
        let result = match timeout {
            Some(spec) => call_tool_with_timeout(&client, &raw, arguments, &spec).await,
            None => client.call_tool(&raw, arguments).await,
        };
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                self.record(&server, full_name, false, Some(e.message.clone()));
                return Err(from_mcp_error(e));
            }
        };
        let out = result_to_string_or_error(&result);
        self.record(&server, full_name, out.is_ok(), None);
        out
    }

    /// 把统一注册表转成 `BaseTool` 适配器列表(挂 Agent 用)。
    ///
    /// 每个适配器自动带命名空间前缀 + per-Server 超时 + 沙箱。需要先 `sync` 过
    /// 才有工具可转。
    pub async fn as_base_tools(&self) -> Result<Vec<Arc<dyn BaseTool>>, MCPError> {
        let mut out = Vec::new();
        for nt in self.tools().await {
            let client = self.manager.client(&nt.server).await?;
            let policy = self.policies.read().await.get(&nt.server).cloned();
            let mut adapter = MCPToolAdapter::namespaced(client, &nt.server, nt.definition);
            if let Some(t) = policy.as_ref().and_then(|p| p.timeout.clone()) {
                adapter = adapter.with_timeout(t);
            }
            if let Some(sb) = policy.as_ref().and_then(|p| p.sandbox.clone()) {
                adapter = adapter.with_sandbox(sb);
            }
            out.push(Arc::new(adapter) as Arc<dyn BaseTool>);
        }
        Ok(out)
    }

    /// 健康探活(P2-5):委托给连接管理器。
    pub async fn health(&self, name: &str) -> Result<ServerHealth, MCPError> {
        self.manager.health(name).await
    }

    /// 摘除所有熔断的 Server(P2-5)。
    pub async fn reap_unhealthy(&self) -> Vec<String> {
        self.manager.reap_unhealthy().await
    }

    /// 手动触发一轮空闲回收。
    pub async fn reap_idle(&self) -> usize {
        self.manager.reap_idle().await
    }

    /// 显式关闭某个 Server 的连接(下次 `call` 惰性重建)。
    pub async fn release(&self, name: &str) -> Result<(), MCPError> {
        self.manager.release(name).await
    }

    /// 关闭全部连接并停止后台回收 task。
    pub async fn shutdown(&self) {
        self.manager.shutdown().await
    }

    /// 已登记的 Server 数。
    pub async fn server_count(&self) -> usize {
        self.policies.read().await.len()
    }

    /// 统一审计日志(按时间先后)。
    pub fn audit_log(&self) -> Vec<GatewayAuditRecord> {
        self.audit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// 清空统一审计日志。
    pub fn clear_audit(&self) {
        self.audit.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// 从全名解析路由目标;未注册返回 `None`。
    async fn resolve(&self, full_name: &str) -> Option<(String, String)> {
        self.namespace
            .read()
            .await
            .resolve(full_name)
            .map(|(s, r)| (s.to_string(), r.to_string()))
    }

    /// 某个 Server 已同步的命名空间化工具。
    async fn tools_for_server(&self, server: &str) -> Vec<NamespacedTool> {
        let ns = self.namespace.read().await;
        let defs = self.definitions.read().await;
        ns.names()
            .into_iter()
            .filter_map(|full| {
                let (s, _) = ns.resolve(&full)?;
                if s != server {
                    return None;
                }
                let definition = defs.get(&full)?.clone();
                Some(NamespacedTool {
                    full_name: full,
                    server: server.to_string(),
                    definition,
                })
            })
            .collect()
    }

    /// 记一条统一审计(环形,上限 max_audit)。
    fn record(&self, server: &str, tool: &str, allowed: bool, reason: Option<String>) {
        let rec = GatewayAuditRecord {
            server: server.to_string(),
            tool: tool.to_string(),
            allowed,
            reason,
            at: SystemTime::now(),
        };
        let mut audit = self.audit.lock().unwrap_or_else(|e| e.into_inner());
        if audit.len() >= self.max_audit {
            audit.pop_front();
        }
        audit.push_back(rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ParamRule;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use serde_json::json;

    /// 固定窗口限流:窗口内超出上限拒绝,窗口过期后恢复。
    #[test]
    fn test_rate_limiter_window_blocks_then_recovers() {
        let mut rl = RateLimiter::new(2, Duration::from_millis(30));
        assert!(rl.allow());
        assert!(rl.allow());
        assert!(!rl.allow(), "窗口内第 3 次应拒绝");
        assert_eq!(rl.remaining(), 0);
        std::thread::sleep(Duration::from_millis(50));
        assert!(rl.allow(), "窗口过期后应恢复配额");
    }

    /// 限流器至少放行 1 次。
    #[test]
    fn test_rate_limiter_min_one() {
        let mut rl = RateLimiter::new(0, Duration::from_secs(60));
        assert!(rl.allow(), "max_calls 至少为 1");
    }

    /// register 惰性:不建连、不拉工具;统一注册表为空。
    #[tokio::test]
    async fn test_register_is_lazy_and_empty_registry() {
        let gw = MCPGateway::new();
        gw.register(GatewayServerSpec::new(
            "bad",
            MCPConfig::stdio("no_such_cmd_xyz", vec![]),
        ))
        .await
        .expect("register 不应建连");
        assert_eq!(gw.server_count().await, 1);
        assert!(gw.tools().await.is_empty(), "未 sync 前统一注册表应为空");
    }

    /// 重复登记同名 Server 报错。
    #[tokio::test]
    async fn test_register_duplicate_rejected() {
        let gw = MCPGateway::new();
        let spec = GatewayServerSpec::new("dup", MCPConfig::stdio("no_such_cmd_xyz", vec![]));
        gw.register(spec.clone()).await.expect("首次登记成功");
        let err = gw.register(spec).await.unwrap_err();
        assert!(err.to_string().contains("已注册"), "{}", err);
    }

    /// sync 从假 SSE Server 拉工具,统一注册表出现 `server:tool`。
    #[tokio::test]
    async fn test_sync_populates_namespaced_registry() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .expect("登记成功");

        let namespaced = gw.sync("fs").await.expect("sync 应成功");
        assert_eq!(namespaced.len(), 1);
        assert_eq!(namespaced[0].full_name, "fs:echo");
        assert_eq!(namespaced[0].server, "fs");
        assert_eq!(namespaced[0].definition.name, "echo");

        let tools = gw.tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].full_name, "fs:echo");
    }

    /// sync 幂等:重复 sync 不重复入表。
    #[tokio::test]
    async fn test_sync_is_idempotent() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();
        let first = gw.sync("fs").await.unwrap();
        let second = gw.sync("fs").await.unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1, "幂等:第二次 sync 不重复");
        assert_eq!(gw.tools().await.len(), 1);
    }

    /// 统一入口:call("server:tool") 按原始名路由到 Server(未手动 sync 时自动同步)。
    #[tokio::test]
    async fn test_call_dispatches_with_auto_sync() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();

        let out = gw.call("fs:echo", json!({})).await.expect("按全名分发");
        assert_eq!(out, "echo", "应携带原始工具名到达 Server 并回显");
    }

    /// 未注册的工具返回 ToolNotFound。
    #[tokio::test]
    async fn test_call_unknown_tool_not_found() {
        let gw = MCPGateway::new();
        let err = gw.call("ghost:read_file", json!({})).await.unwrap_err();
        assert!(
            matches!(err, ToolError::ToolNotFound(ref n) if n == "ghost:read_file"),
            "{}",
            err
        );
    }

    /// 沙箱拦截:违规参数在 Gateway 入口被拦,不进 Server,并记审计。
    #[tokio::test]
    async fn test_call_sandbox_blocks_and_audits() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        }));
        let gw = MCPGateway::new();
        gw.register(
            GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)).with_sandbox(sandbox),
        )
        .await
        .unwrap();

        let err = gw
            .call("fs:echo", json!({ "path": "file:///etc/passwd" }))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::InvalidInput(ref m) if m.contains("最小权限")),
            "{}",
            err
        );

        let log = gw.audit_log();
        assert!(!log.is_empty(), "拦截应记审计");
        assert!(!log[0].allowed);
        assert_eq!(log[0].tool, "fs:echo");
        assert!(log[0].reason.as_deref().unwrap().contains("最小权限"));
    }

    /// 速率限制:窗口内超限拒绝并记审计,放行与拦截各一条。
    #[tokio::test]
    async fn test_call_rate_limit_blocks_and_audits() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(
            GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url))
                .with_rate_limit(1, Duration::from_secs(60)),
        )
        .await
        .unwrap();

        let first = gw.call("fs:echo", json!({})).await;
        assert!(first.is_ok(), "窗口内第 1 次应放行");
        let err = gw.call("fs:echo", json!({})).await.unwrap_err();
        assert!(err.to_string().contains("速率限制"), "{}", err);

        let log = gw.audit_log();
        assert_eq!(log.len(), 2, "放行 + 拦截各一条");
        assert!(log[0].allowed);
        assert!(!log[1].allowed);
        assert!(log[1].reason.as_deref().unwrap().contains("速率限制"));
    }

    /// 静态层 + 动态层:pin 后 select 命中全名工具。
    #[tokio::test]
    async fn test_select_over_synced_registry() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();
        gw.sync("fs").await.unwrap();
        assert!(gw.pin("fs:echo").await, "pin 已同步工具应成功");
        assert!(!gw.pin("fs:ghost").await, "未同步工具 pin 失败");

        let picked = gw.select("echo tool", 5, usize::MAX).await;
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].name, "fs:echo", "发现层返回全名");
    }

    /// 转 BaseTool:适配器带命名空间 + 超时 + 沙箱,可正常调用。
    #[tokio::test]
    async fn test_as_base_tools_builds_adapters() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new();
        gw.register(
            GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url))
                .with_timeout(ToolSpec::new("echo", Duration::from_secs(5))),
        )
        .await
        .unwrap();
        gw.sync("fs").await.unwrap();

        let tools = gw.as_base_tools().await.expect("构建适配器成功");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "fs:echo");
        let out = tools[0].run("{}".into()).await.expect("适配器可调用");
        assert_eq!(out, "echo");
    }

    /// 熔断委托:健康探活 -> Down,reap_unhealthy 摘除。
    #[tokio::test]
    async fn test_gateway_health_and_reap() {
        let gw = MCPGateway::new();
        gw.register(
            GatewayServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                .with_max_failures(1),
        )
        .await
        .unwrap();

        let h = gw.health("bad").await.expect("健康探活不报错");
        assert_eq!(h.status, crate::HealthStatus::Down, "1 次失败即熔断");
        let removed = gw.reap_unhealthy().await;
        assert_eq!(removed, vec!["bad".to_string()]);
    }

    /// 审计环形上限:只保留最新 max_audit 条。
    #[tokio::test]
    async fn test_audit_cap_keeps_newest() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = MCPGateway::new().with_max_audit(1);
        gw.register(GatewayServerSpec::new("fs", MCPConfig::sse(&fake.sse_url)))
            .await
            .unwrap();
        gw.call("fs:echo", json!({})).await.expect("第 1 次调用");
        gw.call("fs:echo", json!({})).await.expect("第 2 次调用");

        let log = gw.audit_log();
        assert_eq!(log.len(), 1, "环形只留最新 1 条");
        assert_eq!(log[0].tool, "fs:echo");
    }
}
