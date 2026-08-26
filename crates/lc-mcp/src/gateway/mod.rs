//! MCP Gateway(P2-8):统一入口,内部按需分发。
//!
//! 100+ Server 场景需要对外只暴露一个入口:调用方 `register` 声明所有 Server,
//! 之后按 `server:tool` 全名调用,内部自动路由到对应 Server。本模块把
//! P2-1~P2-6 的能力整合成一个统一工具注册表:
//!
//! - **P2-1** 惰性连接 / 空闲回收 / 连接池([`ConnectionManager`]);
//! - **P2-2** 工具命名空间 + 冲突策略([`ToolNamespace`]);
//! - **P2-3** 静态层 + 动态层工具发现([`ToolDiscovery`]);
//! - **P2-4** per-tool 超时 + Progress 重置([`ToolSpec`](crate::tool_timeout::ToolSpec));
//! - **P2-5** 健康检查 + 熔断([`crate::ServerHealth`]);
//! - **P2-6** per-Server 安全沙箱([`ServerSandbox`](crate::sandbox::ServerSandbox));
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

mod audit;
mod policy;
mod rate_limiter;

pub use audit::GatewayAuditRecord;
pub use policy::GatewayServerSpec;
pub use rate_limiter::RateLimiter;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::SystemTime;

use serde_json::Value;
use tokio::sync::{Mutex, RwLock};

use crate::connection_manager::ConnectionManager;
use crate::health::ServerHealth;
use crate::protocol::MCPError;
use crate::tool_adapter::{from_mcp_error, result_to_string_or_error, MCPToolAdapter};
use crate::tool_discovery::ToolDiscovery;
use crate::tool_namespace::{NamespacedTool, ToolConflict, ToolNamespace};
use crate::tool_timeout::call_tool_with_timeout;
use crate::types::MCPToolDefinition;
use lc_core::tools::ToolError;
use lc_core::BaseTool;
use policy::ServerPolicy;

/// MCP Gateway(P2-8):统一入口 + 统一工具注册表 + 按需分发。
///
/// 整合 P2-1~P2-6 的能力:
/// - 连接管理([`ConnectionManager`],惰性 / 空闲回收 / 熔断);
/// - 命名空间([`ToolNamespace`])与静态/动态发现([`ToolDiscovery`]);
/// - per-tool 超时([`ToolSpec`](crate::tool_timeout::ToolSpec))、安全沙箱([`ServerSandbox`](crate::sandbox::ServerSandbox));
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
                    self.record(
                        &server,
                        full_name,
                        false,
                        Some("rate limit exceeded".to_string()),
                    );
                    return Err(ToolError::ExecutionFailed(format!(
                        "MCP server '{server}' exceeded rate limit, request rejected"
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
mod tests;
