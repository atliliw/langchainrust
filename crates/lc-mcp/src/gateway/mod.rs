//! MCP Gateway (P2-8): unified entry point, on-demand internal dispatch.
//!
//! In a 100+ server scenario you want exactly one external entry point: the caller `register`s all
//! servers, then calls by the `server:tool` full name, and the gateway routes internally to the
//! right server. This module folds the P2-1~P2-6 capabilities into one unified tool registry:
//!
//! - **P2-1** lazy connection / idle reaping / connection pooling ([`ConnectionManager`]);
//! - **P2-2** tool namespacing + conflict policy ([`ToolNamespace`]);
//! - **P2-3** static-layer + dynamic-layer tool discovery ([`ToolDiscovery`]);
//! - **P2-4** per-tool timeout + Progress reset ([`ToolSpec`](crate::tool_timeout::ToolSpec));
//! - **P2-5** health checks + circuit breaking ([`crate::ServerHealth`]);
//! - **P2-6** per-server security sandbox ([`ServerSandbox`](crate::sandbox::ServerSandbox));
//! - **Rate limiting**: fixed-window limit per server ([`RateLimiter`]);
//! - **Unified audit**: the Gateway entry records every allow/block ([`GatewayAuditRecord`]).
//!
//! # Unified registry
//!
//! `register` only records the server + policy (lazy: no connection, no tool pull); `sync` /
//! `sync_all` actually connect the server, fetch `tools/list`, and namespace it into the registry.
//! A `call` that misses the registry auto-`sync`s by the `server:tool` prefix (on-demand dispatch).
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
//! gw.sync("fs").await?;                      // pull tools into the unified registry
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

/// MCP Gateway (P2-8): unified entry point + unified tool registry + on-demand dispatch.
///
/// Folds together the P2-1~P2-6 capabilities:
/// - connection management ([`ConnectionManager`], lazy / idle reaping / circuit breaking);
/// - namespacing ([`ToolNamespace`]) and static/dynamic discovery ([`ToolDiscovery`]);
/// - per-tool timeouts ([`ToolSpec`](crate::tool_timeout::ToolSpec)), security sandbox ([`ServerSandbox`](crate::sandbox::ServerSandbox));
/// - per-server rate limiting ([`RateLimiter`]) + unified audit ([`GatewayAuditRecord`]).
///
/// Exposes the unified tool registry ([`tools`](Self::tools)), callable directly via `call("server:tool")`
/// or converted into [`BaseTool`](Self::as_base_tools) adapters to attach to an Agent.
pub struct MCPGateway {
    manager: ConnectionManager,
    /// Server name → policy.
    policies: RwLock<HashMap<String, ServerPolicy>>,
    /// Unified tool registry: full_name → (server, raw) routing + definition.
    namespace: RwLock<ToolNamespace>,
    /// full_name → tool definition (for select / as_base_tools).
    definitions: RwLock<HashMap<String, MCPToolDefinition>>,
    /// Static-layer + dynamic-layer discovery (indexed by full_name).
    discovery: RwLock<ToolDiscovery>,
    /// Server names whose tools have been synced (idempotent sync).
    synced: RwLock<HashSet<String>>,
    /// Per-server rate limiter.
    rate_limiters: Mutex<HashMap<String, RateLimiter>>,
    /// Unified audit ring buffer.
    audit: Arc<StdMutex<VecDeque<GatewayAuditRecord>>>,
    max_audit: usize,
}

impl Default for MCPGateway {
    fn default() -> Self {
        Self::new()
    }
}

impl MCPGateway {
    /// Creates a Gateway (default audit cap 1000 entries).
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

    /// Sets a custom audit-log cap (minimum 1).
    pub fn with_max_audit(mut self, max_audit: usize) -> Self {
        self.max_audit = max_audit.max(1);
        self
    }

    /// Registers a server (lazy: no connection, no tool pull — only stores the declaration + policy).
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

    /// Syncs a server's tools into the unified registry (lazy connection; idempotent).
    ///
    /// Returns an error on connection failure / conflict-policy rejection; already-registered entries are not rolled back.
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
                // The discovery layer indexes by full name; select returns the `server:tool` the LLM sees.
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

    /// Syncs all registered servers' tools, returning how many synced successfully (a single failure doesn't abort).
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

    /// Unified tool registry (namespaced tools of synced servers).
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

    /// Static-layer + dynamic-layer tool selection (P2-3): takes top-k from synced tools by query.
    ///
    /// The returned tools' `name` is the `server:tool` full name, fed straight to the LLM.
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

    /// Pins a full-name tool into the static layer (high-frequency resident); returns false if not synced.
    pub async fn pin(&self, full_name: &str) -> bool {
        self.discovery.write().await.pin(full_name)
    }

    /// Unified call entry: dispatches to the right server by the `server:tool` full name.
    ///
    /// Internal order: resolve (auto-`sync` by prefix if unsynced) → rate limit → get the client
    /// (lazy connection + breaker gate) → sandbox parameter check → timed call. Every allow/block is recorded in the unified audit.
    pub async fn call(&self, full_name: &str, arguments: Value) -> Result<String, ToolError> {
        let (server, raw) = match self.resolve(full_name).await {
            Some(x) => x,
            None => {
                // A registered-but-unsynced server: auto-sync by the server:tool prefix (on-demand dispatch).
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

        // Rate limit (fixed window).
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

        // Get the client: lazy connection + breaker gate (P2-1 / P2-5).
        let client = self.manager.client(&server).await.map_err(from_mcp_error)?;

        // Sandbox: parameter-level least privilege (P2-6).
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

        // Timed call (P2-4); on failure record an audit entry and keep code/message.
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

    /// Converts the unified registry into a `BaseTool` adapter list (for attaching to an Agent).
    ///
    /// Each adapter automatically carries the namespace prefix + per-server timeout + sandbox.
    /// Tools must be `sync`ed first before they can be converted.
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

    /// Health probe (P2-5): delegates to the connection manager.
    pub async fn health(&self, name: &str) -> Result<ServerHealth, MCPError> {
        self.manager.health(name).await
    }

    /// Reaps all tripped (broken) servers (P2-5).
    pub async fn reap_unhealthy(&self) -> Vec<String> {
        self.manager.reap_unhealthy().await
    }

    /// Manually triggers a round of idle reaping.
    pub async fn reap_idle(&self) -> usize {
        self.manager.reap_idle().await
    }

    /// Explicitly closes a server's connection (lazily rebuilt on the next `call`).
    pub async fn release(&self, name: &str) -> Result<(), MCPError> {
        self.manager.release(name).await
    }

    /// Closes all connections and stops the background reaping task.
    pub async fn shutdown(&self) {
        self.manager.shutdown().await
    }

    /// Number of registered servers.
    pub async fn server_count(&self) -> usize {
        self.policies.read().await.len()
    }

    /// Unified audit log (in chronological order).
    pub fn audit_log(&self) -> Vec<GatewayAuditRecord> {
        self.audit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Clears the unified audit log.
    pub fn clear_audit(&self) {
        self.audit.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Resolves the routing target from a full name; returns `None` if not registered.
    async fn resolve(&self, full_name: &str) -> Option<(String, String)> {
        self.namespace
            .read()
            .await
            .resolve(full_name)
            .map(|(s, r)| (s.to_string(), r.to_string()))
    }

    /// A server's synced, namespaced tools.
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

    /// Records one unified audit entry (ring buffer, cap max_audit).
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
