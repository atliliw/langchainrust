//! Multi-tenant isolation (P2-10): each tenant gets its own independent tool registry, mutually invisible.
//!
//! 100+ Server deployments are often shared by multiple business parties; tenants must be isolated:
//! the tools, namespaces, discovery layer, rate limits and audits registered by tenant A are invisible to tenant B.
//! [`TenantGateway`] holds an independent [`MCPGateway`] per tenant;
//! register / sync / call / audit all route per tenant; removing a tenant clears its whole registry.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use super::gateway::{GatewayAuditRecord, GatewayServerSpec, MCPGateway};
use super::protocol::MCPError;
use lc_core::tools::ToolError;

/// Multi-tenant Gateway container (P2-10): `tenant_id` → independent Gateway.
///
/// Each tenant's registry is fully isolated (namespaces / discovery / rate limits / audit are mutually invisible).
/// Servers register per tenant — a same-named server in different tenants is independent; sync and call must
/// name the tenant explicitly, preventing cross-tenant leakage. Tenants are lazy: a missing tenant creates an
/// empty registry and connects to no server.
#[derive(Default)]
pub struct TenantGateway {
    tenants: RwLock<HashMap<String, Arc<MCPGateway>>>,
}

impl TenantGateway {
    /// An empty tenant container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets a tenant's Gateway; creates an empty registry when missing (lazy).
    pub async fn tenant(&self, tenant_id: &str) -> Arc<MCPGateway> {
        let mut map = self.tenants.write().await;
        map.entry(tenant_id.to_string())
            .or_insert_with(|| Arc::new(MCPGateway::new()))
            .clone()
    }

    /// Registered tenant ids (order not stable).
    pub async fn tenant_ids(&self) -> Vec<String> {
        self.tenants.read().await.keys().cloned().collect()
    }

    /// Registers a Server for the given tenant (lazy: no connection, first sync/call starts it).
    pub async fn register(&self, tenant_id: &str, spec: GatewayServerSpec) -> Result<(), MCPError> {
        self.tenant(tenant_id).await.register(spec).await
    }

    /// Syncs all Servers under the given tenant.
    pub async fn sync_all(&self, tenant_id: &str) -> Result<usize, MCPError> {
        self.tenant(tenant_id).await.sync_all().await
    }

    /// All externally exposed tool full names (`server:tool`) in that tenant's registry.
    pub async fn tools(&self, tenant_id: &str) -> Vec<String> {
        self.tenant(tenant_id)
            .await
            .tools()
            .await
            .into_iter()
            .map(|t| t.full_name)
            .collect()
    }

    /// Calls a tool by tenant (`server:tool`); auto-syncs when the Server is registered but not yet synced.
    pub async fn call(
        &self,
        tenant_id: &str,
        full_name: &str,
        args: Value,
    ) -> Result<String, ToolError> {
        self.tenant(tenant_id).await.call(full_name, args).await
    }

    /// That tenant's audit log (isolated from other tenants).
    pub async fn audit_log(&self, tenant_id: &str) -> Vec<GatewayAuditRecord> {
        self.tenant(tenant_id).await.audit_log()
    }

    /// Removes a tenant, clearing its whole registry / connections / audit. Returns whether it existed.
    pub async fn remove_tenant(&self, tenant_id: &str) -> bool {
        self.tenants.write().await.remove(tenant_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::MCPConfig;
    use serde_json::json;

    fn fs_spec(sse_url: &str) -> GatewayServerSpec {
        GatewayServerSpec::new("fs", MCPConfig::sse(sse_url))
    }

    /// Registry isolation: sync only A, tools do not leak into B; B calls its own registered Server on demand,
    /// filling its registry via its own sync, independent of A.
    #[tokio::test]
    async fn test_tenants_registry_isolated() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = TenantGateway::new();
        gw.register("tenant_a", fs_spec(&fake.sse_url))
            .await
            .unwrap();
        gw.register("tenant_b", fs_spec(&fake.sse_url))
            .await
            .unwrap();

        // Sync only A: the namespace fills only A, does not leak into B.
        gw.sync_all("tenant_a").await.unwrap();
        assert_eq!(gw.tools("tenant_a").await, vec!["fs:echo"]);
        assert!(
            gw.tools("tenant_b").await.is_empty(),
            "B 租户注册表不应看到 A 的工具"
        );

        // B calls its own registered Server on demand: auto-sync, unrelated to A's registry.
        let out = gw.call("tenant_b", "fs:echo", json!({})).await;
        assert!(out.is_ok(), "B 租户的调用应成功");
        assert_eq!(out.unwrap(), "echo");
        assert_eq!(
            gw.tools("tenant_b").await,
            vec!["fs:echo"],
            "B 的注册表由自己的同步填充"
        );
    }

    /// Audit isolation: tenant A's calls go only into A's audit, not polluting B.
    #[tokio::test]
    async fn test_tenant_audit_isolated() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let gw = TenantGateway::new();
        gw.register("a", fs_spec(&fake.sse_url)).await.unwrap();

        gw.call("a", "fs:echo", json!({})).await.unwrap();
        assert!(!gw.audit_log("a").await.is_empty(), "A 的调用应进 A 的审计");
        assert!(
            gw.audit_log("b").await.is_empty(),
            "B 租户审计不应看到 A 的记录"
        );
    }

    /// Removing a tenant: clears everything; re-accessing rebuilds an empty registry.
    #[tokio::test]
    async fn test_remove_tenant_cleans_up() {
        let gw = TenantGateway::new();
        // register is lazy, no real server needed.
        gw.register("a", fs_spec("http://localhost:1/sse"))
            .await
            .unwrap();
        assert!(gw.tenant_ids().await.contains(&"a".to_string()));

        assert!(gw.remove_tenant("a").await, "已存在的租户应被移除");
        assert!(
            !gw.tenant_ids().await.contains(&"a".to_string()),
            "移除后租户 id 不应再出现"
        );
        assert!(!gw.remove_tenant("a").await, "重复移除应返回 false");

        // Re-accessing after removal: rebuilds an empty registry (no error).
        assert!(gw.tools("a").await.is_empty());
    }
}
