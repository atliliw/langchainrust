//! 多租户隔离(P2-10):每个租户一把独立工具注册表,互不可见。
//!
//! 100+ Server 部署常由多个业务方共享同一批 Server,租户之间必须隔离:
//! A 租户注册的工具、命名空间、发现层、限流与审计,对 B 租户不可见。
//! [`TenantGateway`] 为每个租户持有独立的 [`MCPGateway`],
//! 注册 / 同步 / 调用 / 审计全部按租户路由;移除租户即整体清理其注册表。

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use super::gateway::{GatewayAuditRecord, GatewayServerSpec, MCPGateway};
use super::protocol::MCPError;
use lc_core::tools::ToolError;

/// 多租户 Gateway 容器(P2-10):`tenant_id` → 独立 Gateway。
///
/// 每个租户的注册表完全隔离(命名空间 / 发现 / 限流 / 审计互不可见)。
/// Server 按租户注册——同名 Server 在不同租户各自独立;同步与调用必须
/// 显式指定租户,杜绝跨租户泄漏。取租户是惰性的:不存在的租户创建一个
/// 空注册表,不连接任何 Server。
#[derive(Default)]
pub struct TenantGateway {
    tenants: RwLock<HashMap<String, Arc<MCPGateway>>>,
}

impl TenantGateway {
    /// 空租户容器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 取某个租户的 Gateway;不存在则创建空注册表(惰性)。
    pub async fn tenant(&self, tenant_id: &str) -> Arc<MCPGateway> {
        let mut map = self.tenants.write().await;
        map.entry(tenant_id.to_string())
            .or_insert_with(|| Arc::new(MCPGateway::new()))
            .clone()
    }

    /// 已注册的租户 id(顺序不稳定)。
    pub async fn tenant_ids(&self) -> Vec<String> {
        self.tenants.read().await.keys().cloned().collect()
    }

    /// 向指定租户注册一个 Server(惰性:不连接,首次 sync/call 才启动)。
    pub async fn register(&self, tenant_id: &str, spec: GatewayServerSpec) -> Result<(), MCPError> {
        self.tenant(tenant_id).await.register(spec).await
    }

    /// 同步指定租户下的全部 Server。
    pub async fn sync_all(&self, tenant_id: &str) -> Result<usize, MCPError> {
        self.tenant(tenant_id).await.sync_all().await
    }

    /// 该租户注册表里全部对外工具全名(`server:tool`)。
    pub async fn tools(&self, tenant_id: &str) -> Vec<String> {
        self.tenant(tenant_id)
            .await
            .tools()
            .await
            .into_iter()
            .map(|t| t.full_name)
            .collect()
    }

    /// 按租户调用工具(`server:tool`);Server 已注册但未同步时自动同步。
    pub async fn call(
        &self,
        tenant_id: &str,
        full_name: &str,
        args: Value,
    ) -> Result<String, ToolError> {
        self.tenant(tenant_id).await.call(full_name, args).await
    }

    /// 该租户的审计日志(与其他租户隔离)。
    pub async fn audit_log(&self, tenant_id: &str) -> Vec<GatewayAuditRecord> {
        self.tenant(tenant_id).await.audit_log()
    }

    /// 移除一个租户,整体清理其注册表 / 连接 / 审计。返回是否曾存在。
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

    /// 注册表隔离:只同步 A,工具不泄漏到 B;B 按需调用自己注册的 Server,
    /// 由自己的 sync 填充注册表,与 A 无关。
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

        // 只同步 A:命名空间只填充 A,不泄漏到 B。
        gw.sync_all("tenant_a").await.unwrap();
        assert_eq!(gw.tools("tenant_a").await, vec!["fs:echo"]);
        assert!(
            gw.tools("tenant_b").await.is_empty(),
            "B 租户注册表不应看到 A 的工具"
        );

        // B 按需调用自己注册的 Server:自动同步,与 A 的注册表无关。
        let out = gw.call("tenant_b", "fs:echo", json!({})).await;
        assert!(out.is_ok(), "B 租户的调用应成功");
        assert_eq!(out.unwrap(), "echo");
        assert_eq!(
            gw.tools("tenant_b").await,
            vec!["fs:echo"],
            "B 的注册表由自己的同步填充"
        );
    }

    /// 审计隔离:租户 A 的调用只进 A 的审计,不污染 B。
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

    /// 移除租户:整体清理;再次访问重建空注册表。
    #[tokio::test]
    async fn test_remove_tenant_cleans_up() {
        let gw = TenantGateway::new();
        // register 是惰性的,不需要真实服务器。
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

        // 移除后再次访问:重建一个空注册表(不报错)。
        assert!(gw.tools("a").await.is_empty());
    }
}
