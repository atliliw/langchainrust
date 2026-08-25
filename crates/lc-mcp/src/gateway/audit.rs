//! Gateway 统一审计记录。

use std::time::SystemTime;

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
