//! Gateway unified audit records.

use std::time::SystemTime;

/// Gateway unified audit record (P2-8): one allow/block decision at the entry layer.
#[derive(Debug, Clone)]
pub struct GatewayAuditRecord {
    /// Target server.
    pub server: String,
    /// Full name of the tool called (`server:tool`).
    pub tool: String,
    /// Whether the call was allowed (blocked = rate limit / sandbox / breaker / unsynced).
    pub allowed: bool,
    /// Reason for blocking (present when `allowed` is false).
    pub reason: Option<String>,
    /// Record timestamp.
    pub at: SystemTime,
}
