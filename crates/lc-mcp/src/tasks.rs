//! MCP Tasks extension trait placeholder (0.22.0 S2.8; full implementation
//! in 0.22.1/0.23.0).
//!
//! The 2026-07-28 spec's Tasks extension defines async long-running requests
//! with durable handles (`tasks/get`, `tasks/result`, mid-flight input). This
//! module only fixes the client-side interface so the StatelessMcpClient can
//! adopt it later without another breaking change.

use serde_json::Value;

use crate::protocol::MCPError;

/// A durable handle to a server-side task created by a stateless request.
///
/// Full wire semantics (`tasks/get` polling, `tasks/result` retrieval,
/// cancellation) land with the 0.23.0 implementation; this placeholder pins
/// the shape.
#[async_trait::async_trait]
pub trait McpTaskHandle: Send + Sync {
    /// The opaque task id issued by the server.
    fn task_id(&self) -> &str;

    /// Polls the task status. Returns the raw status payload; interpretation
    /// (queued/running/completed/failed) is the 0.23.0 concern.
    async fn status(&self) -> Result<Value, MCPError>;

    /// Fetches the final result once the task completes.
    async fn result(&self) -> Result<Value, MCPError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Placeholder contract: a handle exposes its id; status/result are
    /// 0.23.0 concerns (stub returns an explicit not-implemented error).
    struct Stub {
        id: String,
    }

    #[async_trait::async_trait]
    impl McpTaskHandle for Stub {
        fn task_id(&self) -> &str {
            &self.id
        }
        async fn status(&self) -> Result<Value, MCPError> {
            Err(MCPError::new(
                -1,
                "MCP Tasks: implementation pending (0.22.1/0.23.0)",
            ))
        }
        async fn result(&self) -> Result<Value, MCPError> {
            Err(MCPError::new(
                -1,
                "MCP Tasks: implementation pending (0.22.1/0.23.0)",
            ))
        }
    }

    #[tokio::test]
    async fn placeholder_contract() {
        let h = Stub { id: "t-1".into() };
        assert_eq!(h.task_id(), "t-1");
        assert!(h.status().await.is_err());
        assert!(h.result().await.is_err());
    }
}
