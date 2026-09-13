//! Tool-execution authorization gate (0.22.4 A16, security).
//!
//! Before A16 both the standalone [`crate::MCPToolAdapter`] and the
//! [`crate::MCPGateway`] treated "no sandbox configured" as "allow anything":
//! a server registered without a [`crate::ServerSandbox`] accepted every tool
//! call with arbitrary arguments, and there was no hook for human-in-the-loop
//! confirmation. Both surfaces now default to **fail-closed**: a freshly
//! registered server / freshly wrapped tool cannot actually execute until the
//! embedding application declares, explicitly, how unattended execution is
//! authorized.
//!
//! The declaration is a per-server [`ToolExecutionPolicy`]:
//!
//! - [`ToolExecutionPolicy::RequireApproval`] (default): each call is passed
//!   through a runtime [`ToolCallApprover`] (a human confirmation dialog, a
//!   policy engine, an allow-list). With no approver configured every call is
//!   refused before any network I/O.
//! - [`ToolExecutionPolicy::Sandboxed`]: a shared [`crate::ServerSandbox`]
//!   applies parameter-level least-privilege checks; compliant calls run
//!   unattended.
//! - [`ToolExecutionPolicy::AllowUnattended`]: the operator explicitly opts
//!   into unrestricted unattended execution for a fully trusted server (local
//!   tools, tests, sandboxed sidecars). The name is deliberately long and
//!   appears at every registration site.
//!
//! Authorization always runs before the request is sent, and a denial becomes
//! [`lc_core::tools::ToolError::PermissionDenied`] — the tool never executed,
//! which distinguishes it from an execution failure.

use std::sync::Arc;

use serde_json::Value;

use crate::sandbox::ServerSandbox;
use lc_core::tools::ToolError;

/// How unattended execution of one server's tools is authorized (A16).
///
/// The default is fail-closed ([`ToolExecutionPolicy::RequireApproval`]).
#[derive(Debug, Clone, Default)]
pub enum ToolExecutionPolicy {
    /// Every call must be approved at runtime by a configured
    /// [`ToolCallApprover`]; without one every call is refused before dispatch.
    #[default]
    RequireApproval,
    /// Calls run unattended only when the attached
    /// [`ServerSandbox`] accepts the arguments.
    Sandboxed(Arc<ServerSandbox>),
    /// The operator explicitly accepts unrestricted unattended execution
    /// (fully trusted server / tests).
    AllowUnattended,
}

/// Runtime, per-call approval gate ("confirm before execution", A16).
///
/// Implement against a human-in-the-loop UI, a policy engine or an allow-list.
/// `Ok(())` approves the call; `Err(reason)` denies it (the reason is surfaced
/// in the [`ToolError::PermissionDenied`] message). The approver receives the
/// server namespace (empty for a non-namespaced adapter), the server-side raw
/// tool name, and the parsed arguments.
#[async_trait::async_trait]
pub trait ToolCallApprover: Send + Sync {
    /// Approves or denies one tool call before it is dispatched.
    async fn approve(&self, server: &str, tool: &str, arguments: &Value) -> Result<(), String>;
}

/// Applies the execution gate shared by the adapter and the Gateway.
///
/// `label` is the externally visible call name (used only in denial messages),
/// `tool` is the server-side raw tool name (used by the sandbox and the
/// approver). Runs no network I/O itself; callers must invoke it **before**
/// dispatching the request so a denial is unobservable to the MCP server.
pub(crate) async fn authorize_call(
    policy: &ToolExecutionPolicy,
    approver: Option<&Arc<dyn ToolCallApprover>>,
    label: &str,
    server: &str,
    tool: &str,
    arguments: &Value,
) -> Result<(), ToolError> {
    match policy {
        ToolExecutionPolicy::AllowUnattended => Ok(()),
        ToolExecutionPolicy::Sandboxed(sandbox) => sandbox
            .check_call(tool, arguments)
            .map_err(|e| ToolError::InvalidInput(e.to_string())),
        ToolExecutionPolicy::RequireApproval => match approver {
            Some(approver) => approver
                .approve(server, tool, arguments)
                .await
                .map_err(|reason| {
                    ToolError::PermissionDenied(format!(
                        "call to '{label}' rejected by the approval gate: {reason}"
                    ))
                }),
            None => Err(ToolError::PermissionDenied(format!(
                "call to '{label}' refused before execution: no unattended-execution policy \
                 declared for the server (attach a sandbox, explicitly opt in with \
                 allow_unattended_execution(), or configure an approval gate)"
            ))),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ParamRule;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Test approver whose decision is an externally flippable flag.
    struct FlagApprover {
        allow: AtomicBool,
    }

    #[async_trait::async_trait]
    impl ToolCallApprover for FlagApprover {
        async fn approve(
            &self,
            _server: &str,
            _tool: &str,
            _arguments: &Value,
        ) -> Result<(), String> {
            if self.allow.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err("human said no".to_string())
            }
        }
    }

    /// The default policy with no approver is fail-closed.
    #[tokio::test]
    async fn require_approval_without_approver_denies() {
        let err = authorize_call(
            &ToolExecutionPolicy::default(),
            None,
            "fs:rm",
            "fs",
            "rm",
            &json!({}),
        )
        .await
        .unwrap_err();
        match err {
            ToolError::PermissionDenied(msg) => {
                assert!(msg.contains("fs:rm"), "{msg}");
                assert!(msg.contains("no unattended-execution policy"), "{msg}");
            }
            other => panic!("expected PermissionDenied, got {other:?}"),
        }
    }

    /// An approver denying produces a gate-specific PermissionDenied carrying the reason.
    #[tokio::test]
    async fn approver_denial_is_permission_denied_with_reason() {
        let approver: Arc<dyn ToolCallApprover> = Arc::new(FlagApprover {
            allow: AtomicBool::new(false),
        });
        let err = authorize_call(
            &ToolExecutionPolicy::RequireApproval,
            Some(&approver),
            "fs:rm",
            "fs",
            "rm",
            &json!({}),
        )
        .await
        .unwrap_err();
        match err {
            ToolError::PermissionDenied(msg) => assert!(msg.contains("human said no"), "{msg}"),
            other => panic!("expected PermissionDenied, got {other:?}"),
        }
    }

    /// An approver approving authorizes the call.
    #[tokio::test]
    async fn approver_approval_authorizes() {
        let approver: Arc<dyn ToolCallApprover> = Arc::new(FlagApprover {
            allow: AtomicBool::new(true),
        });
        authorize_call(
            &ToolExecutionPolicy::RequireApproval,
            Some(&approver),
            "fs:rm",
            "fs",
            "rm",
            &json!({}),
        )
        .await
        .expect("approver allowed");
    }

    /// Sandboxed policy: violating arguments are InvalidInput even with no approver.
    #[tokio::test]
    async fn sandboxed_policy_checks_arguments() {
        let sandbox = Arc::new(ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        }));
        let policy = ToolExecutionPolicy::Sandboxed(sandbox);
        authorize_call(
            &policy,
            None,
            "fs:read",
            "fs",
            "read",
            &json!({"path": "file:///tmp/a"}),
        )
        .await
        .expect("compliant arguments");
        let err = authorize_call(
            &policy,
            None,
            "fs:read",
            "fs",
            "read",
            &json!({"path": "file:///etc/passwd"}),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)), "{err:?}");
    }

    /// Explicit unattended opt-in authorizes everything regardless of approvers.
    #[tokio::test]
    async fn allow_unattended_bypasses_gate() {
        authorize_call(
            &ToolExecutionPolicy::AllowUnattended,
            None,
            "fs:rm",
            "fs",
            "rm",
            &json!({"path": "/"}),
        )
        .await
        .expect("operator opted into unattended execution");
    }
}
