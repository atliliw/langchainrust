//! Server policies and Gateway registration declarations.

use std::sync::Arc;
use std::time::Duration;

use crate::connection_manager::ServerSpec;
use crate::sandbox::ServerSandbox;
use crate::tool_namespace::ToolConflict;
use crate::tool_timeout::ToolSpec;
use crate::types::MCPConfig;

/// A single server's Gateway policy (conflict / timeout / sandbox / static layer).
///
/// Rate limiting is not part of the policy: the limiter's runtime state lives in `rate_limiters`,
/// so the policy only needs to build the limiter from it at register time; no duplicate storage.
#[derive(Debug, Clone)]
pub(crate) struct ServerPolicy {
    pub(crate) conflict: ToolConflict,
    pub(crate) timeout: Option<ToolSpec>,
    pub(crate) sandbox: Option<Arc<ServerSandbox>>,
    /// All tools of this server go into the static layer automatically (P2-3).
    pub(crate) pin_all: bool,
}

/// A complete declaration for registering one server with the Gateway (P2-8).
#[derive(Debug, Clone)]
pub struct GatewayServerSpec {
    /// Server name (registry key / tool-namespace prefix).
    pub name: String,
    /// Connection config (Stdio / SSE).
    pub config: MCPConfig,
    /// Stateful server: not reaped when idle (default false).
    pub keep_alive: bool,
    /// Idle-reap threshold.
    pub max_idle: Duration,
    /// Health-breaker threshold (default 3).
    pub max_failures: u32,
    /// Tool-name conflict policy (default [`ToolConflict::Prefix`]).
    pub conflict: ToolConflict,
    /// Default per-tool timeout (P2-4): applied uniformly to all tools of this server.
    pub default_timeout: Option<ToolSpec>,
    /// Per-server security sandbox (P2-6).
    pub sandbox: Option<Arc<ServerSandbox>>,
    /// Rate limit (P2-8): `(max_calls, window)`; `None` means no limit.
    pub rate_limit: Option<(usize, Duration)>,
    /// All tools of this server go into the static layer for persistent injection (P2-3).
    pub pin_all: bool,
}

impl GatewayServerSpec {
    /// Creates a Gateway server declaration.
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

    /// Marks the server stateful: not reaped when idle.
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// Sets the idle-reap threshold.
    pub fn with_max_idle(mut self, max_idle: Duration) -> Self {
        self.max_idle = max_idle;
        self
    }

    /// Sets the health-breaker threshold.
    pub fn with_max_failures(mut self, max_failures: u32) -> Self {
        self.max_failures = max_failures.max(1);
        self
    }

    /// Sets the tool-name conflict policy.
    pub fn with_conflict(mut self, conflict: ToolConflict) -> Self {
        self.conflict = conflict;
        self
    }

    /// Attaches a per-tool default timeout (P2-4), effective for all tools of this server.
    pub fn with_timeout(mut self, spec: ToolSpec) -> Self {
        self.default_timeout = Some(spec);
        self
    }

    /// Attaches a per-server security sandbox (P2-6).
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Attaches a fixed-window rate limit (P2-8): at most `max_calls` per `window`.
    pub fn with_rate_limit(mut self, max_calls: usize, window: Duration) -> Self {
        self.rate_limit = Some((max_calls, window));
        self
    }

    /// All tools of this server go into the static layer for persistent injection (P2-3).
    pub fn pin_all_tools(mut self) -> Self {
        self.pin_all = true;
        self
    }

    /// Converts into the underlying connection manager's ServerSpec (borrows fields, clones config).
    pub(crate) fn to_server_spec(&self) -> ServerSpec {
        let mut spec = ServerSpec::new(&self.name, self.config.clone())
            .with_max_idle(self.max_idle)
            .with_max_failures(self.max_failures);
        if self.keep_alive {
            spec = spec.keep_alive();
        }
        spec
    }
}
