//! MCP multi-Server connection management (P2-1): lazy startup + idle reaping + connection pooling.
//!
//! 100+ Servers each calling `MCPClient::connect` directly = hundreds of subprocesses / long
//! connections, exhausting both memory and file descriptors. This module provides a managed registry:
//!
//! - **Lazy startup**: `register` only records the `ServerSpec`; only the first `client(name)` actually
//!   spawns a subprocess / opens an SSE connection, so unused Servers cost nothing.
//! - **Idle reaping**: a background task scans periodically; non-`keep_alive` Servers idle longer than
//!   `max_idle` get `close()`d to free the connection; stateful Servers marked `keep_alive` are exempt.
//! - **Connection pooling**: `client()` on the same `ManagedServer` is idempotent; later calls reuse the
//!   connection, no repeated spawning.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_mcp::{ConnectionManager, ServerSpec, MCPConfig};
//!
//! let manager = ConnectionManager::new();
//! manager.register(ServerSpec::new("fs", MCPConfig::stdio("npx", vec!["@anthropic/mcp-server-filesystem".into(), "/tmp".into()]))).await?;
//! manager.register(ServerSpec::new("db", MCPConfig::sse("http://localhost:8080/sse")).keep_alive()).await?;
//!
//! // The connection is started lazily, only on first call
//! let client = manager.client("fs").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::sync::RwLock;
use tokio::time::sleep;

use crate::client::MCPClient;
use crate::health::{probe_health, BreakerState, CircuitBreaker, HealthStatus, ServerHealth};
use crate::protocol::MCPError;
use crate::types::MCPConfig;

/// Default idle-reap scan interval.
const DEFAULT_REAP_INTERVAL: Duration = Duration::from_secs(60);

/// Declaration of a single managed Server.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    /// Server name (registry key / tool namespace prefix).
    pub name: String,
    /// Connection config (Stdio / SSE).
    pub config: MCPConfig,
    /// Stateful-Server flag: not reaped while idle (default false).
    pub keep_alive: bool,
    /// Idle-reap threshold: connections unused beyond this duration are closed to free resources.
    pub max_idle: Duration,
    /// Health-breaker threshold (P2-5): tripped and removed after N consecutive failures (default 3).
    pub max_failures: u32,
}

impl ServerSpec {
    /// Creates a managed Server declaration.
    pub fn new(name: impl Into<String>, config: MCPConfig) -> Self {
        Self {
            name: name.into(),
            config,
            keep_alive: false,
            max_idle: Duration::from_secs(300),
            max_failures: 3,
        }
    }

    /// Marks a stateful Server: not reaped while idle.
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// Sets the idle-reap threshold.
    pub fn with_max_idle(mut self, max_idle: Duration) -> Self {
        self.max_idle = max_idle;
        self
    }

    /// Sets the health-breaker threshold (P2-5): trips after N consecutive failures, refusing requests until the backoff elapses.
    pub fn with_max_failures(mut self, max_failures: u32) -> Self {
        self.max_failures = max_failures.max(1);
        self
    }
}

/// Runtime state of a managed Server: lazy connection + last-used time + health breaker.
struct ManagedServer {
    spec: ServerSpec,
    /// Lazy init: connects only on the first `client()`, reused afterwards.
    client: tokio::sync::Mutex<Option<MCPClient>>,
    /// Last-used time (basis for idle-reap decisions).
    last_used: tokio::sync::Mutex<Instant>,
    /// Health breaker (P2-5): trips on consecutive failures + exponential backoff retry.
    breaker: tokio::sync::Mutex<CircuitBreaker>,
    /// Time of the last health probe (P2-5).
    last_probe: tokio::sync::Mutex<Option<Instant>>,
}

impl ManagedServer {
    fn new(spec: ServerSpec) -> Self {
        let max_failures = spec.max_failures;
        Self {
            spec,
            client: tokio::sync::Mutex::new(None),
            last_used: tokio::sync::Mutex::new(Instant::now()),
            breaker: tokio::sync::Mutex::new(CircuitBreaker::new(max_failures)),
            last_probe: tokio::sync::Mutex::new(None),
        }
    }

    /// Lazily gets the client: connects on the first call, reuses afterwards; also refreshes the last-used time.
    ///
    /// Breaker gating (P2-5): `Open` with the backoff still running → fast-fail (no more requests to a
    /// broken Server). Connect success/failure feeds the breaker — success recovers, failure advances the
    /// failure count.
    async fn client(&self) -> Result<MCPClient, MCPError> {
        {
            let breaker = self.breaker.lock().await;
            if !breaker.allow_request() {
                return Err(MCPError::new(
                    -1,
                    format!(
                        "MCP server '{}' is circuit-broken, refusing connections during backoff",
                        self.spec.name
                    ),
                ));
            }
        }
        let mut guard = self.client.lock().await;
        if guard.is_none() {
            log::debug!(
                target: "lc_mcp::connection_manager",
                "server '{}' first use, lazily starting connection",
                self.spec.name
            );
            match MCPClient::connect(self.spec.config.clone()).await {
                Ok(c) => {
                    self.breaker.lock().await.record_success();
                    *guard = Some(c);
                }
                Err(e) => {
                    self.breaker.lock().await.record_failure();
                    return Err(e);
                }
            }
        }
        *self.last_used.lock().await = Instant::now();
        Ok(guard
            .as_ref()
            .ok_or_else(|| MCPError::new(-1, "client not initialized".to_string()))?
            .clone())
    }

    /// Health probe (P2-5): `list_tools` acts as the probe; the result feeds the breaker.
    ///
    /// A connect failure was already recorded inside `client()`; do not count it again here.
    async fn probe(&self) -> Result<(), MCPError> {
        *self.last_probe.lock().await = Some(Instant::now());
        let client = match self.client().await {
            Ok(c) => c,
            Err(e) => return Err(e),
        };
        let result = probe_health(&client).await;
        let mut breaker = self.breaker.lock().await;
        if result.is_ok() {
            breaker.record_success();
        } else {
            breaker.record_failure();
        }
        result
    }

    /// Derives the current health status from the breaker (P2-5).
    async fn status(&self) -> HealthStatus {
        self.breaker.lock().await.health_status()
    }

    /// Idle duration (since the last use).
    async fn idle_duration(&self) -> Duration {
        self.last_used.lock().await.elapsed()
    }

    /// Closes the connection and releases the client (idle reap / explicit deactivation). No-op if not connected.
    async fn close(&self) -> Result<(), MCPError> {
        let mut guard = self.client.lock().await;
        if let Some(client) = guard.take() {
            let result = client.close().await;
            // Reset the timer after release so it isn't counted twice in the same reap round.
            *self.last_used.lock().await = Instant::now();
            result
        } else {
            Ok(())
        }
    }
}

/// MCP multi-Server connection manager.
///
/// `register` lazily records → `client(name)` connects on first call → background idle reaping.
/// On `Drop`, sends a shutdown signal to the reaper task, which then exits.
pub struct ConnectionManager {
    /// Registry: name → managed Server.
    servers: Arc<RwLock<HashMap<String, Arc<ManagedServer>>>>,
    /// Idle-reap scan interval.
    _reap_interval: Duration,
    /// Reaper-task shutdown signal; sent on `shutdown()` or Drop.
    shutdown_tx: watch::Sender<bool>,
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionManager {
    /// Creates the connection manager, starting the background idle-reap task.
    pub fn new() -> Self {
        Self::with_reap_interval(DEFAULT_REAP_INTERVAL)
    }

    /// Creates the connection manager with a custom reap scan interval.
    pub fn with_reap_interval(reap_interval: Duration) -> Self {
        let servers = Arc::new(RwLock::new(HashMap::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let reaper_servers = servers.clone();
        let mut reaper_shutdown = shutdown_rx;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Exits on a shutdown signal (false→managed) or when the sender is dropped (Closed → Err).
                    changed = reaper_shutdown.changed() => {
                        if changed.is_err() || *reaper_shutdown.borrow() {
                            break;
                        }
                    }
                    _ = sleep(reap_interval) => {
                        reap_idle(&reaper_servers).await;
                    }
                }
            }
        });

        Self {
            servers,
            _reap_interval: reap_interval,
            shutdown_tx,
        }
    }

    /// Registers a managed Server (lazily; no connection yet).
    ///
    /// Re-registering the same name returns an error, preventing accidental overwrite.
    pub async fn register(&self, spec: ServerSpec) -> Result<(), MCPError> {
        let mut map = self.servers.write().await;
        if map.contains_key(&spec.name) {
            return Err(MCPError::new(
                -1,
                format!("MCP server '{}' is already registered", spec.name),
            ));
        }
        log::debug!(
            target: "lc_mcp::connection_manager",
            "register server '{}' (keep_alive={}, max_idle={:?})",
            spec.name,
            spec.keep_alive,
            spec.max_idle
        );
        map.insert(spec.name.clone(), Arc::new(ManagedServer::new(spec)));
        Ok(())
    }

    /// Gets a Server's client (lazily connects on first call, reuses afterwards).
    pub async fn client(&self, name: &str) -> Result<MCPClient, MCPError> {
        let map = self.servers.read().await;
        let server = map
            .get(name)
            .ok_or_else(|| MCPError::new(-1, format!("MCP server '{name}' is not registered")))?;
        server.client().await
    }

    /// Explicitly closes and releases a Server's connection (rebuilt lazily on the next `client`).
    pub async fn release(&self, name: &str) -> Result<(), MCPError> {
        let map = self.servers.read().await;
        if let Some(server) = map.get(name) {
            server.close().await
        } else {
            Ok(())
        }
    }

    /// Unregisters and closes a Server, removing it from the registry.
    pub async fn unregister(&self, name: &str) -> Result<(), MCPError> {
        let mut map = self.servers.write().await;
        if let Some(server) = map.remove(name) {
            server.close().await
        } else {
            Ok(())
        }
    }

    /// Number of registered Servers.
    pub async fn len(&self) -> usize {
        self.servers.read().await.len()
    }

    /// Whether the registry is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Manually triggers one idle-reap round, returning the number of reaped Servers.
    ///
    /// `keep_alive` Servers are exempt; those idle beyond `max_idle` are closed and counted.
    pub async fn reap_idle(&self) -> usize {
        reap_idle(&self.servers).await
    }

    /// Health probe (P2-5): runs one `list_tools` probe on the Server and returns a health snapshot.
    ///
    /// The probe result feeds the breaker — after `max_failures` consecutive failures the status goes
    /// `Down`, and `client()` then fast-fails while the backoff is running. Unregistered Servers return
    /// an error.
    pub async fn health(&self, name: &str) -> Result<ServerHealth, MCPError> {
        let map = self.servers.read().await;
        let server = map
            .get(name)
            .ok_or_else(|| MCPError::new(-1, format!("MCP server '{name}' is not registered")))?;
        let _ = server.probe().await; // Trigger one probe; the breaker records it internally
        let status = server.status().await;
        let failures = server.breaker.lock().await.failures();
        let last_check = *server.last_probe.lock().await;
        Ok(ServerHealth {
            status,
            failures,
            last_check,
            max_failures: server.spec.max_failures,
        })
    }

    /// Removes all tripped Servers (P2-5), returning the names of the removed Servers.
    ///
    /// Servers tripped by consecutive failures (`BreakerState::Open`) are removed from the registry and
    /// closed; the caller can re-register by name (a new registration resets the breaker count).
    pub async fn reap_unhealthy(&self) -> Vec<String> {
        let mut removed = Vec::new();
        {
            let map = self.servers.read().await;
            for server in map.values() {
                if server.breaker.lock().await.state() == BreakerState::Open {
                    removed.push(server.spec.name.clone());
                }
            }
        }
        for name in &removed {
            log::info!(
                target: "lc_mcp::connection_manager",
                "reap unhealthy server '{}'",
                name
            );
            let _ = self.unregister(name).await;
        }
        removed
    }

    /// Closes all connections and stops the background reaper task.
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
        let map = self.servers.read().await;
        for server in map.values() {
            let _ = server.close().await;
        }
    }
}

/// Scans the registry, reaping non-`keep_alive` connections idle beyond the threshold.
async fn reap_idle(servers: &Arc<RwLock<HashMap<String, Arc<ManagedServer>>>>) -> usize {
    let mut to_reap = Vec::new();
    {
        let map = servers.read().await;
        for server in map.values() {
            if server.spec.keep_alive {
                continue;
            }
            if server.idle_duration().await >= server.spec.max_idle {
                to_reap.push(server.clone());
            }
        }
    }
    let mut reaped = 0usize;
    for server in to_reap {
        log::info!(
            target: "lc_mcp::connection_manager",
            "reap idle connection for server '{}'",
            server.spec.name
        );
        if server.close().await.is_ok() {
            reaped += 1;
        }
    }
    reaped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lazy startup: register doesn't connect (a connect failure is fine); only the first client() attempts to connect.
    #[tokio::test]
    async fn test_lazy_start_register_does_not_spawn() {
        let manager = ConnectionManager::new();
        // The command can't exist — if register spawned, this would fail here.
        let spec = ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]));
        manager
            .register(spec)
            .await
            .expect("register should not spawn a connection");
        assert_eq!(manager.len().await, 1);

        // Only the first client() actually attempts to spawn → command missing → Err.
        let result = manager.client("bad").await;
        assert!(
            result.is_err(),
            "lazy connect should fail because the command does not exist"
        );
    }

    /// Re-registering the same Server name errors.
    #[tokio::test]
    async fn test_register_duplicate_rejected() {
        let manager = ConnectionManager::new();
        let spec = ServerSpec::new("dup", MCPConfig::sse("http://localhost:1/sse"));
        manager
            .register(spec.clone())
            .await
            .expect("first register should succeed");
        let err = manager.register(spec).await.unwrap_err();
        assert!(err.to_string().contains("already registered"), "{}", err);
    }

    /// Getting a client for an unregistered Server errors.
    #[tokio::test]
    async fn test_client_unknown_server_errors() {
        let manager = ConnectionManager::new();
        let result = manager.client("ghost").await;
        match result {
            Err(e) => assert!(e.to_string().contains("not registered"), "{}", e),
            Ok(_) => panic!("unknown server should error"),
        }
    }

    /// Idle reaping: non-keep_alive Servers with max_idle zero are reaped;
    /// keep_alive ones are exempt.
    #[tokio::test]
    async fn test_reap_idle_respects_keep_alive() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("idle", MCPConfig::sse("http://localhost:1/sse"))
                    .with_max_idle(Duration::ZERO),
            )
            .await
            .expect("register idle server");
        manager
            .register(
                ServerSpec::new("sticky", MCPConfig::sse("http://localhost:1/sse"))
                    .keep_alive()
                    .with_max_idle(Duration::ZERO),
            )
            .await
            .expect("register keep_alive server");

        // idle isn't connected, so close is a no-op, but it still counts as reaped for being idle beyond the threshold.
        let reaped = manager.reap_idle().await;
        assert_eq!(reaped, 1, "non keep_alive idle server should be reaped");
        // keep_alive is exempt, not reaped.
        assert_eq!(
            manager.len().await,
            2,
            "registry should be unaffected by reaping"
        );
    }

    /// release is idempotent: releasing a not-yet-connected Server doesn't error.
    #[tokio::test]
    async fn test_release_unconnected_is_noop() {
        let manager = ConnectionManager::new();
        manager
            .register(ServerSpec::new(
                "x",
                MCPConfig::sse("http://localhost:1/sse"),
            ))
            .await
            .expect("register should succeed");
        manager
            .release("x")
            .await
            .expect("release without connection should be a no-op");
        manager
            .release("missing")
            .await
            .expect("release of unknown server should be a no-op");
    }

    /// Registry capacity and unregistration.
    #[tokio::test]
    async fn test_len_and_unregister() {
        let manager = ConnectionManager::new();
        for i in 0..3 {
            let spec = ServerSpec::new(format!("s{i}"), MCPConfig::sse("http://localhost:1/sse"));
            manager.register(spec).await.unwrap();
        }
        assert_eq!(manager.len().await, 3);
        manager
            .unregister("s1")
            .await
            .expect("unregister should succeed");
        assert_eq!(manager.len().await, 2);
        assert!(!manager.is_empty().await);
    }

    /// Health probe: consecutive failures escalate Degraded → Down at the threshold (P2-5).
    ///
    /// A nonexistent command fails to connect immediately (fast), no real Server needed; with
    /// `max_failures=2` the status should turn Down after two probes.
    #[tokio::test]
    async fn test_health_probe_tracks_degraded_then_down() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(2),
            )
            .await
            .expect("register should succeed");

        let h1 = manager
            .health("bad")
            .await
            .expect("health probe should not error");
        assert_eq!(h1.status, HealthStatus::Degraded, "1 failure -> Degraded");
        assert_eq!(h1.failures, 1);
        assert!(h1.last_check.is_some(), "probe should record the time");

        let h2 = manager
            .health("bad")
            .await
            .expect("health probe should not error");
        assert_eq!(
            h2.status,
            HealthStatus::Down,
            "2 consecutive failures -> Down"
        );
        assert_eq!(h2.failures, 2);
    }

    /// After tripping, `client()` fast-fails, no longer issuing requests to the broken Server (P2-5).
    #[tokio::test]
    async fn test_client_blocked_when_circuit_open() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(1),
            )
            .await
            .expect("register should succeed");

        // One failure trips the breaker.
        let health = manager
            .health("bad")
            .await
            .expect("health probe should not error");
        assert_eq!(health.status, HealthStatus::Down);

        let err = manager
            .client("bad")
            .await
            .err()
            .expect("should error while circuit is open");
        assert!(err.to_string().contains("circuit"), "{}", err);
    }

    /// Removes tripped Servers (P2-5): healthy ones are unaffected; tripped ones are removed and their names returned.
    #[tokio::test]
    async fn test_reap_unhealthy_removes_down_servers() {
        let manager = ConnectionManager::new();
        // "ok" never probes, so its breaker stays Closed.
        manager
            .register(ServerSpec::new(
                "ok",
                MCPConfig::stdio("no_such_cmd_xyz", vec![]),
            ))
            .await
            .expect("register ok");
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(1),
            )
            .await
            .expect("register bad");

        manager
            .health("bad")
            .await
            .expect("should trigger circuit breaker");
        assert_eq!(manager.len().await, 2);

        let removed = manager.reap_unhealthy().await;
        assert_eq!(
            removed,
            vec!["bad".to_string()],
            "should reap the circuit-broken bad"
        );
        assert_eq!(
            manager.len().await,
            1,
            "bad should be removed from the registry"
        );
        assert!(
            manager.health("ok").await.is_ok(),
            "ok should be unaffected"
        );
    }

    /// Health-probing an unregistered Server errors.
    #[tokio::test]
    async fn test_health_unknown_server_errors() {
        let manager = ConnectionManager::new();
        let err = manager.health("ghost").await.unwrap_err();
        assert!(err.to_string().contains("not registered"), "{}", err);
    }
}
