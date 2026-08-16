//! MCP 多 Server 连接管理(P2-1):惰性启动 + 空闲回收 + 连接池。
//!
//! 100+ Server 直接各自 `MCPClient::connect` = 上百个子进程/长连接,内存与
//! FD 双耗尽。本模块提供一个托管注册表:
//!
//! - **惰性启动**:`register` 只登记 `ServerSpec`,首次 `client(name)` 才真正
//!   spawn 子进程 / 建 SSE 连接,未用到的 Server 零成本。
//! - **空闲回收**:后台 task 周期性扫描,非 `keep_alive` 的 Server 空闲超过
//!   `max_idle` 即 `close()` 释放连接;有状态 Server 标记 `keep_alive` 豁免。
//! - **连接池**:同一 `ManagedServer` 的 `client()` 幂等,后续调用复用连接,
//!   不重复 spawn。
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
//! // 首次调用才惰性启动连接
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

/// 默认空闲回收扫描周期。
const DEFAULT_REAP_INTERVAL: Duration = Duration::from_secs(60);

/// 单个托管 Server 的声明。
#[derive(Debug, Clone)]
pub struct ServerSpec {
    /// Server 名称(注册表 key / 工具命名空间前缀)。
    pub name: String,
    /// 连接配置(Stdio / SSE)。
    pub config: MCPConfig,
    /// 有状态 Server 标记:空闲不回收(默认 false)。
    pub keep_alive: bool,
    /// 空闲回收阈值:超过该时长未被使用则关闭连接释放资源。
    pub max_idle: Duration,
    /// 健康熔断阈值(P2-5):连续失败 N 次熔断摘除(默认 3)。
    pub max_failures: u32,
}

impl ServerSpec {
    /// 创建一个托管 Server 声明。
    pub fn new(name: impl Into<String>, config: MCPConfig) -> Self {
        Self {
            name: name.into(),
            config,
            keep_alive: false,
            max_idle: Duration::from_secs(300),
            max_failures: 3,
        }
    }

    /// 标记有状态 Server:空闲不回收。
    pub fn keep_alive(mut self) -> Self {
        self.keep_alive = true;
        self
    }

    /// 设置空闲回收阈值。
    pub fn with_max_idle(mut self, max_idle: Duration) -> Self {
        self.max_idle = max_idle;
        self
    }

    /// 设置健康熔断阈值(P2-5):连续失败 N 次后熔断,拒绝请求直到退避结束。
    pub fn with_max_failures(mut self, max_failures: u32) -> Self {
        self.max_failures = max_failures.max(1);
        self
    }
}

/// 托管 Server 的运行态:惰性连接 + 最近使用时间 + 健康熔断器。
struct ManagedServer {
    spec: ServerSpec,
    /// 惰性初始化:首次 `client()` 才 connect,之后复用。
    client: tokio::sync::Mutex<Option<MCPClient>>,
    /// 最近一次使用时间(空闲回收判定依据)。
    last_used: tokio::sync::Mutex<Instant>,
    /// 健康熔断器(P2-5):连续失败熔断 + 指数退避重试。
    breaker: tokio::sync::Mutex<CircuitBreaker>,
    /// 最近一次探活时间(P2-5)。
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

    /// 惰性获取客户端:首次调用才建连,后续复用;同时刷新最近使用时间。
    ///
    /// 熔断门控(P2-5):`Open` 且退避期未过 → 快速失败(不再往坏 Server 上打)。
    /// 建连成败记入熔断器——成功恢复,失败推进失败计数。
    async fn client(&self) -> Result<MCPClient, MCPError> {
        {
            let breaker = self.breaker.lock().await;
            if !breaker.allow_request() {
                return Err(MCPError::new(
                    -1,
                    format!("MCP server '{}' 熔断中,退避期拒绝连接", self.spec.name),
                ));
            }
        }
        let mut guard = self.client.lock().await;
        if guard.is_none() {
            log::debug!(
                target: "lc_mcp::connection_manager",
                "server '{}' 首次使用,惰性启动连接",
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
            .ok_or_else(|| MCPError::new(-1, "client 未初始化".to_string()))?
            .clone())
    }

    /// 健康探活(P2-5):`list_tools` 即探活,结果记入熔断器。
    ///
    /// 建连失败已在 `client()` 内记录一次,这里不重复计数。
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

    /// 由熔断器推导当前健康状态(P2-5)。
    async fn status(&self) -> HealthStatus {
        self.breaker.lock().await.health_status()
    }

    /// 空闲时长(距最近一次使用)。
    async fn idle_duration(&self) -> Duration {
        self.last_used.lock().await.elapsed()
    }

    /// 关闭连接并释放客户端(空闲回收 / 显式停用)。未建连则无操作。
    async fn close(&self) -> Result<(), MCPError> {
        let mut guard = self.client.lock().await;
        if let Some(client) = guard.take() {
            let result = client.close().await;
            // 释放后重置计时,避免同一轮回收里被重复计数。
            *self.last_used.lock().await = Instant::now();
            result
        } else {
            Ok(())
        }
    }
}

/// MCP 多 Server 连接管理器。
///
/// `register` 惰性登记 → `client(name)` 首次调用才建连 → 后台空闲回收。
/// `Drop` 时向回收 task 发关闭信号,回收 task 退出。
pub struct ConnectionManager {
    /// 注册表:name → 托管 Server。
    servers: Arc<RwLock<HashMap<String, Arc<ManagedServer>>>>,
    /// 空闲回收扫描周期。
    _reap_interval: Duration,
    /// 回收 task 关闭信号;`shutdown()` 或 Drop 时发送。
    shutdown_tx: watch::Sender<bool>,
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionManager {
    /// 创建连接管理器,后台启动空闲回收 task。
    pub fn new() -> Self {
        Self::with_reap_interval(DEFAULT_REAP_INTERVAL)
    }

    /// 创建连接管理器并自定义回收扫描周期。
    pub fn with_reap_interval(reap_interval: Duration) -> Self {
        let servers = Arc::new(RwLock::new(HashMap::new()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let reaper_servers = servers.clone();
        let mut reaper_shutdown = shutdown_rx;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // 关闭信号(false→管理)或发送方被 Drop(Closed → Err)都退出。
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

    /// 登记一个托管 Server(惰性,不建连)。
    ///
    /// 同名重复登记返回错误,防止误覆盖。
    pub async fn register(&self, spec: ServerSpec) -> Result<(), MCPError> {
        let mut map = self.servers.write().await;
        if map.contains_key(&spec.name) {
            return Err(MCPError::new(
                -1,
                format!("MCP server '{}' 已注册", spec.name),
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

    /// 获取某个 Server 的客户端(首次调用惰性建连,后续复用)。
    pub async fn client(&self, name: &str) -> Result<MCPClient, MCPError> {
        let map = self.servers.read().await;
        let server = map
            .get(name)
            .ok_or_else(|| MCPError::new(-1, format!("MCP server '{name}' 未注册")))?;
        server.client().await
    }

    /// 显式关闭并释放某个 Server 的连接(下次 `client` 惰性重建)。
    pub async fn release(&self, name: &str) -> Result<(), MCPError> {
        let map = self.servers.read().await;
        if let Some(server) = map.get(name) {
            server.close().await
        } else {
            Ok(())
        }
    }

    /// 注销并关闭某个 Server,从注册表移除。
    pub async fn unregister(&self, name: &str) -> Result<(), MCPError> {
        let mut map = self.servers.write().await;
        if let Some(server) = map.remove(name) {
            server.close().await
        } else {
            Ok(())
        }
    }

    /// 已登记的 Server 数量。
    pub async fn len(&self) -> usize {
        self.servers.read().await.len()
    }

    /// 是否为空注册表。
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// 手动触发一轮空闲回收,返回回收的 Server 数。
    ///
    /// `keep_alive` 的 Server 豁免;空闲超过 `max_idle` 的关闭连接并计数。
    pub async fn reap_idle(&self) -> usize {
        reap_idle(&self.servers).await
    }

    /// 健康探活(P2-5):对该 Server 做一次 `list_tools` 探活并返回健康快照。
    ///
    /// 探活结果记入熔断器——连续失败达 `max_failures` 后状态转 `Down`,
    /// 此后 `client()` 在退避期内快速失败。未注册的 Server 返回错误。
    pub async fn health(&self, name: &str) -> Result<ServerHealth, MCPError> {
        let map = self.servers.read().await;
        let server = map
            .get(name)
            .ok_or_else(|| MCPError::new(-1, format!("MCP server '{name}' 未注册")))?;
        let _ = server.probe().await; // 触发一次探活,内部记录熔断
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

    /// 摘除所有已熔断的 Server(P2-5),返回被摘除的 Server 名列表。
    ///
    /// 连续失败触发熔断(`BreakerState::Open`)的 Server 从注册表移除并关闭连接;
    /// 调用方可据名称重新登记(新注册会重置熔断计数)。
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

    /// 关闭全部连接并停止后台回收 task。
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
        let map = self.servers.read().await;
        for server in map.values() {
            let _ = server.close().await;
        }
    }
}

/// 扫描注册表,回收空闲超阈值的非 `keep_alive` 连接。
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

    /// 惰性启动:register 不建连,连不上也无妨;首次 client() 才尝试建连。
    #[tokio::test]
    async fn test_lazy_start_register_does_not_spawn() {
        let manager = ConnectionManager::new();
        // 命令必然不存在——若 register 就 spawn,这里会失败。
        let spec = ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]));
        manager.register(spec).await.expect("register 不应建连");
        assert_eq!(manager.len().await, 1);

        // 首次 client() 才真正尝试 spawn → 命令不存在 → Err。
        let result = manager.client("bad").await;
        assert!(result.is_err(), "惰性建连应因命令不存在而失败");
    }

    /// 重复注册同名 Server 报错。
    #[tokio::test]
    async fn test_register_duplicate_rejected() {
        let manager = ConnectionManager::new();
        let spec = ServerSpec::new("dup", MCPConfig::sse("http://localhost:1/sse"));
        manager.register(spec.clone()).await.expect("首次登记成功");
        let err = manager.register(spec).await.unwrap_err();
        assert!(err.to_string().contains("已注册"), "{}", err);
    }

    /// 未注册的 Server 取客户端报错。
    #[tokio::test]
    async fn test_client_unknown_server_errors() {
        let manager = ConnectionManager::new();
        let result = manager.client("ghost").await;
        match result {
            Err(e) => assert!(e.to_string().contains("未注册"), "{}", e),
            Ok(_) => panic!("未知 server 应报错"),
        }
    }

    /// 空闲回收:max_idle 为零的非 keep_alive Server 被回收;
    /// keep_alive 的豁免。
    #[tokio::test]
    async fn test_reap_idle_respects_keep_alive() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("idle", MCPConfig::sse("http://localhost:1/sse"))
                    .with_max_idle(Duration::ZERO),
            )
            .await
            .expect("登记 idle server");
        manager
            .register(
                ServerSpec::new("sticky", MCPConfig::sse("http://localhost:1/sse"))
                    .keep_alive()
                    .with_max_idle(Duration::ZERO),
            )
            .await
            .expect("登记 keep_alive server");

        // idle 未建连,close 为无操作,但仍按"空闲超阈值"计入回收。
        let reaped = manager.reap_idle().await;
        assert_eq!(reaped, 1, "非 keep_alive 的 idle 应被回收");
        // keep_alive 豁免,不被回收。
        assert_eq!(manager.len().await, 2, "注册表不受回收影响");
    }

    /// release 幂等:未建连的 Server release 不报错。
    #[tokio::test]
    async fn test_release_unconnected_is_noop() {
        let manager = ConnectionManager::new();
        manager
            .register(ServerSpec::new(
                "x",
                MCPConfig::sse("http://localhost:1/sse"),
            ))
            .await
            .expect("登记成功");
        manager.release("x").await.expect("未建连 release 无操作");
        manager
            .release("missing")
            .await
            .expect("未知 server release 无操作");
    }

    /// 注册表容量与注销。
    #[tokio::test]
    async fn test_len_and_unregister() {
        let manager = ConnectionManager::new();
        for i in 0..3 {
            let spec = ServerSpec::new(format!("s{i}"), MCPConfig::sse("http://localhost:1/sse"));
            manager.register(spec).await.unwrap();
        }
        assert_eq!(manager.len().await, 3);
        manager.unregister("s1").await.expect("注销成功");
        assert_eq!(manager.len().await, 2);
        assert!(!manager.is_empty().await);
    }

    /// 健康探活:连续失败递增 Degraded → 达阈值 Down(P2-5)。
    ///
    /// 不存在的命令建连即失败(快速),无需真实 Server;`max_failures=2` 时
    /// 两次探活后状态应转 Down。
    #[tokio::test]
    async fn test_health_probe_tracks_degraded_then_down() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(2),
            )
            .await
            .expect("登记成功");

        let h1 = manager.health("bad").await.expect("健康探活不报错");
        assert_eq!(h1.status, HealthStatus::Degraded, "1 次失败 → Degraded");
        assert_eq!(h1.failures, 1);
        assert!(h1.last_check.is_some(), "探活应记录时间");

        let h2 = manager.health("bad").await.expect("健康探活不报错");
        assert_eq!(h2.status, HealthStatus::Down, "2 次连续失败 → Down");
        assert_eq!(h2.failures, 2);
    }

    /// 熔断后 `client()` 快速失败,不再向坏 Server 发起请求(P2-5)。
    #[tokio::test]
    async fn test_client_blocked_when_circuit_open() {
        let manager = ConnectionManager::new();
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(1),
            )
            .await
            .expect("登记成功");

        // 一次失败即熔断。
        let health = manager.health("bad").await.expect("健康探活不报错");
        assert_eq!(health.status, HealthStatus::Down);

        let err = manager.client("bad").await.err().expect("熔断期应报错");
        assert!(err.to_string().contains("熔断"), "{}", err);
    }

    /// 摘除熔断的 Server(P2-5):健康的不受影响,熔断的被移除并返回其名。
    #[tokio::test]
    async fn test_reap_unhealthy_removes_down_servers() {
        let manager = ConnectionManager::new();
        // "ok" 从不探活,熔断器保持 Closed。
        manager
            .register(ServerSpec::new(
                "ok",
                MCPConfig::stdio("no_such_cmd_xyz", vec![]),
            ))
            .await
            .expect("登记 ok");
        manager
            .register(
                ServerSpec::new("bad", MCPConfig::stdio("no_such_cmd_xyz", vec![]))
                    .with_max_failures(1),
            )
            .await
            .expect("登记 bad");

        manager.health("bad").await.expect("触发熔断");
        assert_eq!(manager.len().await, 2);

        let removed = manager.reap_unhealthy().await;
        assert_eq!(removed, vec!["bad".to_string()], "应摘除熔断的 bad");
        assert_eq!(manager.len().await, 1, "bad 已从注册表移除");
        assert!(manager.health("ok").await.is_ok(), "ok 不受影响");
    }

    /// 未注册的 Server 健康探活报错。
    #[tokio::test]
    async fn test_health_unknown_server_errors() {
        let manager = ConnectionManager::new();
        let err = manager.health("ghost").await.unwrap_err();
        assert!(err.to_string().contains("未注册"), "{}", err);
    }
}
