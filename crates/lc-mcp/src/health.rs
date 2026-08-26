//! MCP Server 健康检查 + 熔断器(P2-5)。
//!
//! 100+ Server 中任意一个都可能随时挂掉。本模块提供:
//!
//! - **心跳探活**:`list_tools` 即探活——能列出工具说明 Server 活着;
//! - **熔断摘除**:`CircuitBreaker` 连续 `N` 次失败后 Open(熔断),`client()`
//!   快速失败不再往坏 Server 上打请求;
//! - **指数退避重连**:熔断后按 0.5s → 1s → 2s → …(上限 30s)退避,退避结束
//!   进入半开探测窗口——`allow_request` 放行一次,成功即恢复、失败则下一档退避。
//!
//! [`ServerHealth`] 是单次健康快照,`ConnectionManager::health(name)` 返回它;
//! [`CircuitBreaker`] 是持久化的熔断状态机,挂在每个被托管的 Server 上。

use std::time::{Duration, Instant};

use crate::client::MCPClient;
use crate::protocol::MCPError;

/// 指数退避基数(0.5s)。
const BASE_BACKOFF: Duration = Duration::from_millis(500);
/// 退避上限:防等待时间无限增长。
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Server 健康状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// 正常(无失败)。
    Healthy,
    /// 出现过失败但仍可用(未达摘除阈值)。
    Degraded,
    /// 连续失败已达阈值,被熔断摘除。
    Down,
}

/// 单个 Server 的健康快照(P2-5)。
///
/// 由 [`ConnectionManager::health`](crate::connection_manager::ConnectionManager::health) 返回;`last_check` 为最近一次探活时间,
/// `failures` 为连续失败次数,`max_failures` 为触发 Down 的阈值。
#[derive(Debug, Clone)]
pub struct ServerHealth {
    /// 当前健康状态。
    pub status: HealthStatus,
    /// 连续失败次数(探活 + 建连失败累计)。
    pub failures: u32,
    /// 最近一次探活时间(尚未探过则为 `None`)。
    pub last_check: Option<Instant>,
    /// 连续失败 N 次判 Down(摘除阈值)。
    pub max_failures: u32,
}

impl ServerHealth {
    /// 构造初始健康快照。
    pub fn new(max_failures: u32) -> Self {
        Self {
            status: HealthStatus::Healthy,
            failures: 0,
            last_check: None,
            max_failures,
        }
    }

    /// 记录一次探活成功:恢复健康。
    pub fn record_success(&mut self) {
        self.status = HealthStatus::Healthy;
        self.failures = 0;
        self.last_check = Some(Instant::now());
    }

    /// 记录一次探活失败:递增失败计数,达阈值转 Down,否则 Degraded。
    pub fn record_failure(&mut self) {
        self.failures += 1;
        self.last_check = Some(Instant::now());
        self.status = if self.failures >= self.max_failures.max(1) {
            HealthStatus::Down
        } else {
            HealthStatus::Degraded
        };
    }

    /// 是否已熔断摘除。
    pub fn is_down(&self) -> bool {
        self.status == HealthStatus::Down
    }
}

/// 熔断器状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// 正常放行。
    Closed,
    /// 熔断:拒绝请求,直到退避时间到(进入半开探测窗口)。
    Open,
}

/// 指数退避:0.5s → 1s → 2s → 4s → … 上限 30s。
fn backoff_delay(step: u32) -> Duration {
    let ms = (BASE_BACKOFF.as_millis() as u64)
        .checked_shl(step.min(6))
        .unwrap_or(MAX_BACKOFF.as_millis() as u64)
        .min(MAX_BACKOFF.as_millis() as u64);
    Duration::from_millis(ms)
}

/// 熔断器(P2-5):连续失败熔断 + 指数退避 + 半开探测。
///
/// 由 [`ConnectionManager`](crate::connection_manager::ConnectionManager) 内每个 `ManagedServer` 持有一个。请求放行规则:
///
/// - `Closed`:放行(可正常建连/调用);
/// - `Open` 且未到退避时间:拒绝(快速失败,不打坏 Server);
/// - `Open` 且已到退避时间(半开窗口):放行一次探测,成功恢复、失败推进退避。
pub struct CircuitBreaker {
    /// 连续失败 N 次熔断。
    max_failures: u32,
    /// 当前连续失败次数。
    failures: u32,
    state: BreakerState,
    /// 下次允许探测的时间(退避截止)。
    next_retry_at: Option<Instant>,
    /// 最近一次熔断已武装的退避时长(指数退避)。
    backoff: Duration,
    /// 下一档退避步数(每次熔断递增,指数增长)。
    backoff_step: u32,
}

impl CircuitBreaker {
    /// 创建熔断器。
    pub fn new(max_failures: u32) -> Self {
        Self {
            max_failures: max_failures.max(1),
            failures: 0,
            state: BreakerState::Closed,
            next_retry_at: None,
            backoff: Duration::ZERO,
            backoff_step: 0,
        }
    }

    /// 当前熔断状态。
    pub fn state(&self) -> BreakerState {
        self.state
    }

    /// 当前连续失败次数。
    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// 是否允许发起请求。
    ///
    /// `Closed` 恒放行;`Open` 仅在退避时间已过(半开探测窗口)时放行一次。
    pub fn allow_request(&self) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::Open => self
                .next_retry_at
                .map(|t| Instant::now() >= t)
                .unwrap_or(true),
        }
    }

    /// 记录一次成功:恢复 `Closed`,重置失败计数与退避步数。
    pub fn record_success(&mut self) {
        self.state = BreakerState::Closed;
        self.failures = 0;
        self.backoff = Duration::ZERO;
        self.backoff_step = 0;
        self.next_retry_at = None;
    }

    /// 记录一次失败:递增失败计数;达阈值 → Open + 记下次退避重试时间。
    ///
    /// 半开探测失败时(已 Open),重新 Open 并推进退避步数(指数增长)。
    pub fn record_failure(&mut self) {
        self.failures += 1;
        if self.failures >= self.max_failures {
            self.state = BreakerState::Open;
            self.backoff = backoff_delay(self.backoff_step);
            self.next_retry_at = Some(Instant::now() + self.backoff);
            self.backoff_step += 1;
        }
    }

    /// 当前已武装的退避时长(最近一次熔断采用的指数退避)。
    pub fn backoff(&self) -> Duration {
        self.backoff
    }

    /// 由当前熔断状态推导健康状态。
    pub fn health_status(&self) -> HealthStatus {
        to_status(self.state, self.failures)
    }
}

/// 心跳探活:`list_tools` 即探活(P2-5)。
///
/// 能列出工具说明该 Server 至少能响应请求;探活失败由调用方
/// ([`CircuitBreaker::record_failure`]) 记入熔断计数。
pub async fn probe_health(client: &MCPClient) -> Result<(), MCPError> {
    client.list_tools().await.map(|_| ())
}

/// 由熔断状态推导健康状态。
fn to_status(state: BreakerState, failures: u32) -> HealthStatus {
    match state {
        BreakerState::Open => HealthStatus::Down,
        BreakerState::Closed if failures > 0 => HealthStatus::Degraded,
        BreakerState::Closed => HealthStatus::Healthy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::MCPConfig;

    /// `ServerHealth` 状态流转:Healthy → Degraded → Down → 恢复 Healthy。
    #[test]
    fn test_server_health_transitions() {
        let mut health = ServerHealth::new(2);
        assert_eq!(health.status, HealthStatus::Healthy);
        health.record_failure();
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.failures, 1);
        assert!(health.last_check.is_some());
        health.record_failure();
        assert_eq!(health.status, HealthStatus::Down);
        assert!(health.is_down());
        health.record_success();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.failures, 0);
    }

    /// 熔断器:达阈值才 Open;Open 后未到退避时间拒绝放行。
    #[test]
    fn test_circuit_breaker_trips_and_blocks() {
        let mut cb = CircuitBreaker::new(2);
        assert!(cb.allow_request());
        cb.record_failure();
        // 未达阈值,仍 Closed。
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
        cb.record_failure();
        // 达阈值 → Open,退避期内拒绝。
        assert_eq!(cb.state(), BreakerState::Open);
        assert!(
            !cb.allow_request(),
            "should reject requests while circuit is open"
        );
    }

    /// 指数退避:每档翻倍,上限 30s。
    #[test]
    fn test_circuit_breaker_exponential_backoff() {
        let mut cb = CircuitBreaker::new(1);
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(500));
        cb.record_failure(); // 半开探测失败 → 下一档
        assert_eq!(cb.backoff(), Duration::from_millis(1000));
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(2000));
        cb.record_failure();
        assert_eq!(cb.backoff(), Duration::from_millis(4000));
        // 步数 6 → 500 << 6 = 32000 → 封顶 30000。
        for _ in 0..4 {
            cb.record_failure();
        }
        assert_eq!(cb.backoff(), MAX_BACKOFF);
    }

    /// 恢复:成功调用关闭熔断,允许请求。
    #[test]
    fn test_circuit_breaker_recovers_on_success() {
        let mut cb = CircuitBreaker::new(1);
        cb.record_failure();
        assert_eq!(cb.state(), BreakerState::Open);
        cb.record_success();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_request());
        assert_eq!(cb.failures(), 0);
    }

    /// 最小阈值下限为 1。
    #[test]
    fn test_circuit_breaker_max_failures_min_one() {
        let mut cb = CircuitBreaker::new(0);
        cb.record_failure();
        assert_eq!(
            cb.state(),
            BreakerState::Open,
            "max_failures must be at least 1"
        );
    }

    /// 探活:对假 SSE Server 执行 list_tools 成功。
    #[tokio::test]
    async fn test_probe_health_with_fake_server() {
        let server = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&server.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");
        probe_health(&client)
            .await
            .expect("list_tools probe should succeed");
    }
}
