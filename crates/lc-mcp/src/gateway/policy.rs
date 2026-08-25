//! Server 策略与 Gateway 登记声明。

use std::sync::Arc;
use std::time::Duration;

use crate::connection_manager::ServerSpec;
use crate::sandbox::ServerSandbox;
use crate::tool_namespace::ToolConflict;
use crate::tool_timeout::ToolSpec;
use crate::types::MCPConfig;

/// 单个 Server 的 Gateway 策略(冲突 / 超时 / 沙箱 / 静态层)。
///
/// 速率限制不在策略里:限流器运行时状态存 `rate_limiters`,策略只需在 register
/// 时据此建限流器即可,无需重复存储。
#[derive(Debug, Clone)]
pub(crate) struct ServerPolicy {
    pub(crate) conflict: ToolConflict,
    pub(crate) timeout: Option<ToolSpec>,
    pub(crate) sandbox: Option<Arc<ServerSandbox>>,
    /// 该 Server 全部工具自动进静态层(P2-3)。
    pub(crate) pin_all: bool,
}

/// Gateway 登记一个 Server 的完整声明(P2-8)。
#[derive(Debug, Clone)]
pub struct GatewayServerSpec {
    /// Server 名称(注册表 key / 工具命名空间前缀)。
    pub name: String,
    /// 连接配置(Stdio / SSE)。
    pub config: MCPConfig,
    /// 有状态 Server:空闲不回收(默认 false)。
    pub keep_alive: bool,
    /// 空闲回收阈值。
    pub max_idle: Duration,
    /// 健康熔断阈值(默认 3)。
    pub max_failures: u32,
    /// 工具命名冲突策略(默认 [`ToolConflict::Prefix`])。
    pub conflict: ToolConflict,
    /// per-tool 默认超时(P2-4):该 Server 所有工具统一挂。
    pub default_timeout: Option<ToolSpec>,
    /// per-Server 安全沙箱(P2-6)。
    pub sandbox: Option<Arc<ServerSandbox>>,
    /// 速率限制(P2-8):`(max_calls, window)`,`None` 不限流。
    pub rate_limit: Option<(usize, Duration)>,
    /// 该 Server 全部工具进静态层常驻注入(P2-3)。
    pub pin_all: bool,
}

impl GatewayServerSpec {
    /// 创建一个 Gateway Server 声明。
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

    /// 设置健康熔断阈值。
    pub fn with_max_failures(mut self, max_failures: u32) -> Self {
        self.max_failures = max_failures.max(1);
        self
    }

    /// 设置工具命名冲突策略。
    pub fn with_conflict(mut self, conflict: ToolConflict) -> Self {
        self.conflict = conflict;
        self
    }

    /// 挂 per-tool 默认超时(P2-4),该 Server 所有工具生效。
    pub fn with_timeout(mut self, spec: ToolSpec) -> Self {
        self.default_timeout = Some(spec);
        self
    }

    /// 挂 per-Server 安全沙箱(P2-6)。
    pub fn with_sandbox(mut self, sandbox: Arc<ServerSandbox>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// 挂固定窗口速率限制(P2-8):`window` 内最多 `max_calls` 次。
    pub fn with_rate_limit(mut self, max_calls: usize, window: Duration) -> Self {
        self.rate_limit = Some((max_calls, window));
        self
    }

    /// 该 Server 全部工具进静态层常驻注入(P2-3)。
    pub fn pin_all_tools(mut self) -> Self {
        self.pin_all = true;
        self
    }

    /// 转成底层连接管理器的 ServerSpec(借用字段,config 克隆)。
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
