//! per-Server 安全隔离(P2-6)。
//!
//! 100+ Server 来源各异、权限边界不同,必须按 Server 独立收窄权限:
//!
//! - **进程层独立容器**:每个 Server 是独立子进程 / 独立连接,互不共享内存
//!   (P2-1 惰性连接天然隔离);
//! - **权限层独立凭证**:`ServerSpec` 各自携带自己的 config/凭证,无跨 Server
//!   共享;
//! - **参数级最小权限**:[`ParamRule`] 在工具调用参数上做约束——文件 Server
//!   只允许 `file:///tmp` 前缀、格式只允许枚举值、拒绝路径穿越子串,违反即拦截;
//! - **网络层出站白名单**:[`EgressPolicy`] 声明该 Server 可访问的主机,空白名单
//!   即全禁(fail-closed);
//! - **审计层全量记录**:[`ServerSandbox`] 记录每次放行/拦截调用,供事后审计。
//!
//! [`MCPToolAdapter::with_sandbox`] 把 `ServerSandbox` 挂到工具适配器上,`run()`
//! 在发请求前先过 `check_call`,拦截则返回错误并记审计。

use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// 参数级最小权限规则(P2-6)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRule {
    /// 字符串参数必须以指定前缀开头(最小权限前缀白名单,
    /// 如文件 Server 只允许 `file:///tmp/`)。
    Prefix { field: String, prefix: String },
    /// 字符串参数必须落在允许集合内(枚举白名单)。
    Enum { field: String, allowed: Vec<String> },
    /// 字符串参数禁止包含指定子串(如路径穿越 `..`、危险命令)。
    RejectContains {
        field: String,
        forbidden: Vec<String>,
    },
}

impl ParamRule {
    /// 对一次工具调用参数校验;返回违规原因。
    fn check(&self, arguments: &Value) -> Result<(), String> {
        let obj = match arguments {
            Value::Object(m) => m,
            // 非对象参数(无字段可校验):fail-closed,拦截以保最小权限。
            _ => return Err("参数必须是 JSON 对象才能做最小权限校验".to_string()),
        };
        match self {
            ParamRule::Prefix { field, prefix } => {
                let v = obj
                    .get(field)
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("缺少字符串参数 '{field}'"))?;
                if v.starts_with(prefix) {
                    Ok(())
                } else {
                    Err(format!(
                        "参数 '{field}' 的值 '{v}' 未以最小权限前缀 '{prefix}' 开头"
                    ))
                }
            }
            ParamRule::Enum { field, allowed } => {
                let v = obj
                    .get(field)
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("缺少字符串参数 '{field}'"))?;
                if allowed.iter().any(|a| a == v) {
                    Ok(())
                } else {
                    Err(format!("参数 '{field}' 的值 '{v}' 不在允许集合内"))
                }
            }
            ParamRule::RejectContains { field, forbidden } => {
                let v = obj
                    .get(field)
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("缺少字符串参数 '{field}'"))?;
                if let Some(bad) = forbidden.iter().find(|b| v.contains(b.as_str())) {
                    Err(format!("参数 '{field}' 包含被禁止的子串 '{bad}'"))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// 网络层出站白名单(P2-6):该 Server 允许访问的主机。
///
/// 空策略 = 禁止一切出站(fail-closed)。`allows` 支持子域:
/// 允许 `example.com` 即放行 `api.example.com`,但不放行 `evil-example.com`。
#[derive(Debug, Clone, Default)]
pub struct EgressPolicy {
    allowed: Vec<String>,
}

impl EgressPolicy {
    /// 空白名单(禁止一切出站)。
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一个允许访问的主机名。
    pub fn allow(mut self, host: impl Into<String>) -> Self {
        self.allowed.push(host.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    /// 是否允许访问该主机(大小写不敏感,支持子域通配)。
    pub fn allows(&self, host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        self.allowed.iter().any(|a| {
            let a = a.to_ascii_lowercase();
            host == a || host.ends_with(&format!(".{a}"))
        })
    }
}

/// 审计记录(P2-6):一次放行/拦截的调用。
#[derive(Debug, Clone)]
pub struct AuditRecord {
    /// 所属 Server。
    pub server: String,
    /// 被调用的工具。
    pub tool: String,
    /// 工具调用参数(全量)。
    pub arguments: Value,
    /// 是否放行。
    pub allowed: bool,
    /// 拦截原因(`allowed` 为 false 时有值)。
    pub reason: Option<String>,
    /// 记录时间。
    pub at: SystemTime,
}

/// 沙箱拦截错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    pub reason: String,
}

impl SandboxError {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sandbox blocked: {}", self.reason)
    }
}

impl std::error::Error for SandboxError {}

/// per-Server 安全沙箱(P2-6):参数最小权限 + 出站白名单 + 全量审计。
///
/// 字段均为 `Arc`/`Mutex`,可廉价 `Clone` 分发给同一 Server 的多个工具适配器,
/// 共享同一份规则与审计日志。
#[derive(Debug, Clone)]
pub struct ServerSandbox {
    server: String,
    /// 参数级最小权限规则。
    param_rules: Arc<Vec<ParamRule>>,
    /// 网络层出站白名单。
    egress: Arc<EgressPolicy>,
    /// 全量审计日志(环形,上限 `max_audit`)。
    audit: Arc<Mutex<VecDeque<AuditRecord>>>,
    /// 审计日志上限(默认 1000)。
    max_audit: usize,
}

impl ServerSandbox {
    /// 创建一个 per-Server 安全沙箱(默认无参数规则:放行;出站全禁)。
    pub fn new(server: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            param_rules: Arc::new(Vec::new()),
            egress: Arc::new(EgressPolicy::new()),
            audit: Arc::new(Mutex::new(VecDeque::new())),
            max_audit: 1000,
        }
    }

    /// 追加一条参数级最小权限规则。
    pub fn with_param_rule(mut self, rule: ParamRule) -> Self {
        Arc::make_mut(&mut self.param_rules).push(rule);
        self
    }

    /// 追加一个出站白名单主机。
    pub fn allow_host(mut self, host: impl Into<String>) -> Self {
        let mut policy = Arc::make_mut(&mut self.egress).clone();
        policy.allowed.push(host.into());
        self.egress = Arc::new(policy);
        self
    }

    /// 替换整个出站白名单。
    pub fn with_egress(mut self, policy: EgressPolicy) -> Self {
        self.egress = Arc::new(policy);
        self
    }

    /// 设置审计日志上限(最少 1)。
    pub fn with_max_audit(mut self, max: usize) -> Self {
        self.max_audit = max.max(1);
        self
    }

    /// 校验一次工具调用(参数级最小权限),放行记 Allowed、拦截记 Blocked。
    pub fn check_call(&self, tool: &str, arguments: &Value) -> Result<(), SandboxError> {
        for rule in self.param_rules.iter() {
            if let Err(reason) = rule.check(arguments) {
                self.record(tool, arguments.clone(), false, Some(reason.clone()));
                return Err(SandboxError::new(reason));
            }
        }
        self.record(tool, arguments.clone(), true, None);
        Ok(())
    }

    /// 校验出站目标是否在白名单内(网络层)。
    pub fn check_egress(&self, tool: &str, host: &str) -> Result<(), SandboxError> {
        if !self.egress.allows(host) {
            let reason = format!("出站目标 '{host}' 不在白名单内");
            self.record(tool, Value::Null, false, Some(reason.clone()));
            return Err(SandboxError::new(reason));
        }
        self.record(tool, Value::Null, true, None);
        Ok(())
    }

    /// 全量审计日志(按时间先后)。
    pub fn audit_log(&self) -> Vec<AuditRecord> {
        self.audit.lock().unwrap().iter().cloned().collect()
    }

    /// 清空审计日志。
    pub fn clear_audit(&self) {
        self.audit.lock().unwrap().clear();
    }

    fn record(&self, tool: &str, arguments: Value, allowed: bool, reason: Option<String>) {
        let rec = AuditRecord {
            server: self.server.clone(),
            tool: tool.to_string(),
            arguments,
            allowed,
            reason,
            at: SystemTime::now(),
        };
        let mut audit = self.audit.lock().unwrap();
        if audit.len() >= self.max_audit {
            audit.pop_front();
        }
        audit.push_back(rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 参数级最小权限:文件 Server 只允许 `file:///tmp/` 前缀。
    #[test]
    fn test_prefix_rule_blocks_and_allows() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        });
        sandbox
            .check_call("read_file", &json!({ "path": "file:///tmp/a.txt" }))
            .expect("tmp 前缀应放行");
        let err = sandbox
            .check_call("read_file", &json!({ "path": "file:///etc/passwd" }))
            .unwrap_err();
        assert!(err.to_string().contains("未以最小权限前缀"), "{}", err);
        // 缺字段 fail-closed。
        let err = sandbox.check_call("read_file", &json!({})).unwrap_err();
        assert!(err.to_string().contains("缺少字符串参数"), "{}", err);
    }

    /// 枚举白名单:只允许声明的值。
    #[test]
    fn test_enum_rule() {
        let sandbox = ServerSandbox::new("fmt").with_param_rule(ParamRule::Enum {
            field: "format".to_string(),
            allowed: vec!["json".to_string(), "yaml".to_string()],
        });
        sandbox
            .check_call("parse", &json!({ "format": "json" }))
            .expect("json 应在允许集合内");
        let err = sandbox
            .check_call("parse", &json!({ "format": "xml" }))
            .unwrap_err();
        assert!(err.to_string().contains("不在允许集合内"), "{}", err);
    }

    /// 拒绝路径穿越子串。
    #[test]
    fn test_reject_contains_rule() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::RejectContains {
            field: "path".to_string(),
            forbidden: vec!["..".to_string()],
        });
        sandbox
            .check_call("read_file", &json!({ "path": "/tmp/a.txt" }))
            .expect("正常路径应放行");
        let err = sandbox
            .check_call("read_file", &json!({ "path": "/tmp/../etc/passwd" }))
            .unwrap_err();
        assert!(err.to_string().contains("禁止的子串"), "{}", err);
    }

    /// 非对象参数 fail-closed。
    #[test]
    fn test_non_object_arguments_blocked() {
        let sandbox = ServerSandbox::new("s").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "/tmp/".to_string(),
        });
        let err = sandbox.check_call("t", &json!("hello")).unwrap_err();
        assert!(err.to_string().contains("必须是 JSON 对象"), "{}", err);
    }

    /// 出站白名单:精确匹配 + 子域放行 + 空策略全禁。
    #[test]
    fn test_egress_whitelist() {
        let policy = EgressPolicy::new().allow("example.com");
        assert!(policy.allows("example.com"));
        assert!(policy.allows("api.example.com"), "子域应放行");
        assert!(!policy.allows("example.org"));
        assert!(!policy.allows("evil-example.com"), "不得误放形似子域");
        assert!(EgressPolicy::new().is_empty());
        assert!(
            !EgressPolicy::new().allows("anything.example"),
            "空白名单全禁"
        );
    }

    /// 出站校验记审计:放行 + 拦截各一条。
    #[test]
    fn test_egress_check_records_audit() {
        let sandbox = ServerSandbox::new("fetch").allow_host("example.com");
        sandbox
            .check_egress("http_get", "example.com")
            .expect("白名单内应放行");
        let err = sandbox.check_egress("http_get", "evil.com").unwrap_err();
        assert!(err.to_string().contains("白名单"), "{}", err);

        let log = sandbox.audit_log();
        assert_eq!(log.len(), 2, "放行 + 拦截各记一条");
        assert!(log[0].allowed, "第一条放行");
        assert!(!log[1].allowed, "第二条拦截");
        assert!(log[1].reason.as_deref().unwrap().contains("白名单"));
        assert_eq!(log[1].server, "fetch");
        assert_eq!(log[1].tool, "http_get");
    }

    /// 全量审计:放行与拦截均记录,拦截带原因。
    #[test]
    fn test_audit_log_records_all_calls() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        });
        sandbox
            .check_call("read_file", &json!({ "path": "file:///tmp/a.txt" }))
            .expect("应放行");
        let _ = sandbox.check_call("read_file", &json!({ "path": "file:///etc/passwd" }));
        let log = sandbox.audit_log();
        assert_eq!(log.len(), 2, "放行与拦截均全量记录");
        assert!(log[0].allowed);
        assert!(log[0].arguments.get("path").is_some(), "审计保留全量参数");
        assert!(!log[1].allowed);
        assert!(log[1].reason.is_some());
    }

    /// 审计日志环形上限:只保留最新 max_audit 条。
    #[test]
    fn test_audit_cap_keeps_newest() {
        let sandbox = ServerSandbox::new("fs").with_max_audit(2);
        for i in 0..3 {
            sandbox
                .check_call("t", &json!({ "n": i }))
                .expect("无规则恒放行");
        }
        let log = sandbox.audit_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].arguments["n"], 1);
        assert_eq!(log[1].arguments["n"], 2);
    }

    /// 克隆沙箱共享同一份审计日志。
    #[test]
    fn test_clone_shares_audit() {
        let sandbox = ServerSandbox::new("fs");
        let clone = sandbox.clone();
        sandbox.check_call("t", &json!({})).expect("放行");
        assert_eq!(clone.audit_log().len(), 1, "克隆体共享审计日志");
    }
}
