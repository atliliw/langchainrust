//! per-Server security isolation (P2-6).
//!
//! With 100+ servers of varied provenance and differing permission boundaries, permissions must be narrowed
//! independently per server:
//!
//! - **Process-layer isolated containers**: each server is its own child process / independent connection, sharing
//!   no memory (P2-1 lazy connections are naturally isolated);
//! - **Credential-layer independent credentials**: each `ServerSpec` carries its own config/credentials, never
//!   shared across servers;
//! - **Parameter-level least privilege**: [`ParamRule`] constrains tool-call arguments — a file server only allows
//!   the `file:///tmp` prefix, formats only allow enum values, path-traversal substrings are rejected, violations are blocked;
//! - **Network-layer egress allowlist**: [`EgressPolicy`] declares the hosts a server may reach; an empty allowlist
//!   blocks everything (fail-closed);
//! - **Audit-layer full recording**: [`ServerSandbox`] records every allow/block call for later auditing.
//!
//! [`MCPToolAdapter::with_sandbox`](crate::tool_adapter::MCPToolAdapter::with_sandbox) attaches a `ServerSandbox`
//! to the tool adapter; `run()` passes `check_call` before sending the request, returning an error and recording
//! the audit when blocked.

use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Parameter-level least-privilege rule (P2-6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRule {
    /// The string parameter must start with a given prefix (a least-privilege prefix allowlist,
    /// e.g. a file server only allows `file:///tmp/`).
    Prefix {
        /// Target parameter field name
        field: String,
        /// Required prefix
        prefix: String,
    },
    /// The string parameter must fall within the allowed set (enum allowlist).
    Enum {
        /// Target parameter field name
        field: String,
        /// Allowed value set
        allowed: Vec<String>,
    },
    /// The string parameter must not contain a given substring (e.g. path traversal `..`, dangerous commands).
    RejectContains {
        /// Target parameter field name
        field: String,
        /// List of forbidden substrings
        forbidden: Vec<String>,
    },
}

impl ParamRule {
    /// Validates one tool call's arguments; returns the violation reason.
    fn check(&self, arguments: &Value) -> Result<(), ParamRuleError> {
        let obj = match arguments {
            Value::Object(m) => m,
            // Non-object arguments (no fields to validate): fail-closed, block to preserve least privilege.
            _ => {
                return Err(ParamRuleError::Violation(
                    "arguments must be a JSON object to perform least-privilege validation"
                        .to_string(),
                ))
            }
        };
        match self {
            ParamRule::Prefix { field, prefix } => {
                let v = obj.get(field).and_then(Value::as_str).ok_or_else(|| {
                    ParamRuleError::Violation(format!("missing string parameter '{field}'"))
                })?;
                if v.starts_with(prefix) {
                    Ok(())
                } else {
                    Err(ParamRuleError::Violation(format!(
                        "value '{v}' of parameter '{field}' does not start with the required least-privilege prefix '{prefix}'"
                    )))
                }
            }
            ParamRule::Enum { field, allowed } => {
                let v = obj.get(field).and_then(Value::as_str).ok_or_else(|| {
                    ParamRuleError::Violation(format!("missing string parameter '{field}'"))
                })?;
                if allowed.iter().any(|a| a == v) {
                    Ok(())
                } else {
                    Err(ParamRuleError::Violation(format!(
                        "value '{v}' of parameter '{field}' is not in the allowed set"
                    )))
                }
            }
            ParamRule::RejectContains { field, forbidden } => {
                let v = obj.get(field).and_then(Value::as_str).ok_or_else(|| {
                    ParamRuleError::Violation(format!("missing string parameter '{field}'"))
                })?;
                if let Some(bad) = forbidden.iter().find(|b| v.contains(b.as_str())) {
                    Err(ParamRuleError::Violation(format!(
                        "parameter '{field}' contains a forbidden substring '{bad}'"
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Parameter-level least-privilege validation error (P2-6).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamRuleError {
    /// The specific reason the parameter rule was violated.
    Violation(String),
}

impl std::fmt::Display for ParamRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamRuleError::Violation(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for ParamRuleError {}

/// Network-layer egress allowlist (P2-6): the hosts this Server may access.
///
/// An empty policy = all egress blocked (fail-closed). `allows` supports subdomains:
/// allowing `example.com` permits `api.example.com`, but not `evil-example.com`.
#[derive(Debug, Clone, Default)]
pub struct EgressPolicy {
    allowed: Vec<String>,
}

impl EgressPolicy {
    /// Empty allowlist (blocks all egress).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an allowed hostname.
    pub fn allow(mut self, host: impl Into<String>) -> Self {
        self.allowed.push(host.into());
        self
    }

    /// Whether the allowlist is empty (empty means all egress is blocked).
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    /// Whether the host is allowed (case-insensitive, supports subdomain wildcards).
    pub fn allows(&self, host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        self.allowed.iter().any(|a| {
            let a = a.to_ascii_lowercase();
            host == a || host.ends_with(&format!(".{a}"))
        })
    }
}

/// Audit record (P2-6): one allowed/blocked call.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    /// The owning Server.
    pub server: String,
    /// The tool that was called.
    pub tool: String,
    /// Tool call arguments (in full).
    pub arguments: Value,
    /// Whether it was allowed.
    pub allowed: bool,
    /// Block reason (present when `allowed` is false).
    pub reason: Option<String>,
    /// Record time.
    pub at: SystemTime,
}

/// Sandbox block error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    /// Block reason description
    pub reason: String,
}

impl SandboxError {
    /// Builds a sandbox block error.
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

/// per-Server security sandbox (P2-6): parameter least privilege + egress allowlist + full audit.
///
/// Fields are all `Arc`/`Mutex`, cheaply `Clone`able for distribution to multiple tool adapters of the same
/// server, sharing the same rules and audit log.
#[derive(Debug, Clone)]
pub struct ServerSandbox {
    server: String,
    /// Parameter-level least-privilege rules.
    param_rules: Arc<Vec<ParamRule>>,
    /// Network-layer egress allowlist.
    egress: Arc<EgressPolicy>,
    /// Full audit log (ring buffer, cap `max_audit`).
    audit: Arc<Mutex<VecDeque<AuditRecord>>>,
    /// Audit log cap (default 1000).
    max_audit: usize,
}

impl ServerSandbox {
    /// Creates a per-Server security sandbox (no parameter rules by default: allow; all egress blocked).
    pub fn new(server: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            param_rules: Arc::new(Vec::new()),
            egress: Arc::new(EgressPolicy::new()),
            audit: Arc::new(Mutex::new(VecDeque::new())),
            max_audit: 1000,
        }
    }

    /// Adds a parameter-level least-privilege rule.
    pub fn with_param_rule(mut self, rule: ParamRule) -> Self {
        Arc::make_mut(&mut self.param_rules).push(rule);
        self
    }

    /// Adds an egress-allowlist host.
    pub fn allow_host(mut self, host: impl Into<String>) -> Self {
        let mut policy = Arc::make_mut(&mut self.egress).clone();
        policy.allowed.push(host.into());
        self.egress = Arc::new(policy);
        self
    }

    /// Replaces the whole egress allowlist.
    pub fn with_egress(mut self, policy: EgressPolicy) -> Self {
        self.egress = Arc::new(policy);
        self
    }

    /// Sets the audit-log cap (minimum 1).
    pub fn with_max_audit(mut self, max: usize) -> Self {
        self.max_audit = max.max(1);
        self
    }

    /// Validates one tool call (parameter-level least privilege), recording Allowed on pass and Blocked on intercept.
    pub fn check_call(&self, tool: &str, arguments: &Value) -> Result<(), SandboxError> {
        for rule in self.param_rules.iter() {
            if let Err(e) = rule.check(arguments) {
                let reason = e.to_string();
                self.record(tool, arguments.clone(), false, Some(reason.clone()));
                return Err(SandboxError::new(reason));
            }
        }
        self.record(tool, arguments.clone(), true, None);
        Ok(())
    }

    /// Validates whether the egress target is in the allowlist (network layer).
    pub fn check_egress(&self, tool: &str, host: &str) -> Result<(), SandboxError> {
        if !self.egress.allows(host) {
            let reason = format!("egress target '{host}' is not in the allowlist");
            self.record(tool, Value::Null, false, Some(reason.clone()));
            return Err(SandboxError::new(reason));
        }
        self.record(tool, Value::Null, true, None);
        Ok(())
    }

    /// Full audit log (in chronological order).
    pub fn audit_log(&self) -> Vec<AuditRecord> {
        self.audit
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Clears the audit log.
    pub fn clear_audit(&self) {
        self.audit.lock().unwrap_or_else(|e| e.into_inner()).clear();
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
        let mut audit = self.audit.lock().unwrap_or_else(|e| e.into_inner());
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

    /// Parameter-level least privilege: a file server only allows the `file:///tmp/` prefix.
    #[test]
    fn test_prefix_rule_blocks_and_allows() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        });
        sandbox
            .check_call("read_file", &json!({ "path": "file:///tmp/a.txt" }))
            .expect("tmp prefix should be allowed");
        let err = sandbox
            .check_call("read_file", &json!({ "path": "file:///etc/passwd" }))
            .unwrap_err();
        assert!(
            err.to_string().contains("least-privilege prefix"),
            "{}",
            err
        );
        // Missing field fails closed.
        let err = sandbox.check_call("read_file", &json!({})).unwrap_err();
        assert!(
            err.to_string().contains("missing string parameter"),
            "{}",
            err
        );
    }

    /// Enum allowlist: only declared values are allowed.
    #[test]
    fn test_enum_rule() {
        let sandbox = ServerSandbox::new("fmt").with_param_rule(ParamRule::Enum {
            field: "format".to_string(),
            allowed: vec!["json".to_string(), "yaml".to_string()],
        });
        sandbox
            .check_call("parse", &json!({ "format": "json" }))
            .expect("json should be in the allowed set");
        let err = sandbox
            .check_call("parse", &json!({ "format": "xml" }))
            .unwrap_err();
        assert!(
            err.to_string().contains("not in the allowed set"),
            "{}",
            err
        );
    }

    /// Rejects path-traversal substrings.
    #[test]
    fn test_reject_contains_rule() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::RejectContains {
            field: "path".to_string(),
            forbidden: vec!["..".to_string()],
        });
        sandbox
            .check_call("read_file", &json!({ "path": "/tmp/a.txt" }))
            .expect("normal path should be allowed");
        let err = sandbox
            .check_call("read_file", &json!({ "path": "/tmp/../etc/passwd" }))
            .unwrap_err();
        assert!(err.to_string().contains("forbidden substring"), "{}", err);
    }

    /// Non-object arguments fail closed.
    #[test]
    fn test_non_object_arguments_blocked() {
        let sandbox = ServerSandbox::new("s").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "/tmp/".to_string(),
        });
        let err = sandbox.check_call("t", &json!("hello")).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"), "{}", err);
    }

    /// Egress allowlist: exact match + subdomain allowed + empty policy blocks all.
    #[test]
    fn test_egress_whitelist() {
        let policy = EgressPolicy::new().allow("example.com");
        assert!(policy.allows("example.com"));
        assert!(
            policy.allows("api.example.com"),
            "subdomain should be allowed"
        );
        assert!(!policy.allows("example.org"));
        assert!(
            !policy.allows("evil-example.com"),
            "must not allow a similar-looking subdomain"
        );
        assert!(EgressPolicy::new().is_empty());
        assert!(
            !EgressPolicy::new().allows("anything.example"),
            "empty allowlist blocks all"
        );
    }

    /// Egress check records audit: one record for allow, one for block.
    #[test]
    fn test_egress_check_records_audit() {
        let sandbox = ServerSandbox::new("fetch").allow_host("example.com");
        sandbox
            .check_egress("http_get", "example.com")
            .expect("host in allowlist should be allowed");
        let err = sandbox.check_egress("http_get", "evil.com").unwrap_err();
        assert!(err.to_string().contains("allowlist"), "{}", err);

        let log = sandbox.audit_log();
        assert_eq!(log.len(), 2, "one record for allow and one for block");
        assert!(log[0].allowed, "first record is allowed");
        assert!(!log[1].allowed, "second record is blocked");
        assert!(log[1].reason.as_deref().unwrap().contains("allowlist"));
        assert_eq!(log[1].server, "fetch");
        assert_eq!(log[1].tool, "http_get");
    }

    /// Full audit: both allow and block are recorded, with the reason on block.
    #[test]
    fn test_audit_log_records_all_calls() {
        let sandbox = ServerSandbox::new("fs").with_param_rule(ParamRule::Prefix {
            field: "path".to_string(),
            prefix: "file:///tmp/".to_string(),
        });
        sandbox
            .check_call("read_file", &json!({ "path": "file:///tmp/a.txt" }))
            .expect("should be allowed");
        let _ = sandbox.check_call("read_file", &json!({ "path": "file:///etc/passwd" }));
        let log = sandbox.audit_log();
        assert_eq!(
            log.len(),
            2,
            "both allow and block calls are fully recorded"
        );
        assert!(log[0].allowed);
        assert!(
            log[0].arguments.get("path").is_some(),
            "audit keeps full arguments"
        );
        assert!(!log[1].allowed);
        assert!(log[1].reason.is_some());
    }

    /// Audit-log ring cap: only the newest max_audit entries are kept.
    #[test]
    fn test_audit_cap_keeps_newest() {
        let sandbox = ServerSandbox::new("fs").with_max_audit(2);
        for i in 0..3 {
            sandbox
                .check_call("t", &json!({ "n": i }))
                .expect("no rules should always allow");
        }
        let log = sandbox.audit_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].arguments["n"], 1);
        assert_eq!(log[1].arguments["n"], 2);
    }

    /// A cloned sandbox shares the same audit log.
    #[test]
    fn test_clone_shares_audit() {
        let sandbox = ServerSandbox::new("fs");
        let clone = sandbox.clone();
        sandbox
            .check_call("t", &json!({}))
            .expect("should be allowed");
        assert_eq!(clone.audit_log().len(), 1, "clone shares the audit log");
    }
}
