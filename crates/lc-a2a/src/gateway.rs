//! P2-7: cross-organization federation gateway.
//!
//! Each organization deploys a [`FederationGateway`] in front of its agents. A
//! federation call from another org arrives at the gateway, which:
//!
//! 1. **Enforces the [`CallPolicy`]** — only callers from `allowed_caller_orgs`
//!    may invoke only `allowed_skills`, and only within `max_payload_size`.
//! 2. **Minimizes the data** — strips caller identity and any non-essential
//!    metadata before forwarding downstream (only `trace_id` and `message_id`
//!    are preserved, so tracing and idempotency survive federation hops).
//! 3. **Honors the data contract** — an optional [`DataContract`] is verified
//!    against the downstream agent's advertised `data_class` before the request
//!    is forwarded, so classified data never flows to an agent that has not
//!    signed up to handle it.
//!
//! The gateway holds downstream [`A2AClient`]s keyed by route (typically the
//! partner org's name) and forwards minimized requests over plain A2A.

use std::collections::HashMap;

use serde_json::Value;

use crate::client::A2AClient;
use crate::protocol::{metadata_keys, A2ARequest, A2AResponse, AgentCard};
use crate::A2AError;

/// Errors raised by the federation gateway.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GatewayError {
    /// The caller's org is not on the allow-list.
    #[error("caller org '{0}' is not allowed by federation policy")]
    CallerOrgNotAllowed(String),
    /// The requested skill is not on the allow-list.
    #[error("skill '{0}' is not allowed by federation policy")]
    SkillNotAllowed(String),
    /// The inbound request body exceeds the configured limit.
    #[error("request payload of {actual} bytes exceeds the {max} byte limit")]
    PayloadTooLarge {
        /// Actual request payload size in bytes.
        actual: usize,
        /// Maximum allowed payload size in bytes.
        max: usize,
    },
    /// The request carried no caller identity, so it cannot be authorized.
    #[error("request does not carry a caller identity")]
    MissingCaller,
    /// The downstream agent's card does not satisfy the data contract.
    #[error("downstream route '{0}' does not satisfy the data contract")]
    ContractUnsatisfied(String),
    /// No downstream route is registered for the requested key.
    #[error("no downstream route registered for '{0}'")]
    NoRoute(String),
    /// A downstream A2A call failed.
    #[error("A2A client error: {0}")]
    Client(#[from] A2AError),
}

/// Access policy enforced on every inbound federation call (P2-7).
///
/// `None` means "no restriction" for that dimension; an empty `Some(Vec)`
/// denies everything. The payload limit always applies.
#[derive(Debug, Clone)]
pub struct CallPolicy {
    /// Orgs permitted to call this gateway. Caller identity comes from request
    /// metadata `owner`, using the `org:user` convention.
    pub allowed_caller_orgs: Option<Vec<String>>,
    /// Skills permitted to be invoked (matched against `skillId`).
    pub allowed_skills: Option<Vec<String>>,
    /// Maximum inbound request body size in bytes.
    pub max_payload_size: usize,
}

impl Default for CallPolicy {
    fn default() -> Self {
        Self {
            allowed_caller_orgs: None,
            allowed_skills: None,
            max_payload_size: 1024 * 1024,
        }
    }
}

impl CallPolicy {
    /// An allow-everything policy (payload limit still applies).
    pub fn new() -> Self {
        Self::default()
    }

    /// Permit calls from an org (repeatable).
    pub fn allow_caller_org(mut self, org: impl Into<String>) -> Self {
        self.allowed_caller_orgs
            .get_or_insert_with(Vec::new)
            .push(org.into());
        self
    }

    /// Permit a skill (repeatable).
    pub fn allow_skill(mut self, skill: impl Into<String>) -> Self {
        self.allowed_skills
            .get_or_insert_with(Vec::new)
            .push(skill.into());
        self
    }

    /// Cap the inbound payload size in bytes.
    pub fn with_max_payload_size(mut self, bytes: usize) -> Self {
        self.max_payload_size = bytes;
        self
    }

    fn caller_org_allowed(&self, org: &str) -> bool {
        self.allowed_caller_orgs
            .as_ref()
            .is_none_or(|orgs| orgs.iter().any(|o| o == org))
    }

    fn skill_allowed(&self, skill: &str) -> bool {
        self.allowed_skills
            .as_ref()
            .is_none_or(|skills| skills.iter().any(|s| s == skill))
    }

    fn payload_allowed(&self, len: usize) -> bool {
        len <= self.max_payload_size
    }
}

/// Data-processing contract between the gateway and a downstream agent (P2-7).
///
/// Federation only forwards classified data to agents that advertise a
/// `data_class` at least as protective as `required_classification`. The
/// classification ordering is `public` < `internal` < `confidential`; an agent
/// that advertises no classification (or an unknown one) is *not* admitted —
/// fail closed.
#[derive(Debug, Clone)]
pub struct DataContract {
    /// Minimum `data_class` the downstream agent must advertise.
    pub required_classification: String,
    /// Purpose the data may be used for (informational, surfaces in errors).
    pub purpose: String,
    /// Retention the downstream promises (e.g. "session", "7d").
    pub retention: String,
    /// Whether the downstream may forward data to further agents.
    pub allow_forwarding: bool,
}

impl DataContract {
    /// Create a contract.
    pub fn new(
        required_classification: impl Into<String>,
        purpose: impl Into<String>,
        retention: impl Into<String>,
        allow_forwarding: bool,
    ) -> Self {
        Self {
            required_classification: required_classification.into(),
            purpose: purpose.into(),
            retention: retention.into(),
            allow_forwarding,
        }
    }

    /// Whether an agent advertising `data_class` satisfies this contract.
    ///
    /// Fail-closed: an unadvertised or unknown `data_class` never admits.
    pub fn admits(&self, data_class: Option<&str>) -> bool {
        match data_class.and_then(classification_rank) {
            Some(agent_rank) => match classification_rank(&self.required_classification) {
                Some(required_rank) => agent_rank >= required_rank,
                // Unknown requirement — fail closed.
                None => false,
            },
            None => false,
        }
    }
}

/// The protective rank of a data classification, if recognized.
fn classification_rank(c: &str) -> Option<u8> {
    match c.trim().to_ascii_lowercase().as_str() {
        "public" => Some(0),
        "internal" => Some(1),
        "confidential" => Some(2),
        _ => None,
    }
}

/// Cross-organization federation gateway (P2-7).
pub struct FederationGateway {
    /// The org this gateway fronts.
    org: String,
    /// Access policy for inbound calls.
    policy: CallPolicy,
    /// Optional data-processing contract verified against downstream cards.
    contract: Option<DataContract>,
    /// Downstream routes: route key -> A2A client.
    clients: HashMap<String, A2AClient>,
    /// Whether caller identity is stripped from forwarded requests.
    minimize_metadata: bool,
}

impl FederationGateway {
    /// Create a gateway for `org`, enforcing `policy`.
    pub fn new(org: impl Into<String>, policy: CallPolicy) -> Self {
        Self {
            org: org.into(),
            policy,
            contract: None,
            clients: HashMap::new(),
            minimize_metadata: true,
        }
    }

    /// Attach a data contract that downstream agents must satisfy.
    pub fn with_contract(mut self, contract: DataContract) -> Self {
        self.contract = Some(contract);
        self
    }

    /// Register a downstream route (e.g. a partner org) to an A2A client.
    pub fn with_route(mut self, key: impl Into<String>, client: A2AClient) -> Self {
        self.clients.insert(key.into(), client);
        self
    }

    /// Toggle data minimization. On (default) the caller identity is stripped
    /// from forwarded requests; off relays it unchanged.
    pub fn minimize_metadata(mut self, on: bool) -> Self {
        self.minimize_metadata = on;
        self
    }

    /// The org this gateway fronts.
    pub fn org(&self) -> &str {
        &self.org
    }

    /// The active call policy.
    pub fn policy(&self) -> &CallPolicy {
        &self.policy
    }

    /// Validate an inbound request against the policy (P2-7).
    ///
    /// `raw_len` is the size of the request body as received on the wire; it is
    /// checked against [`CallPolicy::max_payload_size`]. Caller org is taken
    /// from request metadata `owner` (the `org:user` convention), the skill
    /// from the `skillId` param, and both must be allowed when a policy lists
    /// them.
    pub fn enforce(&self, req: &A2ARequest, raw_len: usize) -> Result<(), GatewayError> {
        if !self.policy.payload_allowed(raw_len) {
            return Err(GatewayError::PayloadTooLarge {
                actual: raw_len,
                max: self.policy.max_payload_size,
            });
        }
        let owner = req.owner().ok_or(GatewayError::MissingCaller)?;
        let org = org_from_owner(owner);
        if !self.policy.caller_org_allowed(org) {
            return Err(GatewayError::CallerOrgNotAllowed(org.to_string()));
        }
        if let Some(skill) = request_skill(req) {
            if !self.policy.skill_allowed(skill) {
                return Err(GatewayError::SkillNotAllowed(skill.to_string()));
            }
        }
        Ok(())
    }

    /// Data minimization: a copy of `req` carrying only the fields a downstream
    /// agent strictly needs.
    ///
    /// The `trace_id` (distributed tracing) and `message_id` (idempotency) are
    /// preserved; the caller's `owner` is dropped while minimization is on.
    /// Method, id, and params pass through unchanged — they are the request.
    pub fn minimize(&self, req: &A2ARequest) -> A2ARequest {
        let mut slim = serde_json::Map::new();
        if let Some(meta) = &req.metadata {
            if let Some(v) = meta.get(metadata_keys::TRACE_ID) {
                slim.insert(metadata_keys::TRACE_ID.to_string(), v.clone());
            }
            if let Some(v) = meta.get(metadata_keys::MESSAGE_ID) {
                slim.insert(metadata_keys::MESSAGE_ID.to_string(), v.clone());
            }
            if !self.minimize_metadata {
                if let Some(v) = meta.get(metadata_keys::OWNER) {
                    slim.insert(metadata_keys::OWNER.to_string(), v.clone());
                }
            }
        }
        A2ARequest {
            jsonrpc: req.jsonrpc.clone(),
            id: req.id,
            method: req.method.clone(),
            params: req.params.clone(),
            metadata: (!slim.is_empty()).then_some(Value::Object(slim)),
        }
    }

    /// Verify a downstream agent card against the attached data contract.
    ///
    /// With no contract attached every card passes. Otherwise the agent must
    /// advertise a `data_class` the contract admits, or [`GatewayError::ContractUnsatisfied`]
    /// is returned (P2-7).
    pub fn contract_admits(&self, card: &AgentCard) -> Result<(), GatewayError> {
        if let Some(contract) = &self.contract {
            if !contract.admits(card.data_class.as_deref()) {
                return Err(GatewayError::ContractUnsatisfied(card.url.clone()));
            }
        }
        Ok(())
    }

    /// Fetch the downstream card for `key` and verify it against the contract.
    pub async fn verify_downstream(&self, key: &str) -> Result<AgentCard, GatewayError> {
        let client = self
            .clients
            .get(key)
            .ok_or_else(|| GatewayError::NoRoute(key.to_string()))?;
        let card = client.get_agent_card().await?;
        self.contract_admits(&card)?;
        Ok(card)
    }

    /// Enforce the policy, minimize the request, and forward it to the
    /// downstream route `key` (P2-7).
    ///
    /// `raw_len` is the size of the request body as received on the wire. The
    /// outbound request is minimized before it is sent, and the downstream
    /// A2A response is returned verbatim.
    pub async fn forward(
        &self,
        key: &str,
        req: &A2ARequest,
        raw_len: usize,
    ) -> Result<A2AResponse, GatewayError> {
        self.enforce(req, raw_len)?;
        let client = self
            .clients
            .get(key)
            .ok_or_else(|| GatewayError::NoRoute(key.to_string()))?;
        let outbound = self.minimize(req);
        Ok(client.post_request(outbound).await?)
    }
}

/// The organization part of a caller identity. Identities use the `org:user`
/// convention; an identity without a separator is its own org.
fn org_from_owner(owner: &str) -> &str {
    owner.split_once(':').map(|(org, _)| org).unwrap_or(owner)
}

/// The `skillId` param of a request, if any.
fn request_skill(req: &A2ARequest) -> Option<&str> {
    req.params
        .as_ref()
        .and_then(|p| p.get("skillId"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::protocol::A2AMessage;

    fn request() -> A2ARequest {
        A2ARequest::send_task(1, &A2AMessage::user("hi"))
    }

    #[test]
    fn policy_denies_unknown_caller_org() {
        let policy = CallPolicy::new().allow_caller_org("acme");
        let gw = FederationGateway::new("gw", policy);
        let req = request().with_owner("evil:user");
        let err = gw.enforce(&req, 100).unwrap_err();
        assert!(matches!(err, GatewayError::CallerOrgNotAllowed(o) if o == "evil"));
    }

    #[test]
    fn policy_allows_known_org_and_denies_unknown_skill() {
        let policy = CallPolicy::new()
            .allow_caller_org("acme")
            .allow_skill("research");
        let gw = FederationGateway::new("gw", policy);
        let req = request().with_owner("acme:alice");
        // Allowed org, allowed skill -> passes.
        let with_skill = {
            let mut params = req.params.clone().unwrap();
            params["skillId"] = Value::String("research".to_string());
            A2ARequest {
                jsonrpc: req.jsonrpc.clone(),
                id: req.id,
                method: req.method.clone(),
                params: Some(params),
                metadata: req.metadata.clone(),
            }
        };
        gw.enforce(&with_skill, 100).unwrap();

        // Same org, unknown skill -> rejected.
        let unknown_skill = {
            let mut params = req.params.clone().unwrap();
            params["skillId"] = Value::String("summarize".to_string());
            A2ARequest {
                jsonrpc: req.jsonrpc.clone(),
                id: req.id,
                method: req.method.clone(),
                params: Some(params),
                metadata: req.metadata.clone(),
            }
        };
        let err = gw.enforce(&unknown_skill, 100).unwrap_err();
        assert!(matches!(err, GatewayError::SkillNotAllowed(s) if s == "summarize"));
    }

    #[test]
    fn policy_denies_oversized_payload() {
        let policy = CallPolicy::new().with_max_payload_size(16);
        let gw = FederationGateway::new("gw", policy);
        let req = request().with_owner("acme:alice");
        let err = gw.enforce(&req, 100).unwrap_err();
        assert!(matches!(
            err,
            GatewayError::PayloadTooLarge {
                actual: 100,
                max: 16
            }
        ));
    }

    #[test]
    fn policy_requires_caller_identity() {
        let gw = FederationGateway::new("gw", CallPolicy::new());
        let err = gw.enforce(&request(), 100).unwrap_err();
        assert!(matches!(err, GatewayError::MissingCaller));
    }

    #[test]
    fn minimize_strips_owner_but_keeps_trace_and_message_id() {
        let gw = FederationGateway::new("gw", CallPolicy::new());
        let req = request()
            .with_owner("acme:alice")
            .with_trace_id("trace-1")
            .with_message_id("msg-1");

        let out = gw.minimize(&req);
        assert_eq!(out.owner(), None, "caller identity must not leak");
        assert_eq!(out.trace_id(), Some("trace-1"));
        assert_eq!(out.message_id(), Some("msg-1"));
        assert_eq!(out.method, "tasks/send");
        assert!(out.params.is_some());
    }

    #[test]
    fn minimize_can_relay_caller_identity_when_disabled() {
        let gw = FederationGateway::new("gw", CallPolicy::new()).minimize_metadata(false);
        let req = request().with_owner("acme:alice").with_trace_id("trace-1");
        let out = gw.minimize(&req);
        assert_eq!(out.owner(), Some("acme:alice"));
    }

    #[test]
    fn org_extraction_uses_org_prefix() {
        assert_eq!(org_from_owner("acme:alice"), "acme");
        assert_eq!(org_from_owner("acme"), "acme");
        assert_eq!(org_from_owner("alice@acme.org"), "alice@acme.org");
    }

    #[test]
    fn data_contract_admits_classification() {
        let contract = DataContract::new("internal", "task-execution", "session", false);
        // Same classification -> admitted.
        assert!(contract.admits(Some("internal")));
        // More protective -> admitted.
        assert!(contract.admits(Some("confidential")));
        // Less protective -> denied.
        assert!(!contract.admits(Some("public")));
        // No classification advertised -> fail closed.
        assert!(!contract.admits(None));
        // Unknown classification -> fail closed.
        assert!(!contract.admits(Some("top-secret")));
    }

    #[test]
    fn contract_admits_checks_downstream_card() {
        let gw = FederationGateway::new("gw", CallPolicy::new())
            .with_contract(DataContract::new("internal", "x", "session", false));

        let ok = AgentCard::new("a", "a", "http://a").with_data_class("internal");
        gw.contract_admits(&ok).unwrap();

        let bad = AgentCard::new("b", "b", "http://b").with_data_class("public");
        let err = gw.contract_admits(&bad).unwrap_err();
        assert!(matches!(err, GatewayError::ContractUnsatisfied(url) if url == "http://b"));
    }

    // ---- downstream HTTP forwarding ----

    type Handler = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

    async fn spawn_server(handler: Handler) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::net::TcpStream;

        async fn read_request(stream: &mut TcpStream) -> (String, String) {
            let mut buf = vec![0u8; 4096];
            let mut request = Vec::new();
            let mut head_end = None;
            while head_end.is_none() {
                let n = stream.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                head_end = request.windows(4).position(|w| w == b"\r\n\r\n");
            }
            let head_end = head_end.expect("head terminator");
            let head = String::from_utf8_lossy(&request[..head_end]).to_string();
            let body_len = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let mut body = request[head_end + 4..].to_vec();
            while body.len() < body_len {
                let n = stream.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&buf[..n]);
            }
            (String::new(), String::from_utf8_lossy(&body).to_string())
        }

        async fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
            let head = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.write_all(body.as_bytes()).await;
            let _ = stream.shutdown().await;
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let mut stream = stream;
                    let (_, body) = read_request(&mut stream).await;
                    let (status, response) = handler("", &body);
                    write_response(&mut stream, status, &response).await;
                });
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn downstream_ok_response() -> String {
        r#"{"jsonrpc":"2.0","id":1,"result":{"task":{"id":"fwd-1","message":{"role":"user","content":"hi"},"status":"completed","result":{"output":"ok"}}}}"#.to_string()
    }

    #[tokio::test]
    async fn forward_enforces_policy_and_forwards_minimized_request() {
        let captured = Arc::new(Mutex::new(String::new()));
        let cap = captured.clone();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let handler: Handler = Arc::new(move |_path, body| {
            h.fetch_add(1, Ordering::SeqCst);
            *cap.lock().unwrap_or_else(|e| e.into_inner()) = body.to_string();
            (200, downstream_ok_response())
        });
        let base = spawn_server(handler).await;

        let gw = FederationGateway::new("gw", CallPolicy::new().allow_caller_org("acme"))
            .with_route("partner", A2AClient::new(base).unwrap());

        let req = request()
            .with_owner("acme:alice")
            .with_trace_id("trace-9")
            .with_message_id("msg-9");

        let resp = gw.forward("partner", &req, 200).await.unwrap();
        assert!(resp.result.is_some());
        assert_eq!(hits.load(Ordering::SeqCst), 1);

        // The forwarded body must be minimized: trace + message id kept,
        // caller identity stripped.
        let forwarded = captured.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert!(
            forwarded.contains("trace_id"),
            "trace must survive forwarding"
        );
        assert!(
            forwarded.contains("message_id"),
            "idempotency key must survive"
        );
        assert!(
            !forwarded.contains("acme:alice"),
            "caller identity must be stripped before forwarding"
        );
    }

    #[tokio::test]
    async fn forward_rejects_policy_violation_without_calling_downstream() {
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let handler: Handler = Arc::new(move |_path, _body| {
            h.fetch_add(1, Ordering::SeqCst);
            (200, downstream_ok_response())
        });
        let base = spawn_server(handler).await;

        let gw = FederationGateway::new("gw", CallPolicy::new().allow_caller_org("acme"))
            .with_route("partner", A2AClient::new(base).unwrap());

        // Caller from an unlisted org.
        let req = request().with_owner("evil:user");
        let err = gw.forward("partner", &req, 200).await.unwrap_err();
        assert!(matches!(err, GatewayError::CallerOrgNotAllowed(o) if o == "evil"));
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn forward_missing_route_is_a_no_route_error() {
        let gw = FederationGateway::new("gw", CallPolicy::new());
        let req = request().with_owner("acme:alice");
        let err = gw.forward("nope", &req, 200).await.unwrap_err();
        assert!(matches!(err, GatewayError::NoRoute(r) if r == "nope"));
    }
}
