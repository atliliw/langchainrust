//! P2-5: defense against malicious agents.
//!
//! A4A agents can be impersonated, tampered with, or hostile. This module
//! provides the building blocks to trust only what you should:
//!
//! - **Trust directory** — [`TrustRegistry`] maps agent identities (their card
//!   `url`) to the verification key the directory attests, and keeps a
//!   revocation list (CRL). Cards are only trusted when they come from a known
//!   agent and their signature verifies against the registered key.
//! - **Trust chain propagation** — [`TrustRegistry::verify_chain`] walks a
//!   delegation path root → … → leaf, verifying every hop was signed by its
//!   parent, enforcing a maximum delegation depth and decaying the trust score
//!   with each hop ([`TrustConfig`]).
//! - **Least privilege** — [`SandboxConfig`] restricts an agent's file/network
//!   access and payload size, checked via [`SandboxConfig::check`].
//!
//! Signature primitives are the HMAC-SHA256 card signatures shared with the
//! client (`sign_agent_card` / `verify_card_signature`), so a registry and a
//! client can agree on the same key material.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::client::{sign_agent_card, verify_card_signature};
use crate::protocol::AgentCard;

/// Error raised when a security check fails.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SecurityError {
    /// The agent is not in the trust registry at all.
    #[error("agent `{0}` is not in the trust registry")]
    UntrustedAgent(String),
    /// The agent was issued a certificate but has since been revoked.
    #[error("agent `{0}` has been revoked")]
    RevokedAgent(String),
    /// The card's signature did not verify against the expected key.
    #[error("signature verification failed for `{0}`: {1}")]
    SignatureMismatch(String, String),
    /// A delegation path exceeds the configured depth limit.
    #[error("delegation depth {depth} for `{url}` exceeds the limit {limit}")]
    DeepDelegation {
        /// The agent URL being delegated to.
        url: String,
        /// The actual delegation depth.
        depth: usize,
        /// The configured maximum delegation depth.
        limit: usize,
    },
    /// The key material registered for an agent is unusable.
    #[error("invalid key for `{0}`: {1}")]
    InvalidKey(String, String),
    /// The sandbox denied a requested access.
    #[error("sandbox denied: {0}")]
    SandboxDenied(String),
    /// A payload exceeded the sandbox's size limit.
    #[error("payload of {size} bytes exceeds the {limit} byte limit")]
    PayloadTooLarge {
        /// Actual payload size in bytes.
        size: usize,
        /// Maximum allowed payload size in bytes.
        limit: usize,
    },
}

/// Role of an agent in a trust hierarchy. Drives the base trust score before
/// hop decay is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRole {
    /// Fully trusted issuer (self-attested, top of a delegation chain).
    Root,
    /// Trusted intermediary that delegates to others.
    Intermediate,
    /// Endpoint agent that performs work but does not delegate.
    Leaf,
}

impl TrustRole {
    fn base_trust(&self) -> f64 {
        match self {
            TrustRole::Root => 1.0,
            TrustRole::Intermediate => 0.8,
            TrustRole::Leaf => 0.6,
        }
    }
}

/// An agent the registry trusts, with the key used to verify its card.
#[derive(Debug, Clone)]
pub struct TrustedAgent {
    /// Identity anchor — must match the card's `url`.
    pub url: String,
    /// Human-readable name.
    pub name: String,
    /// Role in the trust hierarchy.
    pub role: TrustRole,
    /// HMAC-SHA256 verification key (the same secret used to sign the card).
    pub verification_key: Vec<u8>,
}

impl TrustedAgent {
    /// Create a trusted agent entry.
    pub fn new(url: impl Into<String>, name: impl Into<String>, role: TrustRole) -> Self {
        Self {
            url: url.into(),
            name: name.into(),
            role,
            verification_key: Vec::new(),
        }
    }

    /// Set the verification key (builder style).
    pub fn with_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.verification_key = key.into();
        self
    }
}

/// Delegation-depth and trust-decay policy for chain verification.
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Maximum number of delegation hops (edges) allowed on a chain.
    pub max_delegation_depth: usize,
    /// Multiplicative trust factor applied per hop, in `(0.0, 1.0]`.
    pub trust_decay: f64,
}

impl Default for TrustConfig {
    /// Conservative default: depth 3, 0.9 decay per hop.
    fn default() -> Self {
        Self::new(3, 0.9)
    }
}

impl TrustConfig {
    /// Create a config with a depth limit and a per-hop decay factor.
    ///
    /// # Panics
    /// Panics if `trust_decay` is not in `(0.0, 1.0]`.
    pub fn new(max_delegation_depth: usize, trust_decay: f64) -> Self {
        assert!(
            trust_decay > 0.0 && trust_decay <= 1.0,
            "trust_decay must be in (0.0, 1.0]"
        );
        Self {
            max_delegation_depth,
            trust_decay,
        }
    }

    /// Trust score after `hops` delegation hops from a base score.
    pub fn effective_trust(&self, base: f64, hops: usize) -> f64 {
        base * self.trust_decay.powi(hops as i32)
    }
}

/// Result of a successful trust verification.
#[derive(Debug, Clone)]
pub struct TrustVerification {
    /// Identity of the verified agent.
    pub url: String,
    /// Its role in the hierarchy.
    pub role: TrustRole,
    /// Effective trust score after depth/decay adjustments.
    pub trust_score: f64,
}

/// Trust directory: known agents, a revocation list (CRL), and chain policy.
///
/// Immutable after construction (builder methods return a new registry), so it
/// can be shared freely behind an `Arc`.
#[derive(Debug, Default)]
pub struct TrustRegistry {
    agents: HashMap<String, TrustedAgent>,
    revoked: HashSet<String>,
    config: TrustConfig,
}

impl TrustRegistry {
    /// Create an empty registry with the given policy.
    pub fn new(config: TrustConfig) -> Self {
        Self {
            agents: HashMap::new(),
            revoked: HashSet::new(),
            config,
        }
    }

    /// Register a trusted agent (builder style).
    pub fn with_agent(mut self, agent: TrustedAgent) -> Self {
        self.agents.insert(agent.url.clone(), agent);
        self
    }

    /// Revoke an agent's credentials (CRL). Returns a new registry.
    pub fn revoke(mut self, url: &str) -> Self {
        self.revoked.insert(url.to_string());
        self
    }

    /// Whether `url` is a known, non-revoked agent.
    pub fn is_trusted(&self, url: &str) -> bool {
        self.agents.contains_key(url) && !self.revoked.contains(url)
    }

    /// Verify a single card against the registry entry for its own `url`.
    ///
    /// The directory attests to the agent's identity: the card must belong to a
    /// known, non-revoked agent and its signature must verify against the key
    /// registered for that agent.
    pub fn verify_card(&self, card: &AgentCard) -> Result<TrustVerification, SecurityError> {
        let agent = self
            .agents
            .get(&card.url)
            .ok_or_else(|| SecurityError::UntrustedAgent(card.url.clone()))?;
        if self.revoked.contains(&card.url) {
            return Err(SecurityError::RevokedAgent(card.url.clone()));
        }
        // Unlike the client (which tolerates unsigned cards with a warning), a
        // registry check REQUIRES a signature — an unsigned card has nothing
        // tying it to the registered key.
        if card.signature.as_deref().is_none_or(|s| s.is_empty()) {
            return Err(SecurityError::SignatureMismatch(
                card.url.clone(),
                "card is unsigned".to_string(),
            ));
        }
        verify_card_signature(card, &agent.verification_key)
            .map_err(|e| SecurityError::SignatureMismatch(card.url.clone(), e.to_string()))?;
        Ok(TrustVerification {
            url: card.url.clone(),
            role: agent.role.clone(),
            trust_score: self.config.effective_trust(agent.role.base_trust(), 0),
        })
    }

    /// Verify a delegation chain `root -> … -> leaf` of cards.
    ///
    /// Every hop must be a known, non-revoked agent. The root card must verify
    /// against its own registered key; each subsequent card must verify against
    /// the *previous* hop's key (the parent issued the child's certificate).
    /// The number of hops is bounded by [`TrustConfig::max_delegation_depth`],
    /// and the returned score is decayed once per hop.
    pub fn verify_chain(&self, cards: &[&AgentCard]) -> Result<TrustVerification, SecurityError> {
        if cards.is_empty() {
            return Err(SecurityError::UntrustedAgent("<empty chain>".to_string()));
        }
        let hops = cards.len() - 1;
        if hops > self.config.max_delegation_depth {
            return Err(SecurityError::DeepDelegation {
                url: cards.last().map(|c| c.url.clone()).unwrap_or_default(),
                depth: hops,
                limit: self.config.max_delegation_depth,
            });
        }

        // Root: self-attested against the registry's key for that agent.
        let root = self.verify_card(cards[0])?;

        // Each child must be signed by its parent. The parent is trusted
        // (verified above or in the previous iteration), so its key is used.
        for (i, child) in cards.iter().enumerate().skip(1) {
            let parent_url = cards[i - 1].url.clone();
            let parent_key = self
                .agents
                .get(&parent_url)
                .map(|a| a.verification_key.clone())
                .ok_or_else(|| SecurityError::UntrustedAgent(parent_url.clone()))?;
            if self.revoked.contains(&child.url) {
                return Err(SecurityError::RevokedAgent(child.url.clone()));
            }
            verify_card_signature(child, &parent_key)
                .map_err(|e| SecurityError::SignatureMismatch(child.url.clone(), e.to_string()))?;
        }

        let leaf = cards[cards.len() - 1];
        let leaf_agent = self
            .agents
            .get(&leaf.url)
            .ok_or_else(|| SecurityError::UntrustedAgent(leaf.url.clone()))?;
        Ok(TrustVerification {
            url: leaf.url.clone(),
            role: leaf_agent.role.clone(),
            trust_score: self.config.effective_trust(root.trust_score, hops),
        })
    }

    /// Effective trust score for a known agent after `hops` of delegation, or
    /// `None` if the agent is unknown or revoked.
    pub fn trust_score(&self, url: &str, hops: usize) -> Option<f64> {
        let agent = self.agents.get(url)?;
        if self.revoked.contains(url) {
            return None;
        }
        Some(self.config.effective_trust(agent.role.base_trust(), hops))
    }

    /// Convenience: sign a card as "issued" by `issuer_url` (certificate
    /// issuance analog). The card's `signature` is computed with the issuer's
    /// registered key, which is what [`Self::verify_chain`] expects of a child.
    pub fn issue_card(&self, issuer_url: &str, card: &mut AgentCard) -> Result<(), SecurityError> {
        let key = self
            .agents
            .get(issuer_url)
            .ok_or_else(|| SecurityError::UntrustedAgent(issuer_url.to_string()))?
            .verification_key
            .clone();
        sign_agent_card(card, &key)
            .map_err(|e| SecurityError::InvalidKey(issuer_url.to_string(), e.to_string()))
    }
}

/// A resource access attempt to be checked against a [`SandboxConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessRequest {
    /// Read a file/directory.
    ReadPath(PathBuf),
    /// Write a file/directory.
    WritePath(PathBuf),
    /// Contact a network host.
    Network(String),
}

/// Least-privilege limits for a delegated agent (P2-5 sandbox).
///
/// Path checks are prefix-based: a request is allowed when it is the allowed
/// directory itself or lives underneath it. Network checks allow the host
/// exactly, or any subdomain of an allowed domain.
#[derive(Debug, Clone, Default)]
pub struct SandboxConfig {
    allowed_read_paths: Vec<PathBuf>,
    allowed_write_paths: Vec<PathBuf>,
    allowed_domains: Vec<String>,
    max_payload_bytes: Option<usize>,
}

impl SandboxConfig {
    /// An empty sandbox that denies everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allow reading within `path` (builder style).
    pub fn allow_read(mut self, path: impl Into<PathBuf>) -> Self {
        self.allowed_read_paths.push(path.into());
        self
    }

    /// Allow writing within `path` (builder style).
    pub fn allow_write(mut self, path: impl Into<PathBuf>) -> Self {
        self.allowed_write_paths.push(path.into());
        self
    }

    /// Allow network access to `domain` and its subdomains (builder style).
    pub fn allow_domain(mut self, domain: impl Into<String>) -> Self {
        self.allowed_domains.push(domain.into());
        self
    }

    /// Cap the payload size accepted from the agent (builder style).
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload_bytes = Some(bytes);
        self
    }

    /// Whether a payload of `size` bytes is within the configured limit.
    pub fn accepts_payload(&self, size: usize) -> bool {
        match self.max_payload_bytes {
            Some(limit) => size <= limit,
            None => true,
        }
    }

    /// Enforce the sandbox for a single access request.
    pub fn check(&self, request: &AccessRequest) -> Result<(), SecurityError> {
        match request {
            AccessRequest::ReadPath(path) => self.check_read(path),
            AccessRequest::WritePath(path) => self.check_write(path),
            AccessRequest::Network(host) => self.check_network(host),
        }
    }

    fn check_read(&self, path: &Path) -> Result<(), SecurityError> {
        if self.allowed_read_paths.iter().any(|a| is_within(path, a)) {
            Ok(())
        } else {
            Err(SecurityError::SandboxDenied(format!(
                "read of `{}` is outside the allowed read roots",
                path.display()
            )))
        }
    }

    fn check_write(&self, path: &Path) -> Result<(), SecurityError> {
        if self.allowed_write_paths.iter().any(|a| is_within(path, a)) {
            Ok(())
        } else {
            Err(SecurityError::SandboxDenied(format!(
                "write of `{}` is outside the allowed write roots",
                path.display()
            )))
        }
    }

    fn check_network(&self, host: &str) -> Result<(), SecurityError> {
        if self.allowed_domains.iter().any(|d| domain_allows(host, d)) {
            Ok(())
        } else {
            Err(SecurityError::SandboxDenied(format!(
                "network access to `{host}` is not allowed"
            )))
        }
    }
}

/// Whether `path` is `root` or lives underneath it.
fn is_within(path: &Path, root: &Path) -> bool {
    path == root || path.starts_with(root)
}

/// Whether `host` is `domain` exactly or one of its subdomains.
fn domain_allows(host: &str, domain: &str) -> bool {
    let domain = domain.trim_start_matches('.');
    host == domain || host.ends_with(&format!(".{domain}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::sign_agent_card;

    fn card(url: &str) -> AgentCard {
        AgentCard::new("agent", "desc", url)
    }

    fn key() -> Vec<u8> {
        b"trust-secret".to_vec()
    }

    // ---- Trust registry ----

    #[test]
    fn registry_verifies_known_signed_card() {
        let mut c = card("https://a.example.com");
        sign_agent_card(&mut c, &key()).unwrap();

        let registry = TrustRegistry::new(TrustConfig::new(3, 0.9)).with_agent(
            TrustedAgent::new("https://a.example.com", "A", TrustRole::Leaf).with_key(key()),
        );
        let v = registry.verify_card(&c).unwrap();
        assert_eq!(v.url, "https://a.example.com");
        assert_eq!(v.role, TrustRole::Leaf);
        assert_eq!(v.trust_score, 0.6);
    }

    #[test]
    fn registry_rejects_unknown_agent() {
        let mut c = card("https://stranger.example.com");
        sign_agent_card(&mut c, &key()).unwrap();
        let registry = TrustRegistry::new(TrustConfig::new(3, 0.9));
        assert!(matches!(
            registry.verify_card(&c),
            Err(SecurityError::UntrustedAgent(_))
        ));
    }

    #[test]
    fn registry_rejects_revoked_agent() {
        let mut c = card("https://a.example.com");
        sign_agent_card(&mut c, &key()).unwrap();
        let registry = TrustRegistry::new(TrustConfig::new(3, 0.9))
            .with_agent(
                TrustedAgent::new("https://a.example.com", "A", TrustRole::Leaf).with_key(key()),
            )
            .revoke("https://a.example.com");
        assert!(matches!(
            registry.verify_card(&c),
            Err(SecurityError::RevokedAgent(_))
        ));
        assert!(!registry.is_trusted("https://a.example.com"));
    }

    #[test]
    fn registry_rejects_tampered_signature() {
        let mut c = card("https://a.example.com");
        sign_agent_card(&mut c, &key()).unwrap();
        c.description = "evil".to_string(); // tamper after signing

        let registry = TrustRegistry::new(TrustConfig::new(3, 0.9)).with_agent(
            TrustedAgent::new("https://a.example.com", "A", TrustRole::Leaf).with_key(key()),
        );
        assert!(matches!(
            registry.verify_card(&c),
            Err(SecurityError::SignatureMismatch(_, _))
        ));
    }

    #[test]
    fn issue_card_signs_child_with_parent_key() {
        let registry = TrustRegistry::new(TrustConfig::new(3, 0.9))
            .with_agent(
                TrustedAgent::new("https://root.example.com", "Root", TrustRole::Root)
                    .with_key(key()),
            )
            // Child is registered with its OWN key, which differs from the key
            // the parent uses to issue it.
            .with_agent(
                TrustedAgent::new("https://child.example.com", "Child", TrustRole::Leaf)
                    .with_key(b"child-key"),
            );
        let mut child = card("https://child.example.com");
        registry
            .issue_card("https://root.example.com", &mut child)
            .unwrap();
        // The child's card is signed by the parent, so direct verification
        // against the child's own registered key fails…
        assert!(matches!(
            registry.verify_card(&child),
            Err(SecurityError::SignatureMismatch(_, _))
        ));
        // …but a chain verifies the parent-issued signature (root self-signed).
        let mut root = card("https://root.example.com");
        sign_agent_card(&mut root, &key()).unwrap();
        let chain = registry.verify_chain(&[&root, &child]).unwrap();
        assert_eq!(chain.trust_score, 0.9); // 1.0 * 0.9^1
    }

    #[test]
    fn verify_chain_enforces_depth_limit() {
        // Build a chain of 4 cards = 3 hops, signed parent->child.
        let config = TrustConfig::new(2, 0.9);
        let registry = TrustRegistry::new(config.clone())
            .with_agent(TrustedAgent::new("https://r.com", "R", TrustRole::Root).with_key(key()))
            .with_agent(
                TrustedAgent::new("https://i.com", "I", TrustRole::Intermediate).with_key(key()),
            )
            .with_agent(
                TrustedAgent::new("https://i2.com", "I2", TrustRole::Intermediate).with_key(key()),
            )
            .with_agent(TrustedAgent::new("https://l.com", "L", TrustRole::Leaf).with_key(key()));

        // root -> i: 1 hop (allowed).
        let mut i = card("https://i.com");
        registry.issue_card("https://r.com", &mut i).unwrap();
        let mut root = card("https://r.com");
        sign_agent_card(&mut root, &key()).unwrap();
        assert!(registry.verify_chain(&[&root, &i]).is_ok());

        // root -> i -> i2 -> l: 3 hops (exceeds limit 2).
        let mut i2 = card("https://i2.com");
        registry.issue_card("https://i.com", &mut i2).unwrap();
        let mut l = card("https://l.com");
        registry.issue_card("https://i2.com", &mut l).unwrap();
        assert!(matches!(
            registry.verify_chain(&[&root, &i, &i2, &l]),
            Err(SecurityError::DeepDelegation {
                depth: 3,
                limit: 2,
                ..
            })
        ));

        // Relaxed config allows it and decays the score per hop.
        let wide = TrustRegistry::new(TrustConfig::new(3, 0.5))
            .with_agent(TrustedAgent::new("https://r.com", "R", TrustRole::Root).with_key(key()))
            .with_agent(
                TrustedAgent::new("https://i.com", "I", TrustRole::Intermediate).with_key(key()),
            )
            .with_agent(
                TrustedAgent::new("https://i2.com", "I2", TrustRole::Intermediate).with_key(key()),
            )
            .with_agent(TrustedAgent::new("https://l.com", "L", TrustRole::Leaf).with_key(key()));
        let v = wide.verify_chain(&[&root, &i, &i2, &l]).unwrap();
        // root base 1.0, decayed 0.5^3 = 0.125.
        assert!((v.trust_score - 0.125).abs() < 1e-9);
    }

    #[test]
    fn verify_chain_rejects_child_signed_by_wrong_parent() {
        let registry = TrustRegistry::new(TrustConfig::new(2, 0.9))
            .with_agent(TrustedAgent::new("https://r.com", "R", TrustRole::Root).with_key(key()))
            .with_agent(
                TrustedAgent::new("https://i.com", "I", TrustRole::Intermediate).with_key(b"other"),
            );

        // Root self-signed with its registered key.
        let mut root = card("https://r.com");
        sign_agent_card(&mut root, &key()).unwrap();
        // Child signed by its OWN key, which differs from the parent's
        // registered key — the chain must reject the parent-issued claim.
        let mut child = card("https://i.com");
        sign_agent_card(&mut child, b"other").unwrap();
        assert!(matches!(
            registry.verify_chain(&[&root, &child]),
            Err(SecurityError::SignatureMismatch(_, _))
        ));
    }

    #[test]
    fn trust_score_returns_none_for_unknown_or_revoked() {
        let registry = TrustRegistry::new(TrustConfig::new(2, 0.9))
            .with_agent(TrustedAgent::new("https://a.com", "A", TrustRole::Leaf).with_key(key()))
            .revoke("https://a.com");
        assert_eq!(registry.trust_score("https://a.com", 0), None);
        assert_eq!(registry.trust_score("https://nope.com", 0), None);
    }

    // ---- Sandbox ----

    #[test]
    fn sandbox_allows_reads_inside_allowed_root() {
        let sandbox = SandboxConfig::new().allow_read("C:/data");
        assert!(sandbox
            .check(&AccessRequest::ReadPath("C:/data/file.txt".into()))
            .is_ok());
        assert!(sandbox
            .check(&AccessRequest::ReadPath("C:/data".into()))
            .is_ok());
    }

    #[test]
    fn sandbox_denies_reads_outside_allowed_root() {
        let sandbox = SandboxConfig::new().allow_read("C:/data");
        assert!(matches!(
            sandbox.check(&AccessRequest::ReadPath("C:/other/secret.txt".into())),
            Err(SecurityError::SandboxDenied(_))
        ));
    }

    #[test]
    fn sandbox_read_and_write_roots_are_separate() {
        let sandbox = SandboxConfig::new()
            .allow_read("C:/in")
            .allow_write("C:/out");
        assert!(sandbox
            .check(&AccessRequest::ReadPath("C:/in/a.txt".into()))
            .is_ok());
        // Read of the write-only root is denied, and vice versa.
        assert!(sandbox
            .check(&AccessRequest::ReadPath("C:/out/a.txt".into()))
            .is_err());
        assert!(sandbox
            .check(&AccessRequest::WritePath("C:/in/a.txt".into()))
            .is_err());
        assert!(sandbox
            .check(&AccessRequest::WritePath("C:/out/a.txt".into()))
            .is_ok());
    }

    #[test]
    fn sandbox_network_allows_exact_and_subdomains() {
        let sandbox = SandboxConfig::new().allow_domain("example.com");
        assert!(sandbox
            .check(&AccessRequest::Network("example.com".into()))
            .is_ok());
        assert!(sandbox
            .check(&AccessRequest::Network("api.example.com".into()))
            .is_ok());
        assert!(sandbox
            .check(&AccessRequest::Network("evil.net".into()))
            .is_err());
        assert!(sandbox
            .check(&AccessRequest::Network("notexample.com".into()))
            .is_err());
    }

    #[test]
    fn sandbox_enforces_payload_limit() {
        let sandbox = SandboxConfig::new().with_max_payload(100);
        assert!(sandbox.accepts_payload(99));
        assert!(sandbox.accepts_payload(100));
        assert!(!sandbox.accepts_payload(101));
        let unbounded = SandboxConfig::new();
        assert!(unbounded.accepts_payload(usize::MAX));
    }
}
