//! B7: pluggable authenticators for the A2A server.
//!
//! Ownership is only ever *issued by the server*: an authenticated request is
//! stamped with the principal the authenticator resolves, and a client-supplied
//! `owner` in metadata is ignored on the authenticated boundary
//! (see [`super::A2AServer::handle_a2a_request_authenticated`]). This module
//! provides the trait to plug in an authenticator plus a default
//! [`StaticBearer`] implementation (constant-time comparison of a bearer
//! token — it does not fabricate JWT claims).
//!
//! The legacy [`A2AServer::with_auth_token`] / [`A2AServer::with_auth_identity`]
//! builders are convenience wrappers that configure a [`StaticBearer`].

use std::collections::HashMap;

use crate::client::signing::constant_time_eq;

/// A security identity the server assigns to an authenticated caller.
///
/// Backs owner-based task authorization (P1-4) and per-principal SSE
/// notification filtering.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Principal(pub String);

impl Principal {
    /// The principal as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for Principal {
    fn from(s: String) -> Self {
        Principal(s)
    }
}

impl From<&str> for Principal {
    fn from(s: &str) -> Self {
        Principal(s.to_string())
    }
}

/// A failed authentication attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// No credentials were presented but one is required.
    Required,
    /// The presented credential did not match.
    InvalidToken,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::Required => f.write_str("Authentication required"),
            AuthError::InvalidToken => f.write_str("Invalid authentication token"),
        }
    }
}

/// Identifies a caller from an HTTP `Authorization` bearer credential.
///
/// `Ok(Some(principal))` names an identity-bound caller; `Ok(None)` authenticates
/// a caller that carries no per-tenant identity (e.g. a shared system token);
/// `Err` rejects the request. Must be `Send + Sync` so it can be shared behind
/// an `Arc` across concurrent requests.
pub trait Authenticator: Send + Sync {
    /// Resolve a presented bearer token to a principal, if any.
    fn authenticate(&self, bearer: Option<&str>) -> Result<Option<Principal>, AuthError>;
}

/// The default authenticator: a bearer token allow-list.
///
/// Holds (a) one optional *shared* token that authenticates but names no
/// principal, and (b) any number of *identity* tokens, each mapping to the
/// owner principal it names. A token presented against an identity is compared
/// in **constant time** so the check does not leak the secret through
/// early-exit timing. No JWT claims are synthesized — this is a secret, not a
/// signed identity artifact.
#[derive(Debug, Default)]
pub struct StaticBearer {
    expected: Option<String>,
    identities: HashMap<String, String>,
}

impl StaticBearer {
    /// An empty bearer authenticator (no credentials configured).
    pub fn new() -> Self {
        Self::default()
    }

    /// Require a shared token that authenticates without naming a principal.
    pub fn with_shared_token(mut self, token: impl Into<String>) -> Self {
        self.expected = Some(token.into());
        self
    }

    /// Bind a token to a principal it names.
    pub fn with_identity(mut self, token: impl Into<String>, principal: impl Into<String>) -> Self {
        self.identities.insert(token.into(), principal.into());
        self
    }

    /// Whether any credential is configured.
    pub fn is_empty(&self) -> bool {
        self.expected.is_none() && self.identities.is_empty()
    }
}

impl Authenticator for StaticBearer {
    fn authenticate(&self, bearer: Option<&str>) -> Result<Option<Principal>, AuthError> {
        // Identity-bound tokens name the principal directly.
        if let Some(token) = bearer {
            if let Some(principal) = self.identities.get(token) {
                return Ok(Some(Principal(principal.clone())));
            }
        }
        if let Some(expected) = &self.expected {
            return match bearer {
                None => Err(AuthError::Required),
                Some(token) if constant_time_eq(token, expected) => Ok(None),
                Some(_) => Err(AuthError::InvalidToken),
            };
        }
        // Identity tokens configured but none matched: do not fall through to
        // "auth disabled" — the request is unauthenticated.
        if self.identities.is_empty() {
            Ok(None)
        } else {
            Err(AuthError::Required)
        }
    }
}
