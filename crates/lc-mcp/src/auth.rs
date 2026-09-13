//! Authentication for the stateless MCP track (0.22.0 S2.5).
//!
//! Client side: [`AuthScheme`] attaches a bearer token to every request.
//! Server side: [`TokenValidator`] decides whether a presented token is
//! acceptable — including the RFC 9207 `iss` (issuer) check required by the
//! OAuth 2.1 resource-server model.
//!
//! # Signature verification (A15, security)
//!
//! [`JwtIssValidator`] requires an injected [`JwtSignatureVerifier`] (verify
//! against the issuer's JWKS, or call the IdP introspection endpoint) and
//! runs it **before** inspecting any claim: accepting a self-signed token's
//! `iss`/`exp` without checking the signature is an authentication bypass.
//! The claim-only type remains available as [`JwtIssAssertionValidator`],
//! but is `#[doc(hidden)]` and must never gate a real deployment by itself.

use serde::{Deserialize, Serialize};

/// Client-side auth scheme: what the transport attaches to requests.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthScheme {
    /// `Authorization: Bearer <token>` on every request.
    Bearer(String),
}

impl AuthScheme {
    /// The `Authorization` header value, if any.
    pub fn header_value(&self) -> Option<String> {
        match self {
            AuthScheme::Bearer(token) => Some(format!("Bearer {token}")),
        }
    }
}

/// Validated token claims relevant to MCP authorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claims {
    /// Subject (client identity).
    pub sub: String,
    /// Issuer (RFC 9207: must match the resource server's expectation).
    pub iss: String,
    /// Expiry (Unix seconds). `None` = no expiry claim (not recommended).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<i64>,
}

/// Server-side token validation. Implement against your IdP (JWKS signature
/// verification or token introspection); the built-in validators cover the
/// structural checks (bearer match, `iss` claim, expiry) for tests and
/// simple deployments.
#[async_trait::async_trait]
pub trait TokenValidator: Send + Sync {
    /// Returns the claims when the token is acceptable, otherwise an error
    /// (mapped to the JSON-RPC unauthorized error by the caller).
    async fn validate(&self, token: &str) -> Result<Claims, String>;
}

/// Accepts exactly one configured bearer token (tests / single-tenant).
pub struct StaticBearerValidator {
    expected: String,
}

impl StaticBearerValidator {
    /// Creates a validator accepting only `expected`.
    pub fn new(expected: impl Into<String>) -> Self {
        Self {
            expected: expected.into(),
        }
    }
}

#[async_trait::async_trait]
impl TokenValidator for StaticBearerValidator {
    async fn validate(&self, token: &str) -> Result<Claims, String> {
        // A17: constant-time comparison — a short-circuiting `==` leaks how
        // long a prefix the presented token shares with the expected one.
        // Token length is not secret (same trade-off as the `subtle` crate),
        // so the length mismatch may return early.
        if constant_time_eq(token, &self.expected) {
            Ok(Claims {
                sub: "static-bearer".into(),
                iss: "static".into(),
                exp: None,
            })
        } else {
            Err("bearer token mismatch".into())
        }
    }
}

/// Compares two strings in time dependent on the expected token's length, not
/// on the length of their shared prefix (A17): every byte is XOR-accumulated
/// and the loop never short-circuits. A length mismatch returns early — the
/// length of a bearer token is not a secret (mirrors `subtle::ConstantTimeEq`).
pub(crate) fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Cryptographic verification of a raw, complete JWT, supplied by the
/// deployment. Return `Ok(())` only after checking the signature against the
/// issuer's key (JWKS, with `kid`/`alg` allow-listing), or after a successful
/// OAuth token-introspection call. Any `Err` rejects the request.
///
/// The callback is shared (`Send + Sync`) because the validator is held
/// behind an `Arc` by MCP servers.
pub type JwtSignatureVerifier = std::sync::Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>;

/// Validates a JWT's signature (via the injected [`JwtSignatureVerifier`])
/// and then its `iss` claim (RFC 9207 resource-server check) and expiry.
///
/// The signature is checked **first**: claim checks on an unverified token
/// prove nothing, because the payload is just attacker-editable base64url.
/// There is deliberately no constructor that omits the verifier — tests and
/// offline harnesses that need claim-only checks should use
/// [`JwtIssAssertionValidator`].
pub struct JwtIssValidator {
    expected_iss: String,
    verify_signature: JwtSignatureVerifier,
}

impl JwtIssValidator {
    /// Creates a validator requiring a valid signature (per `verifier`) and
    /// `iss == expected_iss`.
    ///
    /// The closure must verify the token against the issuer's JWKS or an
    /// introspection endpoint; an always-`Ok` closure reintroduces the
    /// signature-bypass vulnerability this type exists to prevent.
    pub fn new(
        expected_iss: impl Into<String>,
        verifier: impl Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            expected_iss: expected_iss.into(),
            verify_signature: std::sync::Arc::new(verifier),
        }
    }

    /// Same as [`JwtIssValidator::new`] but takes a pre-built shared verifier
    /// (for example, a JWKS client shared across several validators).
    pub fn with_shared_verifier(
        expected_iss: impl Into<String>,
        verifier: JwtSignatureVerifier,
    ) -> Self {
        Self {
            expected_iss: expected_iss.into(),
            verify_signature: verifier,
        }
    }
}

/// Claim-only JWT validation (`iss` + expiry) **without any signature check**.
///
/// # Security warning
///
/// This accepts a token forged by anyone — the payload segment is plain
/// base64url and can be rewritten with any `iss`/`exp`. Use it **only** behind
/// an upstream component that has already verified the signature (an API
/// gateway / envoy that forwards pre-authenticated requests), or in tests.
/// For a self-contained resource server use [`JwtIssValidator`] with a JWKS /
/// introspection verifier. Hidden from the public API surface to prevent
/// accidental use (A15).
#[doc(hidden)]
pub struct JwtIssAssertionValidator {
    expected_iss: String,
}

impl JwtIssAssertionValidator {
    /// Creates a claim-only validator requiring `iss == expected_iss`.
    ///
    /// See the type-level security warning: the signature is NOT checked.
    pub fn new(expected_iss: impl Into<String>) -> Self {
        Self {
            expected_iss: expected_iss.into(),
        }
    }
}

/// Shared claim-level checks (`iss` and `exp`) on an already-decoded payload.
/// Signature verification, if any, is the caller's responsibility.
fn validate_claims(claims: &Claims, expected_iss: &str) -> Result<(), String> {
    if claims.iss != expected_iss {
        return Err(format!(
            "issuer mismatch: expected {expected_iss}, got {}",
            claims.iss
        ));
    }
    if let Some(exp) = claims.exp {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if exp < now {
            return Err("token expired".into());
        }
    }
    Ok(())
}

/// Minimal base64url decode (no external dep; payload-only, no padding).
fn base64url_decode(input: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut buf = Vec::with_capacity(input.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for b in input.bytes() {
        let v = TABLE.iter().position(|t| *t == b)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            buf.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Some(buf)
}

/// Decodes a JWT's payload segment (second dot-separated part) into claims.
///
/// **This does not verify the signature** — decoding is not authentication
/// (A15). Treat the returned claims as untrusted input until a
/// [`JwtSignatureVerifier`] has accepted the raw token.
pub fn decode_jwt_payload(token: &str) -> Option<Claims> {
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    let bytes = base64url_decode(payload)?;
    serde_json::from_slice(&bytes).ok()
}

#[async_trait::async_trait]
impl TokenValidator for JwtIssValidator {
    async fn validate(&self, token: &str) -> Result<Claims, String> {
        // A15: verify the cryptographic signature FIRST. Every subsequent check
        // is meaningless on a token an attacker could have forged.
        (self.verify_signature)(token)
            .map_err(|e| format!("JWT signature verification failed: {e}"))?;

        let claims =
            decode_jwt_payload(token).ok_or_else(|| "token is not a decodable JWT".to_string())?;
        validate_claims(&claims, &self.expected_iss)?;
        Ok(claims)
    }
}

#[async_trait::async_trait]
impl TokenValidator for JwtIssAssertionValidator {
    async fn validate(&self, token: &str) -> Result<Claims, String> {
        // A15: NO signature check here — see the type-level security warning.
        let claims =
            decode_jwt_payload(token).ok_or_else(|| "token is not a decodable JWT".to_string())?;
        validate_claims(&claims, &self.expected_iss)?;
        Ok(claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M7 helper: base64url roundtrip for the JWT payload decoder.
    #[test]
    fn base64url_roundtrip() {
        let raw = br#"{"sub":"a","iss":"https://idp.example"}"#;
        // encode manually via a known-good sample: use decode against a
        // precomputed value produced by standard base64url (no padding).
        let encoded = "eyJzdWIiOiJhIiwiaXNzIjoiaHR0cHM6Ly9pZHAuZXhhbXBsZSJ9";
        let decoded = base64url_decode(encoded).unwrap();
        assert_eq!(
            std::str::from_utf8(&decoded).unwrap(),
            std::str::from_utf8(raw).unwrap()
        );
    }

    /// Toy stand-in for a JWKS/introspection verifier: accepts only tokens
    /// carrying the pre-agreed signature segment `.good-sig`.
    fn require_good_sig(token: &str) -> Result<(), String> {
        if token.ends_with(".good-sig") {
            Ok(())
        } else {
            Err("signature does not match issuer key".to_string())
        }
    }

    /// Header `{"alg":"RS256"}` + payload
    /// `{"sub":"agent","iss":"https://idp.example","exp":9999999999}`.
    const GOOD_PAYLOAD: &str = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZ2VudCIsImlzcyI6Imh0dHBzOi8vaWRwLmV4YW1wbGUiLCJleHAiOjk5OTk5OTk5OTl9";

    /// A15: a token whose signature the verifier rejects is refused even
    /// though its `iss`/`exp` claims look perfect — the self-signed bypass.
    #[tokio::test]
    async fn forged_signature_is_rejected() {
        let forged = format!("{GOOD_PAYLOAD}.forged-signature");
        let v = JwtIssValidator::new("https://idp.example", require_good_sig);
        let err = v.validate(&forged).await.unwrap_err();
        assert!(err.contains("signature"), "{err}");
    }

    /// A15: a correctly signed token with matching claims is accepted.
    #[tokio::test]
    async fn valid_signature_and_claims_are_accepted() {
        let token = format!("{GOOD_PAYLOAD}.good-sig");
        let v = JwtIssValidator::new("https://idp.example", require_good_sig);
        let claims = v.validate(&token).await.unwrap();
        assert_eq!(claims.sub, "agent");
    }

    /// A valid signature does not excuse a mismatched issuer.
    #[tokio::test]
    async fn valid_signature_wrong_iss_rejected() {
        let token = format!("{GOOD_PAYLOAD}.good-sig");
        let v = JwtIssValidator::new("https://other.example", require_good_sig);
        let err = v.validate(&token).await.unwrap_err();
        assert!(err.contains("issuer mismatch"), "{err}");
    }

    /// A valid signature does not excuse an expired token.
    #[tokio::test]
    async fn valid_signature_expired_rejected() {
        // {"sub":"agent","iss":"i","exp":1}
        let token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhZ2VudCIsImlzcyI6ImkiLCJleHAiOjF9.good-sig";
        let v = JwtIssValidator::new("i", require_good_sig);
        let err = v.validate(token).await.unwrap_err();
        assert!(err.contains("expired"), "{err}");
    }

    /// A15: the deliberately hidden claim-only validator checks `iss`/`exp`
    /// and performs NO signature check (pre-authenticated-gateway use only).
    #[tokio::test]
    async fn assertion_validator_checks_claims_only() {
        let forged = format!("{GOOD_PAYLOAD}.attacker-signature");
        let v = JwtIssAssertionValidator::new("https://idp.example");
        let claims = v.validate(&forged).await.unwrap();
        assert_eq!(claims.sub, "agent");

        let bad_iss = JwtIssAssertionValidator::new("https://other.example");
        let err = bad_iss.validate(&forged).await.unwrap_err();
        assert!(err.contains("issuer mismatch"), "{err}");
    }

    /// Static bearer: exact match only (A17: including different lengths).
    #[tokio::test]
    async fn static_bearer_exact_match() {
        let v = StaticBearerValidator::new("secret-1");
        assert!(v.validate("secret-1").await.is_ok());
        assert!(v.validate("secret-2").await.is_err());
        assert!(v.validate("secret-1-extra").await.is_err());
        assert!(v.validate("").await.is_err());
    }

    /// A17: constant-time equality has the same boolean semantics as `==`.
    #[test]
    fn constant_time_eq_matches_equality() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "secreX"));
        assert!(!constant_time_eq("Xsecret", "secret"));
        assert!(!constant_time_eq("secret", "secre"));
        assert!(!constant_time_eq("secre", "secret"));
        assert!(constant_time_eq("", ""));
        assert!(!constant_time_eq("", "x"));
    }

    /// A17: comparison time must not depend on how long a prefix the presented
    /// token shares with the expected one. Timing measurements are noisy, so
    /// take the minimum over several rounds (filters scheduling/preemption
    /// noise) and use a loose bound; a short-circuiting `==` differs by a
    /// factor around the token length, a constant-time scan stays ~equal.
    #[test]
    fn constant_time_eq_timing_independent_of_shared_prefix() {
        use std::hint::black_box;
        use std::time::Instant;

        const LEN: usize = 256;
        let expected = "a".repeat(LEN);
        let mut wrong_first = String::from("b");
        wrong_first.extend(std::iter::repeat_n('a', LEN - 1));
        let mut wrong_last = "a".repeat(LEN - 1);
        wrong_last.push('b');
        assert_eq!(wrong_first.len(), LEN);
        assert_eq!(wrong_last.len(), LEN);

        const ROUNDS: u32 = 7;
        const ITERS: u32 = 20_000;
        let mut min_first = u128::MAX;
        let mut min_last = u128::MAX;
        for _ in 0..ROUNDS {
            let start = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(constant_time_eq)(
                    black_box(&wrong_first),
                    black_box(&expected),
                ));
            }
            min_first = min_first.min(start.elapsed().as_nanos());

            let start = Instant::now();
            for _ in 0..ITERS {
                black_box(black_box(constant_time_eq)(
                    black_box(&wrong_last),
                    black_box(&expected),
                ));
            }
            min_last = min_last.min(start.elapsed().as_nanos());
        }

        let ratio = min_first.max(min_last) as f64 / min_first.min(min_last).max(1) as f64;
        assert!(
            ratio < 5.0,
            "comparison time leaks shared-prefix length: first-byte mismatch {min_first}ns \
             vs last-byte mismatch {min_last}ns (ratio {ratio:.2})"
        );
    }

    /// AuthScheme header formatting.
    #[test]
    fn auth_scheme_header() {
        assert_eq!(
            AuthScheme::Bearer("t".into()).header_value().as_deref(),
            Some("Bearer t")
        );
    }
}
