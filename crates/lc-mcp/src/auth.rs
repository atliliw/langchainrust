//! Authentication for the stateless MCP track (0.22.0 S2.5).
//!
//! Client side: [`AuthScheme`] attaches a bearer token to every request.
//! Server side: [`TokenValidator`] decides whether a presented token is
//! acceptable — including the RFC 9207 `iss` (issuer) check required by the
//! OAuth 2.1 resource-server model. Note: this module validates claims, not
//! cryptographic signatures — production deployments must wire a validator
//! that verifies the token against the IdP (JWKS or introspection).

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
        if token == self.expected {
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

/// Validates a JWT-shaped token's `iss` claim (RFC 9207 resource-server
/// check) and its expiry. **This does not verify the signature** — in
/// production wrap your IdP verification (JWKS) around a custom
/// `TokenValidator`; this validator exists for the claim-level contract.
pub struct JwtIssValidator {
    expected_iss: String,
}

impl JwtIssValidator {
    /// Creates a validator requiring `iss == expected_iss`.
    pub fn new(expected_iss: impl Into<String>) -> Self {
        Self {
            expected_iss: expected_iss.into(),
        }
    }
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
        let claims =
            decode_jwt_payload(token).ok_or_else(|| "token is not a decodable JWT".to_string())?;
        if claims.iss != self.expected_iss {
            return Err(format!(
                "issuer mismatch: expected {}, got {}",
                self.expected_iss, claims.iss
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

    /// M7: matching iss is accepted; mismatched iss is rejected with a
    /// readable error.
    #[tokio::test]
    async fn iss_validation() {
        // {"sub":"agent","iss":"https://idp.example","exp":9999999999}
        let token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZ2VudCIsImlzcyI6Imh0dHBzOi8vaWRwLmV4YW1wbGUiLCJleHAiOjk5OTk5OTk5OTl9.sig";
        let v = JwtIssValidator::new("https://idp.example");
        let claims = v.validate(token).await.unwrap();
        assert_eq!(claims.sub, "agent");

        let bad = JwtIssValidator::new("https://other.example");
        let err = bad.validate(token).await.unwrap_err();
        assert!(err.contains("issuer mismatch"), "{err}");
    }

    /// Expired tokens are rejected.
    #[tokio::test]
    async fn expired_token_rejected() {
        // {"sub":"agent","iss":"i","exp":1}
        let token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhZ2VudCIsImlzcyI6ImkiLCJleHAiOjF9.sig";
        let v = JwtIssValidator::new("i");
        let err = v.validate(token).await.unwrap_err();
        assert!(err.contains("expired"), "{err}");
    }

    /// Static bearer: exact match only.
    #[tokio::test]
    async fn static_bearer_exact_match() {
        let v = StaticBearerValidator::new("secret-1");
        assert!(v.validate("secret-1").await.is_ok());
        assert!(v.validate("secret-2").await.is_err());
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
