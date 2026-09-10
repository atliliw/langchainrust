// ---- P1-3: Agent Card HMAC signatures ----

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::protocol::AgentCard;

use super::A2AError;

/// Sign an agent card with a shared HMAC-SHA256 secret (P1-3).
///
/// The signature is computed over the canonical JSON of the card with the
/// `signature` field stripped, and stored hex-encoded in `card.signature`.
/// Verify with [`verify_card_signature`].
pub fn sign_agent_card(card: &mut AgentCard, secret: &[u8]) -> Result<(), A2AError> {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid signing secret length".to_string()))?;
    mac.update(&canonical_card_bytes(card)?);
    let tag = mac.finalize().into_bytes();
    card.signature = Some(hex_encode(&tag));
    Ok(())
}

/// Verify an agent card's HMAC-SHA256 signature (P1-3).
///
/// Returns `Ok(())` for unsigned cards (nothing to verify). A card whose
/// signature does not match `secret` (or is malformed) yields a
/// [`A2AError::Signature`].
pub fn verify_card_signature(card: &AgentCard, secret: &[u8]) -> Result<(), A2AError> {
    let sig = match card.signature.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => return Ok(()),
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid verification secret length".to_string()))?;
    mac.update(&canonical_card_bytes(card)?);
    let expected = hex_encode(&mac.finalize().into_bytes());
    if constant_time_eq(sig, &expected) {
        Ok(())
    } else {
        Err(A2AError::Signature(
            "agent card signature verification failed".to_string(),
        ))
    }
}

/// Canonical bytes of a card for signing: the card JSON with `signature`
/// removed so signatures don't cover themselves.
fn canonical_card_bytes(card: &AgentCard) -> Result<Vec<u8>, A2AError> {
    let mut value = serde_json::to_value(card)
        .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {}", e)))?;
    if let Some(obj) = value.as_object_mut() {
        obj.remove("signature");
    }
    serde_json::to_vec(&value)
        .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {}", e)))
}

/// Lowercase hex encoding.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{:02x}", b).expect("writing to a String cannot fail");
    }
    s
}

/// Constant-time string comparison (avoids leaking the expected signature).
pub(crate) fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let a = a.as_bytes();
    let b = b.as_bytes();
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---- v1.0.1: JWS (compact, HS256) over RFC 8785-canonical card JSON ----

/// RFC 8785-style canonical JSON (approximation): object keys sorted
/// recursively, no insignificant whitespace. Sufficient for deterministic
/// signing of our own cards; full JCS test-vector parity is a 0.22.1 item.
pub fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value as V;
    match value {
        V::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonical_json(&map[k]));
            }
            V::Object(out)
        }
        V::Array(items) => V::Array(items.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

fn base64url_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for b in bytes {
        acc = (acc << 8) | *b as u32;
        bits += 8;
        while bits >= 6 {
            bits -= 6;
            let idx = ((acc >> bits) & 0x3F) as usize;
            out.push(TABLE[idx] as char);
        }
    }
    if bits > 0 {
        let idx = ((acc << (6 - bits)) & 0x3F) as usize;
        out.push(TABLE[idx] as char);
    }
    out
}

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

/// Signs an agent card as a **compact JWS (HS256)** over the canonical card
/// JSON (v1.0.1: JWS per RFC 7515; the signing secret travels out-of-band —
/// configured, never hardcoded).
///
/// Returns the compact JWS `header.payload.signature` (all base64url). The
/// card itself is not modified; store the returned string wherever the card
/// is served (e.g. alongside `/.well-known/agent-card.json`).
pub fn sign_card_jws(card: &AgentCard, secret: &[u8]) -> Result<String, A2AError> {
    let value = canonical_json(
        &serde_json::to_value(card)
            .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {e}")))?,
    );
    let payload = serde_json::to_vec(&value)
        .map_err(|e| A2AError::Parse(format!("Failed to canonicalize agent card: {e}")))?;
    let header = br#"{"alg":"HS256","typ":"JWS"}"#;
    let signing_input = format!(
        "{}.{}",
        base64url_encode(header),
        base64url_encode(&payload)
    );
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid JWS secret length".to_string()))?;
    mac.update(signing_input.as_bytes());
    let sig = base64url_encode(&mac.finalize().into_bytes());
    Ok(format!("{signing_input}.{sig}"))
}

/// Verifies a compact JWS (HS256) produced by [`sign_card_jws`] against the
/// card's canonical form. Tampering with either the card or the JWS fails.
pub fn verify_card_jws(card: &AgentCard, jws: &str, secret: &[u8]) -> Result<(), A2AError> {
    let value = canonical_json(
        &serde_json::to_value(card)
            .map_err(|e| A2AError::Parse(format!("Failed to serialize agent card: {e}")))?,
    );
    let payload = serde_json::to_vec(&value)
        .map_err(|e| A2AError::Parse(format!("Failed to canonicalize agent card: {e}")))?;
    let header = br#"{"alg":"HS256","typ":"JWS"}"#;
    let signing_input = format!(
        "{}.{}",
        base64url_encode(header),
        base64url_encode(&payload)
    );

    let mut parts = jws.split('.');
    let h = parts
        .next()
        .ok_or_else(|| A2AError::Signature("malformed JWS: no header".into()))?;
    let p = parts
        .next()
        .ok_or_else(|| A2AError::Signature("malformed JWS: no payload".into()))?;
    let s = parts
        .next()
        .ok_or_else(|| A2AError::Signature("malformed JWS: no signature".into()))?;
    if parts.next().is_some() {
        return Err(A2AError::Signature("malformed JWS: extra segments".into()));
    }
    // Header must declare HS256 (no alg-confusion).
    let header_json: serde_json::Value = serde_json::from_slice(
        &base64url_decode(h)
            .ok_or_else(|| A2AError::Signature("JWS header not base64url".into()))?,
    )
    .map_err(|e| A2AError::Signature(format!("JWS header not JSON: {e}")))?;
    if header_json.get("alg").and_then(|a| a.as_str()) != Some("HS256") {
        return Err(A2AError::Signature(
            "JWS alg mismatch: expected HS256".into(),
        ));
    }
    let expected_payload = base64url_encode(&payload);
    if p != expected_payload {
        return Err(A2AError::Signature(
            "JWS payload does not match the card".into(),
        ));
    }
    let mut mac = Hmac::<Sha256>::new_from_slice(secret)
        .map_err(|_| A2AError::Signature("invalid JWS secret length".to_string()))?;
    mac.update(signing_input.as_bytes());
    let expected_sig = base64url_encode(&mac.finalize().into_bytes());
    if constant_time_eq(s, &expected_sig) {
        Ok(())
    } else {
        Err(A2AError::Signature(
            "JWS signature verification failed".into(),
        ))
    }
}

#[cfg(test)]
mod jws_tests {
    use super::*;
    use crate::protocol::{A2ATransport, AgentInterface, A2A_VERSION_V101};

    fn card() -> AgentCard {
        AgentCard::new("agent-a", "test agent", "https://a.example").with_supported_interface(
            AgentInterface::new(
                A2A_VERSION_V101,
                A2ATransport::HttpJson,
                "https://a.example/a2a",
            ),
        )
    }

    /// JWS roundtrip: sign → verify passes.
    #[test]
    fn jws_roundtrip() {
        let card = card();
        let jws = sign_card_jws(&card, b"secret-1").unwrap();
        assert_eq!(jws.split('.').count(), 3);
        verify_card_jws(&card, &jws, b"secret-1").unwrap();
    }

    /// Tampering with the card breaks verification.
    #[test]
    fn jws_detects_card_tamper() {
        let jws = sign_card_jws(&card(), b"secret-1").unwrap();
        let tampered = AgentCard::new("agent-a", "HACKED", "https://a.example");
        assert!(verify_card_jws(&tampered, &jws, b"secret-1").is_err());
    }

    /// Tampering with the JWS signature breaks verification.
    #[test]
    fn jws_detects_signature_tamper() {
        let card = card();
        let jws = sign_card_jws(&card, b"secret-1").unwrap();
        let mut bad = jws.clone();
        bad.pop();
        bad.push('A');
        assert!(verify_card_jws(&card, &bad, b"secret-1").is_err());
    }

    /// Wrong secret fails; alg confusion rejected.
    #[test]
    fn jws_rejects_wrong_secret_and_alg() {
        let card = card();
        let jws = sign_card_jws(&card, b"secret-1").unwrap();
        assert!(verify_card_jws(&card, &jws, b"secret-2").is_err());

        let parts: Vec<&str> = jws.split('.').collect();
        let forged_header = base64url_encode(br#"{"alg":"none","typ":"JWS"}"#);
        let forged = format!("{}.{}.{}", forged_header, parts[1], parts[2]);
        assert!(verify_card_jws(&card, &forged, b"secret-1").is_err());
    }

    /// Canonical JSON sorts keys recursively (determinism across builds).
    #[test]
    fn canonical_json_sorts_keys() {
        let v: serde_json::Value = serde_json::json!({ "b": 1, "a": { "y": 2, "x": 3 } });
        let c = canonical_json(&v);
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(s, r#"{"a":{"x":3,"y":2},"b":1}"#);
    }

    /// v1.0.1 negotiation: matching transport+version picked; miss errors.
    #[test]
    fn v101_negotiation() {
        let card = card();
        let iface = card
            .negotiate(A2ATransport::HttpJson, &[A2A_VERSION_V101])
            .unwrap();
        assert_eq!(iface.protocol_version, A2A_VERSION_V101);
        assert!(card
            .negotiate(A2ATransport::Grpc, &[A2A_VERSION_V101])
            .is_err());
        assert!(card.negotiate(A2ATransport::HttpJson, &["0.3.0"]).is_err());
    }

    /// Tenant propagation on interfaces.
    #[test]
    fn interface_tenant() {
        let iface =
            AgentInterface::new("1.0.1", A2ATransport::HttpJson, "https://a").with_tenant("acme");
        assert_eq!(iface.tenant.as_deref(), Some("acme"));
        let json = serde_json::to_string(&iface).unwrap();
        assert!(json.contains("\"tenant\":\"acme\""), "{json}");
    }
}
