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
fn constant_time_eq(a: &str, b: &str) -> bool {
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
