// lc-core/src/crypto.rs
//! Shared cryptographic primitives.

/// Compares two strings in constant time, independent of the length of their
/// shared prefix.
///
/// Every byte is XOR-accumulated and the loop never short-circuits, so the
/// execution time reveals nothing about how many leading bytes match. Only a
/// length mismatch returns early — lengths are treated as public/public-ish
/// (a bearer token's length is not secret).
///
/// This is the **single implementation** for the whole workspace. It was
/// extracted from `lc-a2a` and `lc-mcp`, which each carried an identical
/// byte-by-byte copy (`framework-code-review, Class 1 #1[direction:跨 crate 抽共享]`);
/// a security comparison must not live in two places that can drift.
/// Mirrors [`subtle::ConstantTimeEq`]'s contract without adding a dependency.
pub fn constant_time_eq(a: &str, b: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_exact_equality_and_rejects_diffs() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "secreX"));
        assert!(!constant_time_eq("Xsecret", "secret"));
        assert!(!constant_time_eq("secret", "secre"));
        assert!(!constant_time_eq("secre", "secret"));
        assert!(constant_time_eq("", ""));
        assert!(!constant_time_eq("", "x"));
    }
}