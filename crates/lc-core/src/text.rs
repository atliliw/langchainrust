//! Minimal text utilities.

/// Returns the prefix of `s` of at most `max_bytes` bytes, always ending on a
/// UTF-8 boundary.
///
/// This is the MSRV-safe equivalent of
/// `&s[..s.floor_char_boundary(max_bytes)]` (`str::floor_char_boundary` only
/// stabilized in Rust 1.91, while the workspace MSRV is 1.85). The cut is a byte
/// budget for log/error messages, not a character count.
///
/// # Examples
///
/// ```
/// use lc_core::text::truncate_at_char_boundary;
///
/// assert_eq!(truncate_at_char_boundary("abcdef", 3), "abc");
/// assert_eq!(truncate_at_char_boundary("abcdef", 100), "abcdef");
/// // Cutting in the middle of a multibyte char walks back to the boundary.
/// let s = "ab中cd"; // 中 is 3 bytes
/// let prefix = truncate_at_char_boundary(s, 3);
/// assert_eq!(prefix, "ab");
/// assert!(std::str::from_utf8(prefix.as_bytes()).is_ok());
/// ```
pub fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    // A UTF-8 sequence is at most 4 bytes; walk back at most 3 continuation bytes.
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::truncate_at_char_boundary;

    #[test]
    fn empty_and_zero_budget() {
        assert_eq!(truncate_at_char_boundary("", 200), "");
        assert_eq!(truncate_at_char_boundary("abc", 0), "");
        assert_eq!(truncate_at_char_boundary("中", 0), "");
    }

    #[test]
    fn exact_len_or_larger_returns_whole_slice() {
        let s = "ab中"; // 2 + 3 = 5 bytes
        assert_eq!(truncate_at_char_boundary(s, 5), s);
        assert_eq!(truncate_at_char_boundary(s, 6), s);
    }

    #[test]
    fn walks_back_inside_multibyte_every_offset() {
        // "a中" = 4 bytes: offsets 2 and 3 are inside the 3-byte char and must
        // both fall back to boundary 1; offset 4 is the char end (covered above).
        assert_eq!(truncate_at_char_boundary("a中", 1), "a");
        assert_eq!(truncate_at_char_boundary("a中", 2), "a");
        assert_eq!(truncate_at_char_boundary("a中", 3), "a");
    }

    #[test]
    fn result_is_always_valid_utf8() {
        // 4-byte emoji: every interior offset must round-trip through UTF-8.
        let s = "x😀y";
        for budget in 0..=s.len() {
            let prefix = truncate_at_char_boundary(s, budget);
            assert!(
                std::str::from_utf8(prefix.as_bytes()).is_ok(),
                "budget {budget} produced invalid UTF-8"
            );
        }
    }
}
