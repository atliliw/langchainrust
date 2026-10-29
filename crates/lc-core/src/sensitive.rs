//! Secret-bearing string wrapper.
//!
//! API keys and bearer tokens used to sit in provider configs as plain
//! `String`s, where an accidental `{:?}` / `{}` / log line leaks the
//! credential. [`SensitiveString`] replaces those fields so the secret is
//! redacted by construction:
//!
//! - `Debug` / `Display` never render the plaintext;
//! - the only way to read the secret is [`SensitiveString::expose_secret`],
//!   which is greppable and obvious at the call site;
//! - the backing bytes are overwritten with zeroes on drop, shrinking the
//!   window in which the secret sits in freed heap memory.
//!
//! # Serialization
//!
//! [`SensitiveString`] deliberately implements **neither** `Serialize` nor
//! `Deserialize`. Silently serializing a redacted placeholder (`"***"`) would
//! produce a config file that looks valid but authenticates with garbage,
//! while implementing a plaintext `Serialize` would re-open exactly the leak
//! this type exists to close. A caller that genuinely needs to persist a
//! secret must call [`SensitiveString::expose_secret`] and make that decision
//! explicitly.

use std::fmt;

/// A `String` holding a secret (API key, bearer token, client secret).
///
/// Construct with [`SensitiveString::new`] or `From<String>` / `From<&str>`;
/// read the plaintext only through [`SensitiveString::expose_secret`].
///
/// ```
/// use lc_core::SensitiveString;
///
/// let key = SensitiveString::new("sk-live-123");
/// assert_eq!(format!("{key:?}"), "SensitiveString(***)");
/// assert_eq!(format!("{key}"), "***");
/// assert_eq!(key.expose_secret(), "sk-live-123");
/// ```
#[derive(Clone, Default)]
pub struct SensitiveString(String);

impl SensitiveString {
    /// Wraps `value`, taking ownership of the secret bytes.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the plaintext secret.
    ///
    /// This is the only plaintext accessor; every call site is therefore an
    /// auditable place where a secret leaves the wrapper.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// Returns `true` when the secret is empty.
    ///
    /// Useful for the fail-fast "no API key configured" checks providers run
    /// before building a request.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the secret's length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Debug for SensitiveString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SensitiveString(***)")
    }
}

impl fmt::Display for SensitiveString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

impl From<String> for SensitiveString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SensitiveString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<&String> for SensitiveString {
    fn from(value: &String) -> Self {
        Self(value.clone())
    }
}

impl Drop for SensitiveString {
    fn drop(&mut self) {
        // SAFETY: `as_mut_vec` requires the string to currently hold valid
        // UTF-8 — it does, because every constructor (`new`, `From`,
        // `Default`, `Clone`) only ever stores a `String`. We hold `&mut self`,
        // so this is the only live reference to the buffer and no aliasing
        // reader can observe the bytes. Zeroing the buffer makes the contents
        // invalid UTF-8, which is why the value is never touched again after
        // this block: `Drop::drop` returns immediately and the `String`'s own
        // destructor only deallocates (it never reads the bytes).
        unsafe {
            let bytes = self.0.as_mut_vec();
            for byte in bytes.iter_mut() {
                // `write_volatile` keeps the overwrite from being elided as a
                // dead store to memory that is about to be freed.
                std::ptr::write_volatile(byte, 0);
            }
        }
        // Stop the compiler from sinking the volatile stores past the
        // deallocation that follows this destructor.
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_never_leak_the_secret() {
        let secret = SensitiveString::new("sk-live-super-secret");
        let debug = format!("{secret:?}");
        let display = format!("{secret}");
        assert!(!debug.contains("super-secret"), "debug leaked: {debug}");
        assert!(
            !display.contains("super-secret"),
            "display leaked: {display}"
        );
        assert_eq!(debug, "SensitiveString(***)");
        assert_eq!(display, "***");
    }

    #[test]
    fn expose_secret_returns_the_original() {
        let secret = SensitiveString::new("sk-live-super-secret");
        assert_eq!(secret.expose_secret(), "sk-live-super-secret");
        assert_eq!(secret.len(), 20);
        assert!(!secret.is_empty());
    }

    #[test]
    fn from_str_and_string_round_trip() {
        let from_str = SensitiveString::from("abc");
        let from_string = SensitiveString::from(String::from("abc"));
        let from_ref = SensitiveString::from(&String::from("abc"));
        let via_new = SensitiveString::new("abc");
        assert_eq!(from_str.expose_secret(), "abc");
        assert_eq!(from_string.expose_secret(), "abc");
        assert_eq!(from_ref.expose_secret(), "abc");
        assert_eq!(via_new.expose_secret(), "abc");
    }

    #[test]
    fn default_is_empty() {
        let empty = SensitiveString::default();
        assert!(empty.is_empty());
        assert_eq!(empty.expose_secret(), "");
    }

    #[test]
    fn clone_preserves_the_secret_and_drops_independently() {
        let original = SensitiveString::new("sk-clone");
        let copy = original.clone();
        drop(original);
        assert_eq!(copy.expose_secret(), "sk-clone");
    }

    /// The struct is usable behind a `&` from several threads — the drop-time
    /// wipe must not make it `!Send`/`!Sync`.
    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SensitiveString>();
    }
}
