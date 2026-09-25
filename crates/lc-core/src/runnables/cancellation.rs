// lc-core/src/runnables/cancellation.rs
//! CancellationToken for aborting long-running operations.
//!
//! # Example
//!
//! ```rust,ignore
//! use lc_core::runnables::CancellationToken;
//! use std::time::Duration;
//!
//! let token = CancellationToken::new();
//!
//! // In another task, cancel after 30 seconds
//! let cloned = token.clone();
//! tokio::spawn(async move {
//!     tokio::time::sleep(Duration::from_secs(30)).await;
//!     cloned.cancel();
//! });
//!
//! // In the agent loop, check for cancellation
//! if token.is_cancelled() {
//!     return Ok("Agent stopped by cancellation".to_string());
//! }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

/// A token that can be used to signal cancellation of a long-running operation.
///
/// Clones share the same underlying cancellation state — cancelling one clone
/// cancels all of them.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl CancellationToken {
    /// Creates a new, uncancelled token.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Signals cancellation.
    ///
    /// All clones of this token will become cancelled.
    pub fn cancel(&self) {
        // Only the transition false→true wakes waiters, so repeated cancels
        // (possibly from several tasks) never produce spurious notifications.
        if self
            .inner
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.notify.notify_waiters();
        }
    }

    /// Returns `true` if cancellation has been signaled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::SeqCst)
    }

    /// Returns a future that resolves when cancellation is signaled.
    ///
    /// Backed by [`tokio::sync::Notify`] — the future parks without busy-looping
    /// and wakes immediately on [`cancel()`](Self::cancel). The flag is
    /// re-checked after registering interest to close the race where
    /// cancellation happens between the flag read and the `notified()`
    /// registration.
    pub async fn cancelled(&self) {
        loop {
            if self.inner.load(Ordering::SeqCst) {
                return;
            }
            let notified = self.notify.notified();
            if self.inner.load(Ordering::SeqCst) {
                return;
            }
            notified.await;
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_token_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_sets_is_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_clone_shares_cancellation_state() {
        let token = CancellationToken::new();
        let clone = token.clone();

        assert!(!token.is_cancelled());
        assert!(!clone.is_cancelled());

        clone.cancel();

        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());
    }

    #[test]
    fn test_multiple_clones_all_cancelled() {
        let token = CancellationToken::new();
        let c1 = token.clone();
        let c2 = token.clone();
        let c3 = token.clone();

        token.cancel();

        assert!(c1.is_cancelled());
        assert!(c2.is_cancelled());
        assert!(c3.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancelled_future_resolves() {
        let token = CancellationToken::new();

        // Cancel in a background task
        let cloned = token.clone();
        tokio::spawn(async move {
            cloned.cancel();
        });

        // This should resolve quickly
        token.cancelled().await;
        assert!(token.is_cancelled());
    }
}
