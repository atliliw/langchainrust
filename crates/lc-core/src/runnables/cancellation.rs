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

    // M4 (review): `cancelled()` was flagged as a lost-wakeup — the claim being
    // that `cancel()`'s `notify_waiters()` has no sticky permit, so a cancel
    // firing between the waiter's final flag re-check and its first poll would
    // be dropped and `cancelled().await` would park forever.
    //
    // Manual verification (2026-09-26) against the locked tokio 1.53.1:
    //  * `notify_waiters()` does NOT store a permit (SPIM notify.rs:749-754).
    //  * But `Notify::notified()` captures `notify_waiters_calls` at *creation*
    //    (SPIM notify.rs:569) and, on first poll, completes immediately if the
    //    counter changed since then (SPIM notify.rs:1121, 1153).
    //
    // So a future created *before* a `notify_waiters` completes even if it has
    // not been polled yet. Combined with the two flag re-checks in `cancelled()`
    // (lines 78 and 82), every interleaving of `cancel` vs `cancelled()` wakes:
    //   * cancel before re-check 78  -> line 78 returns.
    //   * cancel between 78 and future creation -> line 82 returns.
    //   * cancel between 82 and first poll -> counter bump completes the future
    //     (created before the bump) -> loop top returns.
    //   * cancel after registration -> `notify_waiters` wakes the registered
    //     waiter.
    // => no lost wake-up exists; M4 is a review false-positive on this tokio
    //    version. Do NOT "fix" by adding `notify_one()` — a stored permit next
    //    to `notify_waiters` is unnecessary and would only introduce spurious
    //    wake semantics. The tests below pin this down so a tokio upgrade that
    //    breaks the invariant fails loudly instead of silently hanging.

    /// Deterministic: a `notified()` future created *before* `notify_waiters`
    /// (but not yet polled) must still complete on first poll. This is the exact
    /// mechanism the M4-lost-wakeup window relies on.
    #[tokio::test]
    async fn notify_waiters_wakes_future_created_before_it_even_unpolled() {
        use std::time::Duration;
        let notify = Notify::new();
        // Create the future (takes the notify_waiters_calls snapshot) but do NOT
        // poll it yet — simulates the window between `cancelled()` creating the
        // future and its first poll/registration.
        let fut = notify.notified();
        // Broadcast with no waiter registered yet.
        notify.notify_waiters();
        // First poll: if the counter-snapshot invariant holds, resolves at once.
        tokio::time::timeout(Duration::from_millis(500), fut)
            .await
            .expect("unpolled future created before notify_waiters must not park");
    }

    /// Stress: hammer the exact M4 window — cancel racing the moment a fresh
    /// `cancelled()` future is created — and assert no waiter ever hangs. Runs
    /// under a timeout so a real regression fails rather than wedging the test.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancelled_never_parks_when_cancel_races_registration() {
        use std::time::Duration;

        let rounds = 5_000;
        for _ in 0..rounds {
            let token = CancellationToken::new();
            let clone = token.clone();

            // Race: spawner registers a waiter while main cancels immediately.
            let handle = tokio::spawn(async move { clone.cancelled().await });
            token.cancel();

            let _ = tokio::time::timeout(Duration::from_millis(500), handle)
                .await
                .expect("cancelled().await parked forever after cancel raced registration");
        }
    }

    /// Stress with clones: several tasks await `cancelled()` on the same token
    /// while cancellation is signalled from another task. All must wake.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_awaiters_wake_on_cancel() {
        use std::time::Duration;
        let token = CancellationToken::new();
        let mut handles = Vec::new();
        for _ in 0..16 {
            let clone = token.clone();
            handles.push(tokio::spawn(async move { clone.cancelled().await }));
        }
        token.cancel();
        for h in handles {
            tokio::time::timeout(Duration::from_millis(500), h)
                .await
                .expect("an awaiter never woke after cancel")
                .unwrap();
        }
        assert!(token.is_cancelled());
    }
}
