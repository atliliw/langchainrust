//! Fixed-window rate limiter (P2-8).

use std::time::{Duration, Instant};

/// Fixed-window rate limiter (P2-8): at most `max_calls` allowed per window, reset when the window expires.
///
/// One instance per server; the `call` entry point hits it by server name and calls `allow()`.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    max_calls: usize,
    window: Duration,
    window_start: Instant,
    count: usize,
}

impl RateLimiter {
    /// Creates a window limiter: at most `max_calls` calls per `window` (minimum 1).
    pub fn new(max_calls: usize, window: Duration) -> Self {
        Self {
            max_calls: max_calls.max(1),
            window,
            window_start: Instant::now(),
            count: 0,
        }
    }

    /// Tries to allow one call; if the window has passed, resets the count first, then decides.
    pub fn allow(&mut self) -> bool {
        if self.window_start.elapsed() >= self.window {
            self.window_start = Instant::now();
            self.count = 0;
        }
        if self.count < self.max_calls {
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Number of calls still allowed within the current window.
    pub fn remaining(&self) -> usize {
        if self.window_start.elapsed() >= self.window {
            self.max_calls
        } else {
            self.max_calls.saturating_sub(self.count)
        }
    }
}
