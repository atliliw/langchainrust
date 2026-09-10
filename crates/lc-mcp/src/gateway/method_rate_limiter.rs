//! Method-header rate limiting for the stateless track (0.22.0 S2.6).
//!
//! The 2026-07-28 spec's `Mcp-Method` header exists precisely so gateways can
//! throttle **without parsing the body**. [`MethodRateLimiter`] keeps one
//! fixed window per method name; [`crate::StatelessMcpClient`] checks it
//! before sending each request.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Fixed-window rate limiter keyed by the `Mcp-Method` header value.
#[derive(Debug, Clone)]
pub struct MethodRateLimiter {
    max_calls: usize,
    window: Duration,
    windows: HashMap<String, WindowState>,
}

#[derive(Debug, Clone, Copy)]
struct WindowState {
    started: Instant,
    count: usize,
}

impl MethodRateLimiter {
    /// At most `max_calls` per `window` for each distinct method (min 1).
    pub fn new(max_calls: usize, window: Duration) -> Self {
        Self {
            max_calls: max_calls.max(1),
            window,
            windows: HashMap::new(),
        }
    }

    /// Tries to allow one call for `method`; resets the method's window when
    /// it has expired. `false` = limit hit for this method (others unaffected).
    pub fn allow(&mut self, method: &str) -> bool {
        let now = Instant::now();
        let state = self
            .windows
            .entry(method.to_string())
            .or_insert(WindowState {
                started: now,
                count: 0,
            });
        if now.duration_since(state.started) >= self.window {
            state.started = now;
            state.count = 0;
        }
        if state.count < self.max_calls {
            state.count += 1;
            true
        } else {
            false
        }
    }

    /// Methods with live windows (diagnostics).
    pub fn tracked_methods(&self) -> Vec<String> {
        let mut out: Vec<String> = self.windows.keys().cloned().collect();
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M8: limiting is per-method — exhausting `tools/call` does not affect
    /// `tools/list`.
    #[test]
    fn m8_limit_is_per_method() {
        let mut limiter = MethodRateLimiter::new(2, Duration::from_secs(60));
        assert!(limiter.allow("tools/call"));
        assert!(limiter.allow("tools/call"));
        assert!(!limiter.allow("tools/call"), "tools/call window exhausted");

        assert!(limiter.allow("tools/list"), "other methods unaffected");
        assert!(limiter.allow("tools/list"));
        assert!(!limiter.allow("tools/list"));
    }

    /// Window reset after expiry (short window, sleep past it).
    #[tokio::test]
    async fn window_resets_after_expiry() {
        let mut limiter = MethodRateLimiter::new(1, Duration::from_millis(30));
        assert!(limiter.allow("tools/call"));
        assert!(!limiter.allow("tools/call"));
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(limiter.allow("tools/call"), "window expired → reset");
    }

    /// Tracked methods are reported (diagnostics).
    #[test]
    fn tracked_methods_sorted() {
        let mut limiter = MethodRateLimiter::new(5, Duration::from_secs(60));
        let _ = limiter.allow("tools/call");
        let _ = limiter.allow("tools/list");
        assert_eq!(limiter.tracked_methods(), vec!["tools/call", "tools/list"]);
    }
}
