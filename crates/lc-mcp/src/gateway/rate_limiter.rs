//! 固定窗口速率限制器(P2-8)。

use std::time::{Duration, Instant};

/// 固定窗口速率限制器(P2-8):窗口内最多放行 `max_calls` 次,窗口过期重置。
///
/// 每 Server 一个实例,`call` 入口按 Server 名命中并 `allow()`。
#[derive(Debug, Clone)]
pub struct RateLimiter {
    max_calls: usize,
    window: Duration,
    window_start: Instant,
    count: usize,
}

impl RateLimiter {
    /// 创建窗口限流器:`window` 内最多 `max_calls` 次调用(至少 1)。
    pub fn new(max_calls: usize, window: Duration) -> Self {
        Self {
            max_calls: max_calls.max(1),
            window,
            window_start: Instant::now(),
            count: 0,
        }
    }

    /// 尝试放行一次;窗口已过则重置计数再判。
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

    /// 当前窗口内剩余可放行次数。
    pub fn remaining(&self) -> usize {
        if self.window_start.elapsed() >= self.window {
            self.max_calls
        } else {
            self.max_calls.saturating_sub(self.count)
        }
    }
}
