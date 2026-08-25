//! lc-testkit — 测试录播/回放 harness。
//!
//! 让框架"没 key 也能自测":
//! - [`RecordingProvider`]:包裹任意 `BaseChatModel`,真实调用一次后把请求/响应对追加到 JSONL。
//! - [`ReplayProvider`]:从录制文件按 FIFO 顺序回放,零网络。
//!
//! 数据格式与使用方式见 `docs/internal/v0.16.0/HARNESS_DESIGN.md` Part A。

mod error;
mod recording;
mod replay;

pub use error::TestkitError;
pub use recording::{RecordedExchange, Recorder, RecordingProvider};
pub use replay::ReplayProvider;
