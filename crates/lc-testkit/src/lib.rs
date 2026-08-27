//! lc-testkit — a record/replay harness for the framework.
//!
//! Lets the framework self-test without an API key:
//! - [`RecordingProvider`]: wraps any `BaseChatModel`, makes one real call, then appends the
//!   request/response pair to a JSONL file.
//! - [`ReplayProvider`]: replays from the recording file in FIFO order, zero network.
//!
//! Data format and usage are described in `docs/internal/v0.16.0/HARNESS_DESIGN.md` Part A.

mod error;
mod recording;
mod replay;

pub use error::TestkitError;
pub use recording::{RecordedExchange, Recorder, RecordingProvider};
pub use replay::{ReplayProvider, ReplayStrategy};
