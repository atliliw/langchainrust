//! lc-testkit — a record/replay harness for the framework.
//!
//! Lets the framework self-test without an API key:
//! - [`RecordingProvider`]: wraps any `BaseChatModel`, makes one real call, then appends the
//!   request/response pair to a JSONL file.
//! - [`ReplayProvider`]: replays from the recording file in FIFO order, zero network.
//! - N2 (v0.24.0): [`golden_dataset`] / [`ReplayPredictor`] sink recordings into
//!   `lc-evaluation` golden sets and score them offline — the trace → golden → regression loop.
//!
//! Data format and usage are described in `docs/internal/v0.16.0/HARNESS_DESIGN.md` Part A.

mod error;
mod golden;
mod recording;
mod replay;

pub use error::TestkitError;
pub use golden::{
    golden_dataset, golden_dataset_from_file, is_scoring_candidate, replay_golden_from_file,
    write_golden_jsonl, GoldenError, ReplayPredictor,
};
pub use recording::{read_exchanges, RecordedExchange, Recorder, RecordingProvider};
pub use replay::{ReplayProvider, ReplayStrategy};
