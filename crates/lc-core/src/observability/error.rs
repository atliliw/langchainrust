// lc-core/src/observability/error.rs
//! Export error type for the observability layer.

use std::fmt;

/// Export error. Implementations surface failures through this and the framework
/// layer only logs a `warn` — it never propagates into the main flow.
#[derive(Debug)]
#[non_exhaustive]
pub enum ObsError {
    /// Transport failure (I/O, network, database).
    Transport(String),
    /// Serialization failure.
    Encode(String),
}

impl fmt::Display for ObsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObsError::Transport(msg) => write!(f, "observability transport error: {msg}"),
            ObsError::Encode(msg) => write!(f, "observability encode error: {msg}"),
        }
    }
}

impl std::error::Error for ObsError {}
