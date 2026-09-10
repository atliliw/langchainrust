// crates/lc/src/sessions/mod.rs
//! Sessions — re-export from lc-sessions crate
// The legacy SessionManager is deprecated in favor of the events API but
// remains fully functional until 0.23.0.
#[allow(deprecated)]
pub use lc_sessions::*;
