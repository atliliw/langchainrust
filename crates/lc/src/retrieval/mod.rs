// crates/lc/src/retrieval/mod.rs
//! Bridge module: re-exports all types from the lc-rag crate.
//!
//! During the transition period, the main crate's `retrieval` module
//! simply re-exports everything from `lc_rag`, so all existing code
//! continues to work unchanged.
pub use lc_rag::*;
