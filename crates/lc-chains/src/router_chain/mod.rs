// lc-chains/src/router_chain/mod.rs
//! Router Chain
//!
//! Automatically routes to different Chains based on input content.

pub mod base;
pub mod destination;
pub mod llm;

pub use base::RouterChain;
pub use destination::RouteDestination;
pub use llm::LLMRouterChain;

pub(crate) use base::{route_tool, RouteDecision};

#[cfg(test)]
mod tests;
