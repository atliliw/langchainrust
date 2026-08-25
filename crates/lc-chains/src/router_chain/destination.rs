// lc-chains/src/router_chain/destination.rs
//! Route destination for the router chain.

use std::sync::Arc;

use crate::base::BaseChain;

/// Route destination.
pub struct RouteDestination {
    /// Destination name.
    name: String,
    /// Destination description (used for routing decisions).
    description: String,
    /// Destination Chain.
    chain: Arc<dyn BaseChain>,
    /// Keyword list (used for keyword-based routing).
    keywords: Vec<String>,
}

impl RouteDestination {
    /// Create a new route destination.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        chain: Arc<dyn BaseChain>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            chain,
            keywords: Vec::new(),
        }
    }

    /// Set the keyword list used for keyword-based routing.
    pub fn with_keywords(mut self, keywords: Vec<&str>) -> Self {
        self.keywords = keywords.into_iter().map(String::from).collect();
        self
    }

    /// Get the destination name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the destination description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the destination chain.
    pub fn chain(&self) -> &Arc<dyn BaseChain> {
        &self.chain
    }

    /// Get the keyword list.
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }
}
