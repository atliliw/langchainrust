// src/langgraph/edge.rs
//! Edge definition for LangGraph
//!
//! Edges define transitions between nodes. They can be fixed (always go to
//! the same target) or conditional (route based on state).

use super::state::StateSchema;
use super::errors::GraphError;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Edge target specification
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeTarget {
    /// Fixed target node
    Fixed(String),
    
    /// Conditional routing (target determined by routing function name)
    Conditional(String),
}

impl EdgeTarget {
    /// Create a fixed edge target
    pub fn to(node: impl Into<String>) -> Self {
        Self::Fixed(node.into())
    }
    
    /// Create a conditional edge target
    pub fn conditional(router: impl Into<String>) -> Self {
        Self::Conditional(router.into())
    }
}

/// Graph Edge enum
///
/// Represents a transition in the graph. Can be:
/// - Fixed: Always transitions to a specific node
/// - Conditional: Routes based on state via a routing function
#[derive(Debug, Clone)]
pub enum GraphEdge {
    /// Fixed edge from source to target
    Fixed {
        source: String,
        target: String,
    },
    
    /// Conditional edge with routing function
    Conditional {
        source: String,
        router_name: String,
        /// Map of routing result -> target node
        targets: HashMap<String, String>,
        /// Default target if no route matches
        default_target: Option<String>,
    },
}

impl GraphEdge {
    /// Create a fixed edge
    pub fn fixed(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self::Fixed {
            source: source.into(),
            target: target.into(),
        }
    }
    
    /// Create a conditional edge
    pub fn conditional<R, T>(
        source: impl Into<String>,
        router_name: impl Into<String>,
        targets: HashMap<R, T>,
        default: Option<T>,
    ) -> Self
    where
        R: Into<String>,
        T: Into<String>,
    {
        Self::Conditional {
            source: source.into(),
            router_name: router_name.into(),
            targets: targets.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
            default_target: default.map(|d| d.into()),
        }
    }
    
    /// Get source node
    pub fn source(&self) -> &str {
        match self {
            Self::Fixed { source, .. } => source,
            Self::Conditional { source, .. } => source,
        }
    }
    
    /// Get the target for a fixed edge
    pub fn fixed_target(&self) -> Option<&str> {
        match self {
            Self::Fixed { target, .. } => Some(target),
            Self::Conditional { .. } => None,
        }
    }
}

/// Conditional routing function trait
///
/// Routing functions examine the state and return a string key
/// that maps to the next node via the edge's target map.
#[async_trait::async_trait]
pub trait ConditionalEdge<S: StateSchema>: Send + Sync {
    /// Route to next node based on state
    ///
    /// Returns a string key that matches entries in the edge's targets map.
    async fn route(&self, state: &S) -> Result<String, GraphError>;
}

/// Function-based conditional router
pub struct FunctionRouter<S: StateSchema, F> {
    func: F,
    _marker: PhantomData<S>,
}

impl<S: StateSchema, F> FunctionRouter<S, F>
where
    F: Fn(&S) -> String + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func, _marker: PhantomData }
    }
}

#[async_trait::async_trait]
impl<S: StateSchema, F> ConditionalEdge<S> for FunctionRouter<S, F>
where
    F: Fn(&S) -> String + Send + Sync,
{
    async fn route(&self, state: &S) -> Result<String, GraphError> {
        Ok((self.func)(state))
    }
}

/// Async function-based conditional router
pub struct AsyncFunctionRouter<S: StateSchema, F> {
    func: F,
    _marker: PhantomData<S>,
}

impl<S: StateSchema, F, Fut> AsyncFunctionRouter<S, F>
where
    F: Fn(&S) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<String, GraphError>> + Send,
{
    pub fn new(func: F) -> Self {
        Self { func, _marker: PhantomData }
    }
}

#[async_trait::async_trait]
impl<S: StateSchema, F, Fut> ConditionalEdge<S> for AsyncFunctionRouter<S, F>
where
    F: Fn(&S) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<String, GraphError>> + Send,
{
    async fn route(&self, state: &S) -> Result<String, GraphError> {
        (self.func)(state).await
    }
}

/// Common routing keys
pub const ROUTE_CONTINUE: &str = "continue";
pub const ROUTE_END: &str = "end";
pub const ROUTE_ERROR: &str = "error";

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::AgentState;
    
    #[test]
    fn test_fixed_edge() {
        let edge = GraphEdge::fixed("start", "process");
        assert_eq!(edge.source(), "start");
        assert_eq!(edge.fixed_target(), Some("process"));
    }
    
    #[test]
    fn test_conditional_edge() {
        let targets = HashMap::from([
            ("continue", "next_node"),
            ("end", "__end__"),
        ]);
        let edge = GraphEdge::conditional("decision", "router", targets, None);
        assert_eq!(edge.source(), "decision");
        assert!(edge.fixed_target().is_none());
    }
    
    #[tokio::test]
    async fn test_function_router() {
        let router = FunctionRouter::new(|state: &AgentState| {
            if state.output.is_some() {
                ROUTE_END.to_string()
            } else {
                ROUTE_CONTINUE.to_string()
            }
        });
        
        let state = AgentState::new("test".to_string());
        let route = router.route(&state).await.unwrap();
        assert_eq!(route, ROUTE_CONTINUE);
    }
}