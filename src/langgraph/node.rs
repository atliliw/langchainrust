// src/langgraph/node.rs
//! Node definition for LangGraph
//!
//! Nodes are the execution units in a graph. Each node receives the current
//! state and returns a state update.

use async_trait::async_trait;
use super::state::{StateSchema, StateUpdate};
use super::errors::GraphError;
use std::future::Future;
use std::pin::Pin;
use std::marker::PhantomData;

/// Graph Node trait
///
/// Nodes are async functions that take a state and return a state update.
/// They represent the work units in the graph.
#[async_trait]
pub trait GraphNode<S: StateSchema>: Send + Sync {
    /// Execute the node
    ///
    /// # Parameters
    /// - `state`: Current state of the graph
    /// - `config`: Optional configuration for this execution
    ///
    /// # Returns
    /// A state update that will be merged into the current state
    async fn execute(
        &self,
        state: &S,
        config: Option<NodeConfig>,
    ) -> Result<StateUpdate<S>, GraphError>;
    
    /// Get node name
    fn name(&self) -> &str;
}

/// Node configuration
#[derive(Debug, Clone, Default)]
pub struct NodeConfig {
    /// Maximum recursion depth
    pub recursion_limit: usize,
    
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    
    /// Enable debug tracing
    pub debug: bool,
}

/// Node execution result (alias for clarity)
pub type NodeResult<S> = Result<StateUpdate<S>, GraphError>;

/// Async node function type (boxed future)
pub type AsyncNodeFn<S> = Box<dyn Fn(&S) -> Pin<Box<dyn Future<Output = NodeResult<S>> + Send>> + Send + Sync>;

/// AsyncFn trait for simpler async node creation
pub trait AsyncFn<S: StateSchema>: Send + Sync {
    fn call(&self, state: &S) -> Pin<Box<dyn Future<Output = NodeResult<S>> + Send>>;
}

impl<S: StateSchema, F, Fut> AsyncFn<S> for F
where
    F: Fn(&S) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = NodeResult<S>> + Send + 'static,
{
    fn call(&self, state: &S) -> Pin<Box<dyn Future<Output = NodeResult<S>> + Send>> {
        Box::pin((self)(state))
    }
}

/// AsyncNode - Simple async node wrapper
pub struct AsyncNode<S: StateSchema, F: AsyncFn<S>> {
    name: String,
    func: F,
    _marker: PhantomData<S>,
}

impl<S: StateSchema, F: AsyncFn<S>> AsyncNode<S, F> {
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S: StateSchema, F: AsyncFn<S>> GraphNode<S> for AsyncNode<S, F> {
    async fn execute(&self, state: &S, _config: Option<NodeConfig>) -> NodeResult<S> {
        self.func.call(state).await
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Function-based node implementation
///
/// Wraps an async function as a GraphNode.
pub struct FunctionNode<S: StateSchema, F> {
    name: String,
    func: F,
    _marker: PhantomData<S>,
}

impl<S: StateSchema, F> FunctionNode<S, F>
where
    F: Fn(&S) -> Pin<Box<dyn Future<Output = Result<StateUpdate<S>, GraphError>> + Send>> + Send + Sync,
{
    /// Create a new function node
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S: StateSchema, F> GraphNode<S> for FunctionNode<S, F>
where
    F: Fn(&S) -> Pin<Box<dyn Future<Output = Result<StateUpdate<S>, GraphError>> + Send>> + Send + Sync,
{
    async fn execute(
        &self,
        state: &S,
        _config: Option<NodeConfig>,
    ) -> Result<StateUpdate<S>, GraphError> {
        (self.func)(state).await
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Simple sync node wrapper
///
/// For nodes that don't need async execution.
pub struct SyncNode<S: StateSchema, F> {
    name: String,
    func: F,
    _marker: PhantomData<S>,
}

impl<S: StateSchema, F> SyncNode<S, F>
where
    F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync,
{
    /// Create a new sync node
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<S: StateSchema, F> GraphNode<S> for SyncNode<S, F>
where
    F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync,
{
    async fn execute(
        &self,
        state: &S,
        _config: Option<NodeConfig>,
    ) -> Result<StateUpdate<S>, GraphError> {
        (self.func)(state)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Placeholder node for entry/exit points
pub struct SentinelNode {
    name: String,
}

impl SentinelNode {
    pub fn start() -> Self {
        Self { name: super::START.to_string() }
    }
    
    pub fn end() -> Self {
        Self { name: super::END.to_string() }
    }
    
    pub fn custom(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl<S: StateSchema> GraphNode<S> for SentinelNode {
    async fn execute(
        &self,
        _state: &S,
        _config: Option<NodeConfig>,
    ) -> Result<StateUpdate<S>, GraphError> {
        // Sentinel nodes don't modify state
        Ok(StateUpdate::unchanged())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::AgentState;
    
    #[tokio::test]
    async fn test_sync_node() {
        let node = SyncNode::new("test", |state: &AgentState| {
            Ok(StateUpdate::full(AgentState::new(state.input.clone())))
        });
        
        let state = AgentState::new("Hello".to_string());
        let result = node.execute(&state, None).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_sentinel_nodes() {
        let start: SentinelNode = SentinelNode::start();
        assert_eq!(GraphNode::<AgentState>::name(&start), super::super::START);
        
        let end: SentinelNode = SentinelNode::end();
        assert_eq!(GraphNode::<AgentState>::name(&end), super::super::END);
    }
}