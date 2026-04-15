// src/langgraph/compiled.rs
//! CompiledGraph - Executable graph with state management

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use super::state::{StateSchema, StateUpdate, Reducer};
use super::node::{GraphNode, NodeConfig};
use super::edge::{GraphEdge, ConditionalEdge};
use super::errors::{GraphError, GraphResult};
use super::checkpointer::{Checkpointer};
use super::{START, END};
use async_trait::async_trait;
use serde_json::Value as JsonValue;

/// CompiledGraph - Ready-to-execute graph
///
/// Created from StateGraph.compile(). Handles:
/// - State management and updates
/// - Edge routing (fixed and conditional)
/// - Execution with recursion limits
/// - Checkpointing for persistence
pub struct CompiledGraph<S: StateSchema> {
    nodes: HashMap<String, Arc<dyn GraphNode<S>>>,
    edges: Vec<GraphEdge>,
    entry_point: String,
    default_reducer: Arc<dyn Reducer<S>>,
    conditional_routers: HashMap<String, Arc<dyn ConditionalEdge<S>>>,
    checkpointer: Option<Arc<Mutex<dyn Checkpointer<S> + Send>>>,
    recursion_limit: usize,
}

impl<S: StateSchema> CompiledGraph<S> {
    pub(crate) fn new(
        nodes: HashMap<String, Arc<dyn GraphNode<S>>>,
        edges: Vec<GraphEdge>,
        entry_point: String,
        default_reducer: Arc<dyn Reducer<S>>,
    ) -> Self {
        Self {
            nodes,
            edges,
            entry_point,
            default_reducer,
            conditional_routers: HashMap::new(),
            checkpointer: None,
            recursion_limit: 25,
        }
    }
    
    pub(crate) fn add_router(&mut self, name: String, router: Arc<dyn ConditionalEdge<S>>) {
        self.conditional_routers.insert(name, router);
    }
    
    pub fn with_checkpointer<C: Checkpointer<S> + 'static>(mut self, checkpointer: C) -> Self {
        self.checkpointer = Some(Arc::new(Mutex::new(checkpointer)));
        self
    }
    
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }
    
    pub fn validate(&self) -> GraphResult<()> {
        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(
                            format!("Source node '{}' not found", source)
                        ));
                    }
                    if target != END && !self.nodes.contains_key(target) {
                        return Err(GraphError::ValidationError(
                            format!("Target node '{}' not found", target)
                        ));
                    }
                }
                GraphEdge::Conditional { source, router_name, .. } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(
                            format!("Source node '{}' not found", source)
                        ));
                    }
                    if !self.conditional_routers.contains_key(router_name) {
                        return Err(GraphError::ValidationError(
                            format!("Router '{}' not found", router_name)
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    
    pub async fn invoke(&self, input: S) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;
        
        if let Some(ref checkpointer) = self.checkpointer {
            let checkpoint_id = checkpointer.lock().await.save(&state).await?;
            steps.push(ExecutionStep::checkpoint(checkpoint_id, current_node.clone()));
        }
        
        while current_node != END && recursion_count < self.recursion_limit {
            recursion_count += 1;
            
            let node = self.nodes.get(&current_node)
                .ok_or_else(|| GraphError::ExecutionError(
                    format!("Node '{}' not found", current_node)
                ))?;
            
            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };
            
            let update = node.execute(&state, Some(config)).await?;
            
            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
            }
            
            steps.push(ExecutionStep::node(current_node.clone(), update.metadata.clone()));
            
            let next_node = self.find_next_node(&current_node, &state).await?;
            
            if let Some(ref checkpointer) = self.checkpointer {
                let checkpoint_id = checkpointer.lock().await.save(&state).await?;
                steps.push(ExecutionStep::checkpoint(checkpoint_id, next_node.clone()));
            }
            
            current_node = next_node;
        }
        
        if recursion_count >= self.recursion_limit {
            return Err(GraphError::RecursionLimitReached(self.recursion_limit));
        }
        
        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }
    
    pub async fn stream(&self, input: S) -> GraphResult<Vec<StreamEvent<S>>> {
        let mut events = Vec::new();
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut recursion_count = 0;
        
        events.push(StreamEvent::start(state.clone()));
        
        while current_node != END && recursion_count < self.recursion_limit {
            recursion_count += 1;
            
            events.push(StreamEvent::enter_node(current_node.clone(), state.clone()));
            
            let node = self.nodes.get(&current_node)
                .ok_or_else(|| GraphError::ExecutionError(
                    format!("Node '{}' not found", current_node)
                ))?;
            
            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };
            
            let update = node.execute(&state, Some(config)).await?;
            
            events.push(StreamEvent::node_complete(current_node.clone(), update.clone()));
            
            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
                events.push(StreamEvent::state_update(state.clone()));
            }
            
            let next_node = self.find_next_node(&current_node, &state).await?;
            current_node = next_node;
        }
        
        events.push(StreamEvent::end(state.clone()));
        Ok(events)
    }
    
    async fn find_next_node(&self, current: &str, state: &S) -> GraphResult<String> {
        for edge in &self.edges {
            if edge.source() == current {
                match edge {
                    GraphEdge::Fixed { target, .. } => {
                        return Ok(target.clone());
                    }
                    GraphEdge::Conditional { router_name, targets, default_target, .. } => {
                        let router = self.conditional_routers.get(router_name)
                            .ok_or_else(|| GraphError::ExecutionError(
                                format!("Router '{}' not found", router_name)
                            ))?;
                        
                        let route_key = router.route(state).await?;
                        
                        let target = targets.get(&route_key)
                            .or_else(|| default_target.as_ref())
                            .ok_or_else(|| GraphError::RoutingError(
                                format!("No target for route '{}'", route_key)
                            ))?;
                        
                        return Ok(target.clone());
                    }
                }
            }
        }
        
        if current == self.entry_point && self.nodes.len() == 1 {
            return Ok(END.to_string());
        }
        
        Err(GraphError::RoutingError(
            format!("No outgoing edge from node '{}'", current)
        ))
    }
}

/// GraphInvocation - Result of graph execution
pub struct GraphInvocation<S: StateSchema> {
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
}

impl<S: StateSchema> GraphInvocation<S> {
    pub fn state(&self) -> &S {
        &self.final_state
    }
    
    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }
}

/// ExecutionStep - Single step in execution history
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    Node { name: String, metadata: HashMap<String, JsonValue> },
    Checkpoint { id: String, next_node: String },
}

impl ExecutionStep {
    pub fn node(name: String, metadata: HashMap<String, JsonValue>) -> Self {
        Self::Node { name, metadata }
    }
    
    pub fn checkpoint(id: String, next_node: String) -> Self {
        Self::Checkpoint { id, next_node }
    }
}

/// StreamEvent - Event for streaming execution
#[derive(Debug, Clone)]
pub enum StreamEvent<S: StateSchema> {
    Start(S),
    EnterNode(String, S),
    NodeComplete(String, StateUpdate<S>),
    StateUpdate(S),
    End(S),
}

impl<S: StateSchema> StreamEvent<S> {
    pub fn start(state: S) -> Self {
        Self::Start(state)
    }
    
    pub fn enter_node(name: String, state: S) -> Self {
        Self::EnterNode(name, state)
    }
    
    pub fn node_complete(name: String, update: StateUpdate<S>) -> Self {
        Self::NodeComplete(name, update)
    }
    
    pub fn state_update(state: S) -> Self {
        Self::StateUpdate(state)
    }
    
    pub fn end(state: S) -> Self {
        Self::End(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::AgentState;
    use super::super::graph::GraphBuilder;
    
    #[tokio::test]
    async fn test_simple_linear_graph() {
        let compiled = GraphBuilder::<AgentState>::new()
            .add_node_fn("step1", |state| {
                Ok(StateUpdate::full(AgentState::new(state.input.clone())))
            })
            .add_node_fn("step2", |state| {
                let mut new_state = state.clone();
                new_state.set_output("done".to_string());
                Ok(StateUpdate::full(new_state))
            })
            .add_edge(START, "step1")
            .add_edge("step1", "step2")
            .add_edge("step2", END)
            .compile()
            .unwrap();
        
        let input = AgentState::new("test input".to_string());
        let result = compiled.invoke(input).await.unwrap();
        
        assert!(result.final_state.output.is_some());
        assert_eq!(result.recursion_count, 2);
    }
    
    #[tokio::test]
    async fn test_stream_execution() {
        let compiled = GraphBuilder::<AgentState>::new()
            .add_node_fn("process", |state| Ok(StateUpdate::full(state.clone())))
            .add_edge(START, "process")
            .add_edge("process", END)
            .compile()
            .unwrap();
        
        let input = AgentState::new("test".to_string());
        let events = compiled.stream(input).await.unwrap();
        
        assert!(!events.is_empty());
    }
}