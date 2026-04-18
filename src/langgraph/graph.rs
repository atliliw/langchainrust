// src/langgraph/graph.rs
//! StateGraph - Main graph class for LangGraph

use super::compiled::CompiledGraph;
use super::edge::{ConditionalEdge, GraphEdge};
use super::errors::{GraphError, GraphResult};
use super::node::{AsyncFn, AsyncNode, GraphNode, SyncNode};
use super::state::{Reducer, ReplaceReducer, StateSchema, StateUpdate};
use super::subgraph::SubgraphNode;
use std::collections::HashMap;
use std::sync::Arc;

/// START sentinel node identifier
pub const START: &str = "__start__";
/// END sentinel node identifier
pub const END: &str = "__end__";

/// StateGraph - Main graph builder
///
/// Manages nodes, edges, and state schema for graph execution.
/// After building, compile to get an executable CompiledGraph.
pub struct StateGraph<S: StateSchema> {
    nodes: HashMap<String, Arc<dyn GraphNode<S>>>,
    edges: Vec<GraphEdge>,
    entry_point: Option<String>,
    reducers: HashMap<String, Arc<dyn Reducer<S>>>,
    default_reducer: Arc<dyn Reducer<S>>,
    conditional_routers: HashMap<String, Arc<dyn ConditionalEdge<S>>>,
}

impl<S: StateSchema + 'static> StateGraph<S> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: None,
            reducers: HashMap::new(),
            default_reducer: Arc::new(ReplaceReducer),
            conditional_routers: HashMap::new(),
        }
    }

    pub fn add_node<N: GraphNode<S> + 'static>(&mut self, node: N) -> &mut Self {
        let name = node.name().to_string();
        self.nodes.insert(name, Arc::new(node));
        self
    }

    pub fn add_node_fn<F>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync + 'static,
    {
        let node = SyncNode::new(name, func);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    pub fn add_async_node<F>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: AsyncFn<S> + 'static,
    {
        let node = AsyncNode::new(name, func);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    pub fn add_subgraph<SubS: StateSchema + 'static>(
        &mut self,
        name: impl Into<String>,
        subgraph: CompiledGraph<SubS>,
        input_mapper: impl Fn(&S) -> SubS + Send + Sync + 'static,
        output_mapper: impl Fn(&SubS, &mut S) + Send + Sync + 'static,
    ) -> &mut Self {
        let node = SubgraphNode::new(name, subgraph, input_mapper, output_mapper);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    pub fn add_subgraph_same_state(
        &mut self,
        name: impl Into<String>,
        subgraph: CompiledGraph<S>,
    ) -> &mut Self {
        let node: SubgraphNode<S, S> = SubgraphNode::same_state(name, subgraph);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    pub fn add_edge(&mut self, source: impl Into<String>, target: impl Into<String>) -> &mut Self {
        let edge = GraphEdge::fixed(source, target);
        self.edges.push(edge);
        self
    }

    pub fn add_conditional_edges(
        &mut self,
        source: impl Into<String>,
        router_name: impl Into<String>,
        targets: HashMap<String, String>,
        default: Option<String>,
    ) -> &mut Self {
        let edge = GraphEdge::conditional(source, router_name, targets, default);
        self.edges.push(edge);
        self
    }

    pub fn add_fan_out(&mut self, source: impl Into<String>, targets: Vec<String>) -> &mut Self {
        let edge = GraphEdge::fan_out(source, targets);
        self.edges.push(edge);
        self
    }

    pub fn add_fan_in(&mut self, sources: Vec<String>, target: impl Into<String>) -> &mut Self {
        let edge = GraphEdge::fan_in(sources, target);
        self.edges.push(edge);
        self
    }

    pub fn set_conditional_router<R: ConditionalEdge<S> + 'static>(
        &mut self,
        name: impl Into<String>,
        router: R,
    ) -> &mut Self {
        self.conditional_routers
            .insert(name.into(), Arc::new(router));
        self
    }

    pub fn set_entry_point(&mut self, node: impl Into<String>) -> &mut Self {
        self.entry_point = Some(node.into());
        self
    }

    pub fn set_reducer(
        &mut self,
        field: impl Into<String>,
        reducer: Arc<dyn Reducer<S>>,
    ) -> &mut Self {
        self.reducers.insert(field.into(), reducer);
        self
    }

    pub fn compile(&self) -> GraphResult<CompiledGraph<S>> {
        if self.nodes.is_empty() {
            return Err(GraphError::ValidationError(
                "Graph has no nodes".to_string(),
            ));
        }

        let entry = self
            .entry_point
            .clone()
            .or_else(|| self.find_first_node_after_start())
            .ok_or_else(|| GraphError::ValidationError("No entry point defined".to_string()))?;

        if !self.nodes.contains_key(&entry) && entry != START {
            return Err(GraphError::ValidationError(format!(
                "Entry point '{}' not found",
                entry
            )));
        }

        let mut compiled = CompiledGraph::new(
            self.nodes.clone(),
            self.edges.clone(),
            entry,
            self.default_reducer.clone(),
        );

        for (name, router) in &self.conditional_routers {
            compiled.add_router(name.clone(), router.clone());
        }

        compiled.validate()?;
        Ok(compiled)
    }

    fn find_first_node_after_start(&self) -> Option<String> {
        for edge in &self.edges {
            if edge.source() == START {
                if let Some(target) = edge.fixed_target() {
                    return Some(target.to_string());
                }
            }
        }
        None
    }
}

impl<S: StateSchema + 'static> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphBuilder - Fluent builder pattern for StateGraph
pub struct GraphBuilder<S: StateSchema> {
    graph: StateGraph<S>,
}

impl<S: StateSchema + 'static> GraphBuilder<S> {
    pub fn new() -> Self {
        Self {
            graph: StateGraph::new(),
        }
    }

    pub fn add_node<N: GraphNode<S> + 'static>(mut self, node: N) -> Self {
        self.graph.add_node(node);
        self
    }

    pub fn add_node_fn<F>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync + 'static,
    {
        self.graph.add_node_fn(name, func);
        self
    }

    pub fn add_async_node<F>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: AsyncFn<S> + 'static,
    {
        self.graph.add_async_node(name, func);
        self
    }

    pub fn add_subgraph<SubS: StateSchema + 'static>(
        mut self,
        name: impl Into<String>,
        subgraph: CompiledGraph<SubS>,
        input_mapper: impl Fn(&S) -> SubS + Send + Sync + 'static,
        output_mapper: impl Fn(&SubS, &mut S) + Send + Sync + 'static,
    ) -> Self {
        self.graph
            .add_subgraph(name, subgraph, input_mapper, output_mapper);
        self
    }

    pub fn add_subgraph_same_state(
        mut self,
        name: impl Into<String>,
        subgraph: CompiledGraph<S>,
    ) -> Self {
        self.graph.add_subgraph_same_state(name, subgraph);
        self
    }

    pub fn add_edge(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.graph.add_edge(source, target);
        self
    }

    pub fn add_conditional_edges(
        mut self,
        source: impl Into<String>,
        router_name: impl Into<String>,
        targets: HashMap<String, String>,
        default: Option<String>,
    ) -> Self {
        self.graph
            .add_conditional_edges(source, router_name, targets, default);
        self
    }

    pub fn add_fan_out(mut self, source: impl Into<String>, targets: Vec<String>) -> Self {
        self.graph.add_fan_out(source, targets);
        self
    }

    pub fn add_fan_in(mut self, sources: Vec<String>, target: impl Into<String>) -> Self {
        self.graph.add_fan_in(sources, target);
        self
    }

    pub fn set_entry_point(mut self, node: impl Into<String>) -> Self {
        self.graph.set_entry_point(node);
        self
    }

    pub fn compile(self) -> GraphResult<CompiledGraph<S>> {
        self.graph.compile()
    }

    pub fn build(self) -> StateGraph<S> {
        self.graph
    }
}

impl<S: StateSchema + 'static> Default for GraphBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::state::AgentState;
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph: StateGraph<AgentState> = StateGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_add_node_fn() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_node_fn("test_node", |state: &AgentState| {
            Ok(StateUpdate::full(state.clone()))
        });
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_edge(START, "node1");
        graph.add_edge("node1", END);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_compile_empty_graph() {
        let graph: StateGraph<AgentState> = StateGraph::new();
        let result = graph.compile();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let compiled = GraphBuilder::<AgentState>::new()
            .add_node_fn("process", |state| Ok(StateUpdate::full(state.clone())))
            .add_edge(START, "process")
            .add_edge("process", END)
            .compile();
        assert!(compiled.is_ok());
    }
}
