// crates/lc-langgraph/src/graph.rs
//! StateGraph - Main graph class for LangGraph

use crate::compiled::CompiledGraph;
use crate::edge::{ConditionalEdge, GraphEdge};
use crate::errors::{GraphError, GraphResult};
use crate::node::{AsyncFn, AsyncNode, GraphNode, SyncNode};
use crate::state::{MergeReducer, Reducer, ReplaceReducer, StateSchema, StateUpdate};
use crate::subgraph::SubgraphNode;
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
    /// Create a new empty state graph.
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

    /// Add a node to the graph.
    pub fn add_node<N: GraphNode<S> + 'static>(&mut self, node: N) -> &mut Self {
        let name = node.name().to_string();
        self.nodes.insert(name, Arc::new(node));
        self
    }

    /// Add a synchronous function node to the graph.
    pub fn add_node_fn<F>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync + 'static,
    {
        let node = SyncNode::new(name, func);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    /// Add an async function node to the graph.
    pub fn add_async_node<F>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: AsyncFn<S> + 'static,
    {
        let node = AsyncNode::new(name, func);
        let node_name = node.name().to_string();
        self.nodes.insert(node_name, Arc::new(node));
        self
    }

    /// Add a subgraph node with input/output mappers.
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

    /// Add a subgraph node that shares the same state type.
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

    /// Add a fixed edge between two nodes.
    pub fn add_edge(&mut self, source: impl Into<String>, target: impl Into<String>) -> &mut Self {
        let edge = GraphEdge::fixed(source, target);
        self.edges.push(edge);
        self
    }

    /// Add a conditional edge routed by a named router.
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

    /// Add a FanOut edge for parallel execution.
    pub fn add_fan_out(&mut self, source: impl Into<String>, targets: Vec<String>) -> &mut Self {
        let edge = GraphEdge::fan_out(source, targets);
        self.edges.push(edge);
        self
    }

    /// Add a FanIn edge joining multiple sources into one target.
    pub fn add_fan_in(&mut self, sources: Vec<String>, target: impl Into<String>) -> &mut Self {
        let edge = GraphEdge::fan_in(sources, target);
        self.edges.push(edge);
        self
    }

    /// Register a conditional routing function by name.
    pub fn set_conditional_router<R: ConditionalEdge<S> + 'static>(
        &mut self,
        name: impl Into<String>,
        router: R,
    ) -> &mut Self {
        self.conditional_routers
            .insert(name.into(), Arc::new(router));
        self
    }

    /// Set the entry point node for the graph.
    pub fn set_entry_point(&mut self, node: impl Into<String>) -> &mut Self {
        self.entry_point = Some(node.into());
        self
    }

    /// Register a reducer for a specific state field.
    pub fn set_reducer(
        &mut self,
        field: impl Into<String>,
        reducer: Arc<dyn Reducer<S>>,
    ) -> &mut Self {
        self.reducers.insert(field.into(), reducer);
        self
    }

    /// Compile the graph into an executable `CompiledGraph`.
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

        // 0.22.0 C3 fix: honor the per-field reducers registered via
        // `set_reducer`. Previously they were dropped here and every merge
        // point used plain replace, silently overwriting accumulating fields.
        let merge_reducer: Arc<dyn Reducer<S>> = if self.reducers.is_empty() {
            self.default_reducer.clone()
        } else {
            Arc::new(MergeReducer::new(
                self.default_reducer.clone(),
                self.reducers.values().cloned().collect(),
            ))
        };

        let mut compiled = CompiledGraph::new(
            self.nodes.clone(),
            self.edges.clone(),
            entry,
            merge_reducer,
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
    /// Create a new empty graph builder.
    pub fn new() -> Self {
        Self {
            graph: StateGraph::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node<N: GraphNode<S> + 'static>(mut self, node: N) -> Self {
        self.graph.add_node(node);
        self
    }

    /// Add a synchronous function node to the graph.
    pub fn add_node_fn<F>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&S) -> Result<StateUpdate<S>, GraphError> + Send + Sync + 'static,
    {
        self.graph.add_node_fn(name, func);
        self
    }

    /// Add an async function node to the graph.
    pub fn add_async_node<F>(mut self, name: impl Into<String>, func: F) -> Self
    where
        F: AsyncFn<S> + 'static,
    {
        self.graph.add_async_node(name, func);
        self
    }

    /// Add a subgraph node with input/output mappers.
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

    /// Add a subgraph node that shares the same state type.
    pub fn add_subgraph_same_state(
        mut self,
        name: impl Into<String>,
        subgraph: CompiledGraph<S>,
    ) -> Self {
        self.graph.add_subgraph_same_state(name, subgraph);
        self
    }

    /// Add a fixed edge between two nodes.
    pub fn add_edge(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.graph.add_edge(source, target);
        self
    }

    /// Add a conditional edge routed by a named router.
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

    /// Add a FanOut edge for parallel execution.
    pub fn add_fan_out(mut self, source: impl Into<String>, targets: Vec<String>) -> Self {
        self.graph.add_fan_out(source, targets);
        self
    }

    /// Add a FanIn edge joining multiple sources into one target.
    pub fn add_fan_in(mut self, sources: Vec<String>, target: impl Into<String>) -> Self {
        self.graph.add_fan_in(sources, target);
        self
    }

    /// Set the entry point node for the graph.
    pub fn set_entry_point(mut self, node: impl Into<String>) -> Self {
        self.graph.set_entry_point(node);
        self
    }

    /// Compile the graph into an executable `CompiledGraph`.
    pub fn compile(self) -> GraphResult<CompiledGraph<S>> {
        self.graph.compile()
    }

    /// Build and return the underlying `StateGraph`.
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
    use super::*;
    use crate::state::AgentState;

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

    // ------------------------------------------------------------------
    // 0.22.0 C3 fixes: field reducers honored, merge on main-path base,
    // stream fan-out runs all branches, FanIn topology compiles (H-A10).
    // ------------------------------------------------------------------

    use crate::compiled::types::StreamEvent;
    use crate::state::{AppendMessagesReducer, MessageEntry, MessageRole};
    use futures_util::StreamExt;

    /// C3-1: a field reducer registered via `set_reducer` merges its field
    /// into the state instead of being dropped (previously the registered
    /// reducer never reached `CompiledGraph` and every merge was replace).
    #[tokio::test]
    async fn test_field_reducer_accumulates_across_nodes() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_node_fn("a", |_state: &AgentState| {
            // Partial update: only this node's message (drop the rest).
            Ok(StateUpdate {
                update: Some(AgentState {
                    input: String::new(),
                    messages: vec![MessageEntry {
                        role: MessageRole::AI,
                        content: "from-a".into(),
                    }],
                    steps: vec![],
                    output: None,
                }),
                metadata: Default::default(),
            })
        });
        graph.add_node_fn("b", |_state: &AgentState| {
            Ok(StateUpdate {
                update: Some(AgentState {
                    input: String::new(),
                    messages: vec![MessageEntry {
                        role: MessageRole::AI,
                        content: "from-b".into(),
                    }],
                    steps: vec![],
                    output: None,
                }),
                metadata: Default::default(),
            })
        });
        graph.set_entry_point("a");
        graph.add_edge("a", "b");
        graph.add_edge("b", END);
        graph.set_reducer("messages", Arc::new(AppendMessagesReducer));

        let compiled = graph.compile().unwrap();
        let mut initial = AgentState::new("start"); // messages = [human "start"]
        initial.messages.clear();
        let result = compiled.invoke(initial).await.unwrap();
        let contents: Vec<&str> = result
            .final_state
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        // Both node messages survived the merges (replace would keep only b's).
        assert!(
            contents.contains(&"from-a") && contents.contains(&"from-b"),
            "field reducer should accumulate, got {contents:?}"
        );
    }

    /// C3-2: `merge_parallel_states` folds on top of the pre-fan-out
    /// main-path state — the fan-out source node's writes are not dropped.
    #[tokio::test]
    async fn test_parallel_merge_keeps_main_path_state() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_node_fn("source", |state: &AgentState| {
            let mut s = state.clone();
            s.messages.push(MessageEntry {
                role: MessageRole::AI,
                content: "main-path".into(),
            });
            Ok(StateUpdate::full(s))
        });
        graph.add_node_fn("branch1", |state: &AgentState| {
            let mut s = state.clone();
            s.messages.push(MessageEntry {
                role: MessageRole::AI,
                content: "b1".into(),
            });
            Ok(StateUpdate::full(s))
        });
        graph.add_node_fn("branch2", |state: &AgentState| {
            let mut s = state.clone();
            s.messages.push(MessageEntry {
                role: MessageRole::AI,
                content: "b2".into(),
            });
            Ok(StateUpdate::full(s))
        });
        graph.add_node_fn("merge", |state: &AgentState| {
            let mut s = state.clone();
            s.set_output("merged");
            Ok(StateUpdate::full(s))
        });
        graph.set_entry_point("source");
        graph.add_fan_out("source", vec!["branch1".into(), "branch2".into()]);
        graph.add_fan_in(vec!["branch1".into(), "branch2".into()], "merge");
        graph.add_edge("merge", END);
        graph.set_reducer("messages", Arc::new(AppendMessagesReducer));

        let compiled = graph.compile().unwrap();
        let mut initial = AgentState::new("start");
        initial.messages.clear();
        let result = compiled.invoke(initial).await.unwrap();
        let contents: Vec<&str> = result
            .final_state
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        // All three writers survive: source (main path) + both branches.
        assert!(
            contents.contains(&"main-path")
                && contents.contains(&"b1")
                && contents.contains(&"b2"),
            "merge must keep main-path + all branch writes, got {contents:?}"
        );
        assert_eq!(result.final_state.output.as_deref(), Some("merged"));
    }

    /// C3-3: `CompiledGraph::stream` executes ALL fan-out branches (it used
    /// to run only `targets[0]` and silently drop the rest).
    #[tokio::test]
    async fn test_stream_fan_out_runs_all_branches() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_node_fn("source", |state: &AgentState| {
            Ok(StateUpdate::full(state.clone()))
        });
        graph.add_node_fn("branch1", |state: &AgentState| {
            let mut s = state.clone();
            s.messages.push(MessageEntry {
                role: MessageRole::AI,
                content: "s-b1".into(),
            });
            Ok(StateUpdate::full(s))
        });
        graph.add_node_fn("branch2", |state: &AgentState| {
            let mut s = state.clone();
            s.messages.push(MessageEntry {
                role: MessageRole::AI,
                content: "s-b2".into(),
            });
            Ok(StateUpdate::full(s))
        });
        graph.set_entry_point("source");
        graph.add_fan_out("source", vec!["branch1".into(), "branch2".into()]);
        graph.add_fan_in(vec!["branch1".into(), "branch2".into()], END);
        graph.set_reducer("messages", Arc::new(AppendMessagesReducer));

        let compiled = graph.compile().unwrap();
        let mut initial = AgentState::new("start");
        initial.messages.clear();
        let events = compiled.stream_collected(initial).await.unwrap();
        let end = events
            .into_iter()
            .find_map(|e| match e {
                StreamEvent::End(s) => Some(s),
                _ => None,
            })
            .expect("stream should end");
        let contents: Vec<&str> = end
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect();
        assert!(
            contents.contains(&"s-b1") && contents.contains(&"s-b2"),
            "stream must execute all fan-out branches, got {contents:?}"
        );
    }

    /// H-A10: a standard fan-out + fan-in topology compiles — FanIn edges
    /// were previously invisible to reachability (`source()` returned a
    /// constant), so `compile()` rejected the graph with "Unreachable node".
    #[test]
    fn test_fan_in_topology_compiles() {
        let mut graph: StateGraph<AgentState> = StateGraph::new();
        graph.add_node_fn("source", |state: &AgentState| {
            Ok(StateUpdate::full(state.clone()))
        });
        graph.add_node_fn("b1", |state: &AgentState| Ok(StateUpdate::full(state.clone())));
        graph.add_node_fn("b2", |state: &AgentState| Ok(StateUpdate::full(state.clone())));
        graph.add_node_fn("merge", |state: &AgentState| Ok(StateUpdate::full(state.clone())));
        graph.set_entry_point("source");
        graph.add_fan_out("source", vec!["b1".into(), "b2".into()]);
        graph.add_fan_in(vec!["b1".into(), "b2".into()], "merge");
        graph.add_edge("merge", END);
        assert!(
            graph.compile().is_ok(),
            "fan-out + fan-in topology must compile"
        );
    }
}
