// src/langgraph/subgraph.rs
//! Subgraph Support for LangGraph
//!
//! Subgraphs allow nesting compiled graphs within nodes of a parent graph.
//! This enables composition of complex workflows from simpler components.
//!
//! # Example
//!
//! ```rust,ignore
//! use langchainrust::{StateGraph, CompiledGraph, SubgraphNode, START, END};
//!
//! // Create subgraph
//! let subgraph = GraphBuilder::<AgentState>::new()
//!     .add_node_fn("process", |state| Ok(StateUpdate::full(state.clone())))
//!     .add_node_fn("output", |state| {
//!         let mut s = state.clone();
//!         s.set_output("done");
//!         Ok(StateUpdate::full(s))
//!     })
//!     .add_edge(START, "process")
//!     .add_edge("process", "output")
//!     .add_edge("output", END)
//!     .compile()?;
//!
//! // Add as subgraph node in parent graph
//! let parent = GraphBuilder::<AgentState>::new()
//!     .add_subgraph("subworkflow", subgraph, 
//!         |parent_state| parent_state.clone(),  // input mapper
//!         |sub_state, parent_state| *parent_state = sub_state.clone()  // output mapper
//!     )
//!     .add_edge(START, "subworkflow")
//!     .add_edge("subworkflow", END)
//!     .compile()?;
//! ```

use async_trait::async_trait;
use std::sync::Arc;
use std::marker::PhantomData;
use super::state::{StateSchema, StateUpdate};
use super::node::{GraphNode, NodeConfig, NodeResult};
use super::compiled::CompiledGraph;
use super::errors::{GraphError, GraphResult};

/// Subgraph Node - A node that executes a nested compiled graph
///
/// This allows composition of graphs by embedding one graph inside another.
/// The subgraph receives mapped input state and returns mapped output state.
pub struct SubgraphNode<S: StateSchema, SubS: StateSchema> {
    name: String,
    subgraph: CompiledGraph<SubS>,
    input_mapper: Arc<dyn Fn(&S) -> SubS + Send + Sync>,
    output_mapper: Arc<dyn Fn(&SubS, &mut S) + Send + Sync>,
    _parent_marker: PhantomData<S>,
    _sub_marker: PhantomData<SubS>,
}

impl<S: StateSchema, SubS: StateSchema> SubgraphNode<S, SubS> {
    pub fn new(
        name: impl Into<String>,
        subgraph: CompiledGraph<SubS>,
        input_mapper: impl Fn(&S) -> SubS + Send + Sync + 'static,
        output_mapper: impl Fn(&SubS, &mut S) + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            subgraph,
            input_mapper: Arc::new(input_mapper),
            output_mapper: Arc::new(output_mapper),
            _parent_marker: PhantomData,
            _sub_marker: PhantomData,
        }
    }
}

impl<S: StateSchema + Clone> SubgraphNode<S, S> {
    pub fn same_state(
        name: impl Into<String>,
        subgraph: CompiledGraph<S>,
    ) -> Self {
        Self::new(
            name,
            subgraph,
            |s| s.clone(),
            |sub_s, parent_s| *parent_s = sub_s.clone(),
        )
    }
}

#[async_trait]
impl<S: StateSchema + 'static, SubS: StateSchema + 'static> GraphNode<S> for SubgraphNode<S, SubS> {
    async fn execute(&self, state: &S, _config: Option<NodeConfig>) -> NodeResult<S> {
        // Map parent state to subgraph input
        let sub_input = (self.input_mapper)(state);
        
        // Execute subgraph
        let sub_result = self.subgraph.invoke(sub_input).await
            .map_err(|e| GraphError::ExecutionError(
                format!("Subgraph '{}' execution failed: {}", self.name, e)
            ))?;
        
        // Map subgraph output back to parent state
        let mut parent_output = state.clone();
        (self.output_mapper)(&sub_result.final_state, &mut parent_output);
        
        // Include subgraph steps in metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "subgraph_steps".to_string(),
            serde_json::json!(sub_result.steps.len()),
        );
        metadata.insert(
            "subgraph_recursion".to_string(),
            serde_json::json!(sub_result.recursion_count),
        );
        
        Ok(StateUpdate::with_metadata(parent_output, metadata))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Builder for creating subgraph nodes with fluent API
pub struct SubgraphBuilder<S: StateSchema, SubS: StateSchema> {
    name: String,
    subgraph: Option<CompiledGraph<SubS>>,
    input_mapper: Option<Arc<dyn Fn(&S) -> SubS + Send + Sync>>,
    output_mapper: Option<Arc<dyn Fn(&SubS, &mut S) + Send + Sync>>,
    _parent_marker: PhantomData<S>,
    _sub_marker: PhantomData<SubS>,
}

impl<S: StateSchema, SubS: StateSchema> SubgraphBuilder<S, SubS> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subgraph: None,
            input_mapper: None,
            output_mapper: None,
            _parent_marker: PhantomData,
            _sub_marker: PhantomData,
        }
    }
    
    pub fn subgraph(mut self, graph: CompiledGraph<SubS>) -> Self {
        self.subgraph = Some(graph);
        self
    }
    
    pub fn input_mapper(mut self, mapper: impl Fn(&S) -> SubS + Send + Sync + 'static) -> Self {
        self.input_mapper = Some(Arc::new(mapper));
        self
    }
    
    pub fn output_mapper(mut self, mapper: impl Fn(&SubS, &mut S) + Send + Sync + 'static) -> Self {
        self.output_mapper = Some(Arc::new(mapper));
        self
    }
    
    pub fn build(self) -> GraphResult<SubgraphNode<S, SubS>> {
        let subgraph = self.subgraph.ok_or_else(|| 
            GraphError::ValidationError("Subgraph not set".to_string())
        )?;
        let input_mapper = self.input_mapper.ok_or_else(|| 
            GraphError::ValidationError("Input mapper not set".to_string())
        )?;
        let output_mapper = self.output_mapper.ok_or_else(|| 
            GraphError::ValidationError("Output mapper not set".to_string())
        )?;
        
        Ok(SubgraphNode {
            name: self.name,
            subgraph,
            input_mapper,
            output_mapper,
            _parent_marker: PhantomData,
            _sub_marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::state::AgentState;
    use super::super::graph::GraphBuilder;
    use super::super::{START, END};
    
    #[tokio::test]
    async fn test_subgraph_same_state() {
        // Create simple subgraph
        let subgraph = GraphBuilder::<AgentState>::new()
            .add_node_fn("sub_process", |state| {
                let mut s = state.clone();
                s.set_output("subgraph_output".to_string());
                Ok(StateUpdate::full(s))
            })
            .add_edge(START, "sub_process")
            .add_edge("sub_process", END)
            .compile()
            .unwrap();
        
        // Create parent graph with subgraph
        let parent = GraphBuilder::<AgentState>::new()
            .add_subgraph_same_state("subworkflow", subgraph)
            .add_edge(START, "subworkflow")
            .add_edge("subworkflow", END)
            .compile()
            .unwrap();
        
        let input = AgentState::new("test".to_string());
        let result = parent.invoke(input).await.unwrap();
        
        assert!(result.final_state.output.is_some());
        assert_eq!(result.final_state.output.unwrap(), "subgraph_output");
    }
    
    #[tokio::test]
    async fn test_nested_subgraphs() {
        // Create innermost subgraph
        let inner = GraphBuilder::<AgentState>::new()
            .add_node_fn("inner_node", |state| {
                let mut s = state.clone();
                s.input = format!("inner:{}", s.input);
                Ok(StateUpdate::full(s))
            })
            .add_edge(START, "inner_node")
            .add_edge("inner_node", END)
            .compile()
            .unwrap();
        
        // Create middle subgraph containing inner
        let middle = GraphBuilder::<AgentState>::new()
            .add_subgraph_same_state("inner_workflow", inner)
            .add_node_fn("middle_node", |state| {
                let mut s = state.clone();
                s.input = format!("middle:{}", s.input);
                Ok(StateUpdate::full(s))
            })
            .add_edge(START, "inner_workflow")
            .add_edge("inner_workflow", "middle_node")
            .add_edge("middle_node", END)
            .compile()
            .unwrap();
        
        // Create outer graph with middle subgraph
        let outer = GraphBuilder::<AgentState>::new()
            .add_node_fn("outer_node", |state| {
                let mut s = state.clone();
                s.input = format!("outer:{}", s.input);
                Ok(StateUpdate::full(s))
            })
            .add_subgraph_same_state("middle_workflow", middle)
            .add_edge(START, "outer_node")
            .add_edge("outer_node", "middle_workflow")
            .add_edge("middle_workflow", END)
            .compile()
            .unwrap();
        
        let input = AgentState::new("test".to_string());
        let result = outer.invoke(input).await.unwrap();
        
        // Verify nested execution: outer:middle:inner:test
        assert!(result.final_state.input.contains("outer"));
        assert!(result.final_state.input.contains("middle"));
        assert!(result.final_state.input.contains("inner"));
    }
}