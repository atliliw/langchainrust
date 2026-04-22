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
use super::persistence::{GraphDefinition, NodeDefinition, EdgeDefinition, NodeType};
use super::{START, END};
use serde_json::Value as JsonValue;
use futures_util::future::join_all;

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
    interrupt_before: Vec<String>,
    interrupt_after: Vec<String>,
}

impl<S: StateSchema> std::fmt::Debug for CompiledGraph<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("entry_point", &self.entry_point)
            .field("recursion_limit", &self.recursion_limit)
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
            .finish()
    }
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
            interrupt_before: Vec::new(),
            interrupt_after: Vec::new(),
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
    
    pub fn with_interrupt_before(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_before = nodes;
        self
    }
    
    pub fn with_interrupt_after(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_after = nodes;
        self
    }
    
    pub fn node_names(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }
    
    pub fn get_edges(&self) -> &[GraphEdge] {
        &self.edges
    }
    
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
    
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }
    
    pub fn interrupt_before(&self) -> &[String] {
        &self.interrupt_before
    }
    
    pub fn interrupt_after(&self) -> &[String] {
        &self.interrupt_after
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
                    if target == START {
                        return Err(GraphError::ValidationError(
                            format!("Edge cannot target START node")
                        ));
                    }
                }
GraphEdge::Conditional { source, router_name, targets, default_target } => {
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
                    for (route, target) in targets {
                        if target != END && !self.nodes.contains_key(target) {
                            return Err(GraphError::ValidationError(
                                format!("Target '{}' for route '{}' not found", target, route)
                        ));
                        }
                        if target == START {
                            return Err(GraphError::ValidationError(
                                format!("Conditional edge cannot target START node")
                            ));
                        }
                    }
                    if let Some(default) = default_target {
                        if default != END && !self.nodes.contains_key(default) {
                            return Err(GraphError::ValidationError(
                                format!("Default target '{}' not found", default)
                            ));
                        }
                    }
                }
                GraphEdge::FanOut { source, targets } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(
                            format!("FanOut source node '{}' not found", source)
                        ));
                    }
                    for target in targets {
                        if target != END && !self.nodes.contains_key(target) {
                            return Err(GraphError::ValidationError(
                                format!("FanOut target node '{}' not found", target)
                            ));
                        }
                    }
                }
                GraphEdge::FanIn { sources, target } => {
                    for source in sources {
                        if source != START && !self.nodes.contains_key(source) {
                            return Err(GraphError::ValidationError(
                                format!("FanIn source node '{}' not found", source)
                            ));
                        }
                    }
                    if target != END && !self.nodes.contains_key(target) {
                        return Err(GraphError::ValidationError(
                            format!("FanIn target node '{}' not found", target)
                        ));
                    }
                }
            }
        }
        
        self.validate_duplicate_edges()?;
        self.validate_unreachable_nodes()?;
        self.validate_cycles()?;
        
        Ok(())
    }
    
    fn validate_duplicate_edges(&self) -> GraphResult<()> {
        let mut seen_fixed: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        
        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    let key = (source.clone(), target.clone());
                    if seen_fixed.contains(&key) {
                        return Err(GraphError::DuplicateEdgeError(
                            format!("Duplicate edge: {} -> {}", source, target)
                        ));
                    }
                    seen_fixed.insert(key);
                }
                _ => {}
            }
        }
        Ok(())
    }
    
    fn validate_unreachable_nodes(&self) -> GraphResult<()> {
        let reachable = self.compute_reachable_nodes();
        
        for node_name in self.nodes.keys() {
            if !reachable.contains(node_name) {
                return Err(GraphError::OrphanNodeError(
                    format!("Unreachable node: {}", node_name)
                ));
            }
        }
        Ok(())
    }
    
    fn compute_reachable_nodes(&self) -> std::collections::HashSet<String> {
        let mut reachable: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut to_visit: Vec<String> = vec![self.entry_point.clone()];
        
        while let Some(current) = to_visit.pop() {
            if reachable.contains(&current) || current == END {
                continue;
            }
            reachable.insert(current.clone());
            
            for edge in &self.edges {
                if edge.source() == current {
                    match edge {
                        GraphEdge::Fixed { target, .. } => {
                            if !reachable.contains(target) && target != END {
                                to_visit.push(target.clone());
                            }
                        }
                        GraphEdge::Conditional { targets, default_target, .. } => {
                            for target in targets.values() {
                                if !reachable.contains(target) && target != END {
                                    to_visit.push(target.clone());
                                }
                            }
                            if let Some(default) = default_target {
                                if !reachable.contains(default) && default != END {
                                    to_visit.push(default.clone());
                                }
                            }
                        }
                        GraphEdge::FanOut { targets, .. } => {
                            for target in targets {
                                if !reachable.contains(target) && target != END {
                                    to_visit.push(target.clone());
                                }
                            }
                        }
                        GraphEdge::FanIn { sources, target } => {
                            // FanIn: if all sources are reachable, target is reachable
                            if sources.iter().all(|s| reachable.contains(s)) {
                                if !reachable.contains(target) && target != END {
                                    to_visit.push(target.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        reachable
    }
    
    fn validate_cycles(&self) -> GraphResult<()> {
        let reachable = self.compute_reachable_nodes();
        let end_reachable = self.compute_end_reachable_nodes();
        
        for node in &reachable {
            if !end_reachable.contains(node) {
                return Err(GraphError::InfiniteCycleError(
                    format!("Node '{}' in cycle with no path to END", node)
                ));
            }
        }
        Ok(())
    }
    
    fn compute_end_reachable_nodes(&self) -> std::collections::HashSet<String> {
        let mut end_reachable: std::collections::HashSet<String> = std::collections::HashSet::new();
        end_reachable.insert(END.to_string());
        
        let mut changed = true;
        while changed {
            changed = false;
            for edge in &self.edges {
                match edge {
                    GraphEdge::Fixed { source, target } => {
                        if end_reachable.contains(target) && !end_reachable.contains(source) {
                            end_reachable.insert(source.clone());
                            changed = true;
                        }
                    }
                    GraphEdge::Conditional { source, targets, default_target, .. } => {
                        let any_target_reaches_end = targets.values().any(|t| end_reachable.contains(t))
                            || default_target.as_ref().map_or(false, |d| end_reachable.contains(d));
                        if any_target_reaches_end && !end_reachable.contains(source) {
                            end_reachable.insert(source.clone());
                            changed = true;
                        }
                    }
                    GraphEdge::FanOut { source, targets } => {
                        let all_targets_reach_end = targets.iter().all(|t| end_reachable.contains(t));
                        if all_targets_reach_end && !end_reachable.contains(source) {
                            end_reachable.insert(source.clone());
                            changed = true;
                        }
                    }
                    GraphEdge::FanIn { sources, target } => {
                        if end_reachable.contains(target) {
                            for source in sources {
                                if !end_reachable.contains(source) {
                                    end_reachable.insert(source.clone());
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        end_reachable
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
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }
            
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
            
            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!("after_{}", current_node)));
            }
            
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
    
    pub async fn invoke_with_execution(&self, execution: GraphExecution<S>) -> GraphResult<GraphInvocation<S>> {
        let mut state = execution.state;
        let mut current_node = if execution.interrupted_at.starts_with("after_") {
            self.find_next_node(&execution.current_node, &state).await?
        } else {
            execution.current_node
        };
        let mut steps = execution.steps;
        let mut recursion_count = execution.recursion_count;
        let first_node = current_node.clone();
        
        while current_node != END && recursion_count < self.recursion_limit {
            if current_node != first_node && self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }
            
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
            
            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!("after_{}", current_node)));
            }
            
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
    
    pub async fn resume(&self, execution: GraphExecution<S>) -> GraphResult<GraphInvocation<S>> {
        self.invoke_with_execution(execution).await
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
                    GraphEdge::FanOut { targets, .. } => {
                        if targets.is_empty() {
                            return Err(GraphError::RoutingError("FanOut has no targets".to_string()));
                        }
                        return Ok(targets[0].clone());
                    }
                    GraphEdge::FanIn { .. } => {
                        continue;
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
    
    fn find_fan_out_targets(&self, current: &str) -> Option<Vec<String>> {
        for edge in &self.edges {
            if edge.source() == current {
                if let GraphEdge::FanOut { targets, .. } = edge {
                    return Some(targets.clone());
                }
            }
        }
        None
    }
    
    fn find_fan_in_target(&self, sources: &[String]) -> Option<String> {
        for edge in &self.edges {
            if let GraphEdge::FanIn { sources: edge_sources, target } = edge {
                if edge_sources.iter().all(|s| sources.contains(s)) {
                    return Some(target.clone());
                }
            }
        }
        None
    }
    
    async fn execute_parallel_branches(
        &self,
        targets: &[String],
        state: &S,
    ) -> GraphResult<Vec<(String, GraphInvocation<S>)>> {
        let futures: Vec<_> = targets.iter()
            .filter(|t| *t != END)
            .map(|target| {
                let target = target.clone();
                let state_clone = state.clone();
                async move {
                    let result = self.invoke_from_node(target.clone(), state_clone).await;
                    result.map(|inv| (target, inv))
                }
            })
            .collect();
        
        let results = join_all(futures).await;
        
        let mut successful = Vec::new();
        for result in results {
            match result {
                Ok((name, inv)) => successful.push((name, inv)),
                Err(e) => return Err(e),
            }
        }
        
        Ok(successful)
    }
    
    pub async fn invoke_from_node(&self, start_node: String, input: S) -> GraphResult<GraphInvocation<S>> {
        let mut state = input;
        let mut current_node = start_node;
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;
        
        while current_node != END && recursion_count < self.recursion_limit {
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }
            
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
            
            if self.interrupt_after.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(format!("after_{}", current_node)));
            }
            
            current_node = self.find_next_node(&current_node, &state).await?;
        }
        
        Ok(GraphInvocation {
            final_state: state,
            steps,
            recursion_count,
        })
    }
    
    pub async fn invoke_parallel(&self, input: S) -> GraphResult<ParallelInvocation<S>> {
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut steps: Vec<ExecutionStep> = Vec::new();
        let mut recursion_count = 0;
        let mut parallel_branches: Vec<ParallelBranch<S>> = Vec::new();
        
        if let Some(ref checkpointer) = self.checkpointer {
            let checkpoint_id = checkpointer.lock().await.save(&state).await?;
            steps.push(ExecutionStep::checkpoint(checkpoint_id, current_node.clone()));
        }
        
        while current_node != END && recursion_count < self.recursion_limit {
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }
            
            recursion_count += 1;
            
            let fan_out_targets = self.find_fan_out_targets(&current_node);
            
            if let Some(targets) = fan_out_targets {
                let branch_results = self.execute_parallel_branches(&targets, &state).await?;
                
                for (name, inv) in branch_results {
                    parallel_branches.push(ParallelBranch {
                        name: name.clone(),
                        final_state: inv.final_state.clone(),
                        steps: inv.steps.clone(),
                    });
                    steps.push(ExecutionStep::ParallelNode { 
                        branch: name, 
                        metadata: HashMap::new() 
                    });
                }
                
                let merge_target = self.find_fan_in_target(&targets);
                if let Some(merge_node) = merge_target {
                    state = self.merge_parallel_states(&parallel_branches);
                    current_node = merge_node;
                } else {
                    state = parallel_branches.last()
                        .map(|b| b.final_state.clone())
                        .unwrap_or(state);
                    current_node = END.to_string();
                }
            } else {
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
                
                if self.interrupt_after.contains(&current_node) {
                    return Err(GraphError::ExecutionInterrupted(format!("after_{}", current_node)));
                }
                
                current_node = self.find_next_node(&current_node, &state).await?;
            }
        }
        
        if recursion_count >= self.recursion_limit {
            return Err(GraphError::RecursionLimitReached(self.recursion_limit));
        }
        
        Ok(ParallelInvocation {
            final_state: state,
            steps,
            recursion_count,
            parallel_branches,
        })
    }
    
    fn merge_parallel_states(&self, branches: &[ParallelBranch<S>]) -> S {
        if branches.is_empty() {
            return branches.first().map(|b| b.final_state.clone()).unwrap();
        }
        
        let mut merged = branches[0].final_state.clone();
        for branch in branches.iter().skip(1) {
            merged = self.default_reducer.reduce(&merged, &branch.final_state);
        }
        merged
    }
    
    /// Visualize the graph structure in ASCII format
    pub fn visualize_ascii(&self) -> String {
        let mut output = String::new();
        output.push_str("┌─────────────────────────────────────┐\n");
        output.push_str("│         LangGraph Structure         │\n");
        output.push_str("└─────────────────────────────────────┘\n\n");
        
        output.push_str(&format!("Entry Point: {}\n\n", self.entry_point));
        
        output.push_str("Nodes:\n");
        for name in self.nodes.keys() {
            output.push_str(&format!("  • {}\n", name));
        }
        
        output.push_str("\nEdges:\n");
        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    output.push_str(&format!("  {} → {}\n", source, target));
                }
                GraphEdge::Conditional { source, router_name, targets, .. } => {
                    output.push_str(&format!("  {} → [{}]\n", source, router_name));
                    for (route, target) in targets {
                        output.push_str(&format!("    {} → {}\n", route, target));
                    }
                }
                GraphEdge::FanOut { source, targets } => {
                    output.push_str(&format!("  {} → [FanOut]\n", source));
                    for target in targets {
                        output.push_str(&format!("    → {}\n", target));
                    }
                }
                GraphEdge::FanIn { sources, target } => {
                    output.push_str(&format!("  [FanIn] → {}\n", target));
                    for source in sources {
                        output.push_str(&format!("    {} →\n", source));
                    }
                }
            }
        }
        
        if !self.conditional_routers.is_empty() {
            output.push_str("\nRouters:\n");
            for name in self.conditional_routers.keys() {
                output.push_str(&format!("  • {}\n", name));
            }
        }
        
        output.push_str(&format!("\nRecursion Limit: {}\n", self.recursion_limit));
        
        output
    }
    
    /// Visualize the graph structure in Mermaid format
    pub fn visualize_mermaid(&self) -> String {
        let mut output = String::new();
        output.push_str("```mermaid\n");
        output.push_str("graph TD\n");
        
        output.push_str(&format!("  START[\"START\"]\n"));
        output.push_str(&format!("  END[\"END\"]\n"));
        
        for name in self.nodes.keys() {
            output.push_str(&format!("  {}[\"{}\"]\n", name, name));
        }
        
        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    output.push_str(&format!("  {} --> {}\n", source, target));
                }
                GraphEdge::Conditional { source, router_name, targets, .. } => {
                    output.push_str(&format!("  {} --> {{{}\n", source, router_name));
                    for (route, target) in targets {
                        output.push_str(&format!("    {} --> {}\n", route, target));
                    }
                    output.push_str("  }\n");
                }
                GraphEdge::FanOut { source, targets } => {
                    output.push_str(&format!("  {} --> {{\n", source));
                    for target in targets {
                        output.push_str(&format!("    --> {}\n", target));
                    }
                    output.push_str("  }\n");
                }
                GraphEdge::FanIn { sources, target } => {
                    for source in sources {
                        output.push_str(&format!("  {} --> {}\n", source, target));
                    }
                }
            }
        }
        
        output.push_str("```\n");
        output
    }
    
    /// Visualize the graph structure as JSON
    pub fn visualize_json(&self) -> serde_json::Value {
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();
        
        let edges: Vec<serde_json::Value> = self.edges.iter().map(|edge| {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    serde_json::json!({
                        "type": "fixed",
                        "source": source,
                        "target": target
                    })
                }
                GraphEdge::Conditional { source, router_name, targets, default_target } => {
                    serde_json::json!({
                        "type": "conditional",
                        "source": source,
                        "router": router_name,
                        "targets": targets,
                        "default": default_target
                    })
                }
                GraphEdge::FanOut { source, targets } => {
                    serde_json::json!({
                        "type": "fanout",
                        "source": source,
                        "targets": targets
                    })
                }
                GraphEdge::FanIn { sources, target } => {
                    serde_json::json!({
                        "type": "fanin",
                        "sources": sources,
                        "target": target
                    })
                }
            }
        }).collect();
        
        let routers: Vec<String> = self.conditional_routers.keys().cloned().collect();
        
        serde_json::json!({
            "entry_point": self.entry_point,
            "nodes": nodes,
            "edges": edges,
            "routers": routers,
            "recursion_limit": self.recursion_limit
        })
    }
    
    pub fn to_definition(&self) -> GraphDefinition {
        let mut definition = GraphDefinition::new(self.entry_point.clone())
            .with_recursion_limit(self.recursion_limit);
        
        for node_name in self.nodes.keys() {
            definition.add_node(NodeDefinition {
                name: node_name.clone(),
                node_type: NodeType::Sync,
                config: serde_json::json!({}),
            });
        }
        
        for edge in &self.edges {
            let edge_def = match edge {
                GraphEdge::Fixed { source, target } => {
                    EdgeDefinition::fixed(source.clone(), target.clone())
                }
                GraphEdge::Conditional { source, router_name, targets, default_target } => {
                    EdgeDefinition::conditional(
                        source.clone(),
                        router_name.clone(),
                        targets.clone(),
                        default_target.clone(),
                    )
                }
                GraphEdge::FanOut { source, targets } => {
                    EdgeDefinition::fan_out(source.clone(), targets.clone())
                }
                GraphEdge::FanIn { sources, target } => {
                    EdgeDefinition::fan_in(sources.clone(), target.clone())
                }
            };
            definition.add_edge(edge_def);
        }
        
        definition
    }
}

/// GraphInvocation - Result of graph execution
#[derive(Debug)]
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
    ParallelNode { branch: String, metadata: HashMap<String, JsonValue> },
}

impl ExecutionStep {
    pub fn node(name: String, metadata: HashMap<String, JsonValue>) -> Self {
        Self::Node { name, metadata }
    }
    
    pub fn checkpoint(id: String, next_node: String) -> Self {
        Self::Checkpoint { id, next_node }
    }
    
    pub fn parallel_node(branch: String, metadata: HashMap<String, JsonValue>) -> Self {
        Self::ParallelNode { branch, metadata }
    }
}

/// ParallelBranch - Result of a parallel execution branch
#[derive(Debug, Clone)]
pub struct ParallelBranch<S: StateSchema> {
    pub name: String,
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
}

/// ParallelInvocation - Result of graph execution with parallel branches
#[derive(Debug)]
pub struct ParallelInvocation<S: StateSchema> {
    pub final_state: S,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
    pub parallel_branches: Vec<ParallelBranch<S>>,
}

impl<S: StateSchema> ParallelInvocation<S> {
    pub fn state(&self) -> &S {
        &self.final_state
    }
    
    pub fn branches(&self) -> &[ParallelBranch<S>] {
        &self.parallel_branches
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

/// GraphExecution - State for interrupted execution that can be resumed
#[derive(Debug, Clone)]
pub struct GraphExecution<S: StateSchema> {
    pub state: S,
    pub current_node: String,
    pub steps: Vec<ExecutionStep>,
    pub recursion_count: usize,
    pub interrupted_at: String,
}

impl<S: StateSchema> GraphExecution<S> {
    pub fn new(state: S, current_node: String, interrupted_at: String) -> Self {
        Self {
            state,
            current_node,
            steps: Vec::new(),
            recursion_count: 0,
            interrupted_at,
        }
    }
    
    pub fn state(&self) -> &S {
        &self.state
    }
    
    pub fn interrupted_at(&self) -> &str {
        &self.interrupted_at
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