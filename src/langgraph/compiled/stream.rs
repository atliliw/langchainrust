// src/langgraph/compiled/stream.rs
//! CompiledGraph stream, find_next_node, find_fan_out_targets,
//! find_fan_in_target, and execute_parallel_branches methods

use super::super::edge::GraphEdge;
use super::super::errors::{GraphError, GraphResult};
use super::super::graph::END;
use super::super::node::NodeConfig;
use super::super::state::StateSchema;
use super::graph::CompiledGraph;
use super::types::{GraphInvocation, StreamEvent};
use futures_util::future::join_all;
use std::collections::HashMap;

impl<S: StateSchema> CompiledGraph<S> {
    pub async fn stream(&self, input: S) -> GraphResult<Vec<StreamEvent<S>>> {
        let mut events = Vec::new();
        let mut state = input;
        let mut current_node = self.entry_point.clone();
        let mut recursion_count = 0;

        events.push(StreamEvent::start(state.clone()));

        if let Some(ref checkpointer) = self.checkpointer {
            let _checkpoint_id = checkpointer.lock().await.save(&state).await?;
        }

        while current_node != END && recursion_count < self.recursion_limit {
            if self.interrupt_before.contains(&current_node) {
                return Err(GraphError::ExecutionInterrupted(current_node.clone()));
            }

            recursion_count += 1;

            events.push(StreamEvent::enter_node(current_node.clone(), state.clone()));

            let node = self.get_node(&current_node).await?;

            let config = NodeConfig {
                recursion_limit: self.recursion_limit,
                debug: false,
                metadata: HashMap::new(),
            };

            let update = node.execute(&state, Some(config)).await?;

            events.push(StreamEvent::node_complete(
                current_node.clone(),
                update.clone(),
            ));

            if let Some(new_state) = update.update {
                state = self.default_reducer.reduce(&state, &new_state);
                events.push(StreamEvent::state_update(state.clone()));
            }

            if self.interrupt_after.contains(&current_node) {
                if let Some(ref checkpointer) = self.checkpointer {
                    let _checkpoint_id = checkpointer.lock().await.save(&state).await?;
                }
                return Err(GraphError::ExecutionInterrupted(format!(
                    "after_{}",
                    current_node
                )));
            }

            let next_node = self.find_next_node(&current_node, &state).await?;

            if let Some(ref checkpointer) = self.checkpointer {
                let _checkpoint_id = checkpointer.lock().await.save(&state).await?;
            }

            current_node = next_node;
        }

        events.push(StreamEvent::end(state.clone()));
        Ok(events)
    }

    pub(super) async fn find_next_node(&self, current: &str, state: &S) -> GraphResult<String> {
        // 1. Check runtime edges — edge is cloned out of the guard before any .await
        'rt: {
            let edge = {
                let re = self.runtime_edges.read().await;
                match re.iter().find(|e| e.source() == current) {
                    Some(e) => e.clone(),
                    None => break 'rt,
                }
            }; // re dropped here
            match edge {
                GraphEdge::Fixed { target, .. } => return Ok(target),
                GraphEdge::Conditional {
                    router_name,
                    targets,
                    default_target,
                    ..
                } => {
                    let router = self
                        .conditional_routers
                        .get(&router_name)
                        .cloned()
                        .or_else(|| {
                            self.runtime_conditional_routers
                                .try_read()
                                .ok()
                                .and_then(|guard| guard.get(&router_name).cloned())
                        })
                        .ok_or_else(|| {
                            GraphError::ExecutionError(format!(
                                "Router '{}' not found (runtime)",
                                router_name
                            ))
                        })?;
                    let route_key = router.route(state).await?;
                    let target = targets
                        .get(&route_key)
                        .or(default_target.as_ref())
                        .ok_or_else(|| {
                            GraphError::RoutingError(format!(
                                "No target for route '{}' (runtime)",
                                route_key
                            ))
                        })?;
                    return Ok(target.clone());
                }
                GraphEdge::FanOut { targets, .. } => {
                    if targets.is_empty() {
                        return Err(GraphError::RoutingError(
                            "FanOut has no targets (runtime)".to_string(),
                        ));
                    }
                    return Ok(targets[0].clone());
                }
                GraphEdge::FanIn { .. } => {}
            }
        }

        // 2. Check static edges
        for edge in &self.edges {
            if edge.source() == current {
                match edge {
                    GraphEdge::Fixed { target, .. } => {
                        return Ok(target.clone());
                    }
                    GraphEdge::Conditional {
                        router_name,
                        targets,
                        default_target,
                        ..
                    } => {
                        let router = self
                            .conditional_routers
                            .get(router_name)
                            .cloned()
                            .or_else(|| {
                                self.runtime_conditional_routers
                                    .try_read()
                                    .ok()
                                    .and_then(|guard| guard.get(router_name).cloned())
                            })
                            .ok_or_else(|| {
                                GraphError::ExecutionError(format!(
                                    "Router '{}' not found",
                                    router_name
                                ))
                            })?;

                        let route_key = router.route(state).await?;

                        let target = targets
                            .get(&route_key)
                            .or(default_target.as_ref())
                            .ok_or_else(|| {
                                GraphError::RoutingError(format!(
                                    "No target for route '{}'",
                                    route_key
                                ))
                            })?;

                        return Ok(target.clone());
                    }
                    GraphEdge::FanOut { targets, .. } => {
                        if targets.is_empty() {
                            return Err(GraphError::RoutingError(
                                "FanOut has no targets".to_string(),
                            ));
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

        Err(GraphError::RoutingError(format!(
            "No outgoing edge from node '{}'",
            current
        )))
    }

    pub(super) async fn find_fan_out_targets(&self, current: &str) -> Option<Vec<String>> {
        {
            let re = self.runtime_edges.read().await;
            if let Some(GraphEdge::FanOut { targets, .. }) =
                re.iter().find(|e| e.source() == current)
            {
                return Some(targets.clone());
            }
        }
        for edge in &self.edges {
            if edge.source() == current {
                if let GraphEdge::FanOut { targets, .. } = edge {
                    return Some(targets.clone());
                }
            }
        }
        None
    }

    pub(super) async fn find_fan_in_target(&self, sources: &[String]) -> Option<String> {
        {
            let re = self.runtime_edges.read().await;
            if let Some(GraphEdge::FanIn {
                sources: edge_sources,
                target,
            }) = re.iter().find(|e| matches!(e, GraphEdge::FanIn { .. }))
            {
                if edge_sources.iter().all(|s| sources.contains(s)) {
                    return Some(target.clone());
                }
            }
        }
        for edge in &self.edges {
            if let GraphEdge::FanIn {
                sources: edge_sources,
                target,
            } = edge
            {
                if edge_sources.iter().all(|s| sources.contains(s)) {
                    return Some(target.clone());
                }
            }
        }
        None
    }

    pub(super) async fn execute_parallel_branches(
        &self,
        targets: &[String],
        state: &S,
    ) -> GraphResult<Vec<(String, GraphInvocation<S>)>> {
        let futures: Vec<_> = targets
            .iter()
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
}
