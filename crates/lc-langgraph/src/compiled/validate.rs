// crates/lc-langgraph/src/compiled/validate.rs
//! CompiledGraph validation methods: validate, validate_duplicate_edges,
//! validate_unreachable_nodes, validate_cycles, and helper graph-traversal methods

use super::graph::CompiledGraph;
use crate::edge::GraphEdge;
use crate::errors::{GraphError, GraphResult};
use crate::state::StateSchema;
use crate::{END, START};

impl<S: StateSchema> CompiledGraph<S> {
    pub fn validate(&self) -> GraphResult<()> {
        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(format!(
                            "Source node '{}' not found",
                            source
                        )));
                    }
                    if target != END && !self.nodes.contains_key(target) {
                        return Err(GraphError::ValidationError(format!(
                            "Target node '{}' not found",
                            target
                        )));
                    }
                    if target == START {
                        return Err(GraphError::ValidationError(
                            "Edge cannot target START node".to_string(),
                        ));
                    }
                }
                GraphEdge::Conditional {
                    source,
                    router_name,
                    targets,
                    default_target,
                } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(format!(
                            "Source node '{}' not found",
                            source
                        )));
                    }
                    if !self.conditional_routers.contains_key(router_name) {
                        return Err(GraphError::ValidationError(format!(
                            "Router '{}' not found",
                            router_name
                        )));
                    }
                    for (route, target) in targets {
                        if target != END && !self.nodes.contains_key(target) {
                            return Err(GraphError::ValidationError(format!(
                                "Target '{}' for route '{}' not found",
                                target, route
                            )));
                        }
                        if target == START {
                            return Err(GraphError::ValidationError(
                                "Conditional edge cannot target START node".to_string(),
                            ));
                        }
                    }
                    if let Some(default) = default_target {
                        if default != END && !self.nodes.contains_key(default) {
                            return Err(GraphError::ValidationError(format!(
                                "Default target '{}' not found",
                                default
                            )));
                        }
                    }
                }
                GraphEdge::FanOut { source, targets } => {
                    if source != START && !self.nodes.contains_key(source) {
                        return Err(GraphError::ValidationError(format!(
                            "FanOut source node '{}' not found",
                            source
                        )));
                    }
                    for target in targets {
                        if target != END && !self.nodes.contains_key(target) {
                            return Err(GraphError::ValidationError(format!(
                                "FanOut target node '{}' not found",
                                target
                            )));
                        }
                    }
                }
                GraphEdge::FanIn { sources, target } => {
                    for source in sources {
                        if source != START && !self.nodes.contains_key(source) {
                            return Err(GraphError::ValidationError(format!(
                                "FanIn source node '{}' not found",
                                source
                            )));
                        }
                    }
                    if target != END && !self.nodes.contains_key(target) {
                        return Err(GraphError::ValidationError(format!(
                            "FanIn target node '{}' not found",
                            target
                        )));
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
        let mut seen_fixed: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();

        for edge in &self.edges {
            if let GraphEdge::Fixed { source, target } = edge {
                let key = (source.clone(), target.clone());
                if seen_fixed.contains(&key) {
                    return Err(GraphError::DuplicateEdgeError(format!(
                        "Duplicate edge: {} -> {}",
                        source, target
                    )));
                }
                seen_fixed.insert(key);
            }
        }
        Ok(())
    }

    fn validate_unreachable_nodes(&self) -> GraphResult<()> {
        let reachable = self.compute_reachable_nodes();

        for node_name in self.nodes.keys() {
            if !reachable.contains(node_name) {
                return Err(GraphError::OrphanNodeError(format!(
                    "Unreachable node: {}",
                    node_name
                )));
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
                        GraphEdge::Conditional {
                            targets,
                            default_target,
                            ..
                        } => {
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
                            if sources.iter().all(|s| reachable.contains(s))
                                && !reachable.contains(target)
                                && target != END
                            {
                                to_visit.push(target.clone());
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
                return Err(GraphError::InfiniteCycleError(format!(
                    "Node '{}' in cycle with no path to END",
                    node
                )));
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
                    GraphEdge::Conditional {
                        source,
                        targets,
                        default_target,
                        ..
                    } => {
                        let any_target_reaches_end =
                            targets.values().any(|t| end_reachable.contains(t))
                                || default_target
                                    .as_ref()
                                    .is_some_and(|d| end_reachable.contains(d));
                        if any_target_reaches_end && !end_reachable.contains(source) {
                            end_reachable.insert(source.clone());
                            changed = true;
                        }
                    }
                    GraphEdge::FanOut { source, targets } => {
                        let all_targets_reach_end =
                            targets.iter().all(|t| end_reachable.contains(t));
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
}
