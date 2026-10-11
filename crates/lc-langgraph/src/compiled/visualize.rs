// crates/lc-langgraph/src/compiled/visualize.rs
//! CompiledGraph visualization and serialization methods:
//! `visualize(format)` and the per-format `visualize_ascii` / `visualize_mermaid` /
//! `visualize_json`, plus `to_definition`.

use super::graph::CompiledGraph;
use crate::edge::GraphEdge;
use crate::persistence::{EdgeDefinition, GraphDefinition, NodeDefinition, NodeType};
use crate::state::StateSchema;

/// The rendering format for [`CompiledGraph::visualize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// Terminal-friendly ASCII box drawing.
    Ascii,
    /// Mermaid Down (editable in editors / rendered by GitHub, Notion…).
    Mermaid,
    /// The structural graph as serialized JSON.
    Json,
}

impl<S: StateSchema> CompiledGraph<S> {
    /// Renders the graph structure in the requested [`GraphFormat`].
    ///
    /// The single entry point for drawing a graph: unlike the per-format methods it always
    /// returns a `String`, so `Ascii`/`Mermaid`/`Json` stay interchangeable (the JSON variant
    /// serializes [`Self::visualize_json`]). See the per-format methods for the raw shapes.
    pub fn visualize(&self, format: GraphFormat) -> String {
        match format {
            GraphFormat::Ascii => self.visualize_ascii(),
            GraphFormat::Mermaid => self.visualize_mermaid(),
            GraphFormat::Json => self.visualize_json().to_string(),
        }
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
                GraphEdge::Conditional {
                    source,
                    router_name,
                    targets,
                    ..
                } => {
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

        output.push_str("  START[\"START\"]\n");
        output.push_str("  END[\"END\"]\n");

        for name in self.nodes.keys() {
            output.push_str(&format!("  {}[\"{}\"]\n", name, name));
        }

        for edge in &self.edges {
            match edge {
                GraphEdge::Fixed { source, target } => {
                    output.push_str(&format!("  {} --> {}\n", source, target));
                }
                GraphEdge::Conditional {
                    source,
                    router_name,
                    targets,
                    ..
                } => {
                    for (route, target) in targets {
                        output.push_str(&format!("  {} -->|{}| {}\n", source, route, target));
                    }
                    let _ = router_name; // router name not rendered in Mermaid edge syntax
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

        let edges: Vec<serde_json::Value> = self
            .edges
            .iter()
            .map(|edge| match edge {
                GraphEdge::Fixed { source, target } => {
                    serde_json::json!({
                        "type": "fixed",
                        "source": source,
                        "target": target
                    })
                }
                GraphEdge::Conditional {
                    source,
                    router_name,
                    targets,
                    default_target,
                } => {
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
            })
            .collect();

        let routers: Vec<String> = self.conditional_routers.keys().cloned().collect();

        serde_json::json!({
            "entry_point": self.entry_point,
            "nodes": nodes,
            "edges": edges,
            "routers": routers,
            "recursion_limit": self.recursion_limit
        })
    }

    /// Convert the graph into a portable [`GraphDefinition`].
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
                GraphEdge::Conditional {
                    source,
                    router_name,
                    targets,
                    default_target,
                } => EdgeDefinition::conditional(
                    source.clone(),
                    router_name.clone(),
                    targets.clone(),
                    default_target.clone(),
                ),
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
