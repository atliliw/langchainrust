// src/core/tools/registry.rs
//! Tool registry for managing multiple tools.
//!
//! Provides unified lookup and execution interface.

use super::BaseTool;
use std::collections::HashMap;
use std::sync::Arc;

/// Tool registry for storing and managing multiple tool instances.
///
/// Provides name-based lookup and execution functionality.
///
/// # Example
/// ```ignore
/// use langchainrust::ToolRegistry;
/// use langchainrust::Calculator;
/// use std::sync::Arc;
///
/// let registry = ToolRegistry::new();
/// registry.register(Arc::new(Calculator::new()));
///
/// let tool = registry.get("calculator").unwrap();
/// ```
pub struct ToolRegistry {
    /// Tool storage (indexed by name).
    tools: HashMap<String, Arc<dyn BaseTool>>,
}

impl ToolRegistry {
    /// Creates an empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool.
    ///
    /// # Arguments
    /// * `tool` - Tool to register (wrapped in Arc for sharing).
    ///
    /// # Returns
    /// Previous tool if name conflict, otherwise None.
    pub fn register(&mut self, tool: Arc<dyn BaseTool>) -> Option<Arc<dyn BaseTool>> {
        let name = tool.name().to_string();
        self.tools.insert(name, tool)
    }

    /// Gets a tool by name.
    ///
    /// # Arguments
    /// * `name` - Tool name.
    ///
    /// # Returns
    /// Tool reference if found, otherwise None.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn BaseTool>> {
        self.tools.get(name)
    }

    /// Returns all tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s: &String| s.as_str()).collect()
    }

    /// Returns all tools.
    pub fn tools(&self) -> Vec<&Arc<dyn BaseTool>> {
        self.tools.values().collect()
    }

    /// Returns tool count.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns whether registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Removes a tool by name.
    ///
    /// # Arguments
    /// * `name` - Tool name.
    ///
    /// # Returns
    /// Removed tool if found, otherwise None.
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn BaseTool>> {
        self.tools.remove(name)
    }

    /// Checks if a tool exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Generates tool description for LLM.
    ///
    /// Used to show available tools to the LLM.
    pub fn describe_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available".to_string();
        }

        let mut description = String::from("Available tools:\n");

        for (name, tool) in &self.tools {
            description.push_str(&format!("- {}: {}\n", name, tool.description()));

            // Add input format description
            if let Some(schema) = tool.args_schema() {
                if let Some(props) = schema.get("properties") {
                    description.push_str("  Input parameters:\n");
                    if let Some(obj) = props.as_object() {
                        for (prop_name, prop_value) in obj {
                            let prop_desc = prop_value
                                .get("description")
                                .and_then(|d: &serde_json::Value| d.as_str())
                                .unwrap_or("No description");
                            description.push_str(&format!("    - {}: {}\n", prop_name, prop_desc));
                        }
                    }
                }
            }
        }

        description
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tool_count", &self.tools.len())
            .field("tool_names", &self.tool_names())
            .finish()
    }
}
