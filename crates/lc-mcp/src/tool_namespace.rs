//! MCP tool namespace isolation (P2-2): `server_name:tool_name` uniquification + conflict policy.
//!
//! Tools declared by 100+ Servers may share names (e.g. several Servers all have `read_file`); merging them
//! straight into the Agent registry would collide. This module uniquifies each tool as
//! `server_name:tool_name` and decides how same-named tools are treated by the explicit [`ToolConflict`]
//! policy:
//!
//! - [`Prefix`](ToolConflict::Prefix): same-named tools are all exposed, each prefixed with `server_name:`;
//! - [`Reject`](ToolConflict::Reject): a second registration of a same-named tool is rejected with an error.
//!
//! Namespaced tools are handed to
//! [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced) to attach to the Agent;
//! the LLM sees the prefixed external name, the actual call strips the prefix automatically and uses the
//! Server-side original tool name.

use std::collections::HashMap;

use crate::protocol::MCPError;
use crate::types::MCPToolDefinition;

/// Tool naming conflict handling policy (P2-2).
///
/// How same-named tools from different Servers are treated during `server_name:tool_name` uniquification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConflict {
    /// Expose all: conflicting tools get a `server_name:` prefix, each same-named tool becomes unique.
    Prefix,
    /// Reject with an error: a second same-named tool (belonging to a different Server) returns an error.
    Reject,
}

/// A namespaced tool entry.
#[derive(Debug, Clone)]
pub struct NamespacedTool {
    /// The unique externally exposed tool name (format `server_name:tool_name`).
    pub full_name: String,
    /// The source Server name (the connection-manager registry key).
    pub server: String,
    /// The original tool definition. The actual call still uses `definition.name` — the Server side does not
    /// know the prefix.
    pub definition: MCPToolDefinition,
}

/// Multi-Server tool namespace registry (P2-2).
///
/// Uniquifies each tool as `server_name:tool_name` and decides how same-named tools are treated by the
/// [`ToolConflict`] policy. External names are generated with [`qualify`](Self::qualify), looked back up with
/// [`resolve`](Self::resolve); namespaced tools handed to
/// [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced) automatically strip the prefix and use
/// the original name when attached to an Agent.
#[derive(Debug, Default)]
pub struct ToolNamespace {
    /// `full_name` → `(server, original tool name)`.
    index: HashMap<String, (String, String)>,
}

impl ToolNamespace {
    /// An empty namespace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generates the unique external name of a tool under a Server.
    pub fn qualify(server: &str, tool: &str) -> String {
        format!("{server}:{tool}")
    }

    /// Parses an external full name back into `(server, original tool name)`; returns `None` when it contains
    /// no `:`.
    pub fn parse(full_name: &str) -> Option<(&str, &str)> {
        full_name.split_once(':')
    }

    /// Registers a batch of tools, namespacing them per the conflict policy.
    ///
    /// - Re-registering the same `full_name` (`server:tool`) always errors;
    /// - under [`Reject`](ToolConflict::Reject), this registration errors when the original tool name already
    ///   belongs to another Server (i.e. it would collide without a prefix);
    /// - under [`Prefix`](ToolConflict::Prefix), same-named tools are each exposed with their own prefix,
    ///   without interfering.
    ///
    /// Returns the namespacing result, from which the caller can construct
    /// [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced).
    pub fn register(
        &mut self,
        server: &str,
        tools: impl IntoIterator<Item = MCPToolDefinition>,
        conflict: ToolConflict,
    ) -> Result<Vec<NamespacedTool>, MCPError> {
        let mut out = Vec::new();
        for tool in tools {
            let full_name = Self::qualify(server, &tool.name);
            if self.index.contains_key(&full_name) {
                return Err(MCPError::new(
                    -1,
                    format!("tool '{full_name}' is already registered"),
                ));
            }
            if conflict == ToolConflict::Reject
                && self.index.values().any(|(_, orig)| orig == &tool.name)
            {
                return Err(MCPError::new(
                    -1,
                    format!(
                        "tool '{}' conflicts with another server, refusing registration",
                        tool.name
                    ),
                ));
            }
            self.index
                .insert(full_name.clone(), (server.to_string(), tool.name.clone()));
            out.push(NamespacedTool {
                full_name,
                server: server.to_string(),
                definition: tool,
            });
        }
        Ok(out)
    }

    /// Looks an external full name back up as `(server, original tool name)`; returns `None` if not registered.
    pub fn resolve(&self, full_name: &str) -> Option<(&str, &str)> {
        self.index
            .get(full_name)
            .map(|(s, t)| (s.as_str(), t.as_str()))
    }

    /// Enumerates all registered external full names (order not stable). Used by the Gateway (P2-8) unified
    /// registry traversal.
    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::MCPClient;
    use crate::test_support::{start_fake_sse_server, PostMode};
    use crate::MCPConfig;
    use crate::MCPToolAdapter;
    use lc_core::BaseTool;
    use serde_json::json;

    fn tool(name: &str) -> MCPToolDefinition {
        MCPToolDefinition {
            name: name.to_string(),
            description: format!("{name} desc"),
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn test_qualify_and_parse() {
        let full = ToolNamespace::qualify("fs", "read_file");
        assert_eq!(full, "fs:read_file");
        assert_eq!(ToolNamespace::parse(&full), Some(("fs", "read_file")));
        assert_eq!(ToolNamespace::parse("no_separator"), None);
    }

    /// Prefix policy: same-named tools from different Servers are all exposed, each unique with its own prefix.
    #[test]
    fn test_prefix_policy_exposes_same_name_tools() {
        let mut ns = ToolNamespace::new();
        let a = ns
            .register("server_a", [tool("read")], ToolConflict::Prefix)
            .unwrap();
        let b = ns
            .register("server_b", [tool("read")], ToolConflict::Prefix)
            .unwrap();

        assert_eq!(a[0].full_name, "server_a:read");
        assert_eq!(b[0].full_name, "server_b:read");
        assert_eq!(ns.len(), 2);
        // Lookup routes to the correct server / original name.
        assert_eq!(ns.resolve("server_a:read"), Some(("server_a", "read")));
        assert_eq!(ns.resolve("server_b:read"), Some(("server_b", "read")));
    }

    /// Reject policy: a second registration of a same-named tool from a different Server errors, and the
    /// registry does not grow.
    #[test]
    fn test_reject_policy_rejects_same_name_across_servers() {
        let mut ns = ToolNamespace::new();
        ns.register("server_a", [tool("read")], ToolConflict::Reject)
            .unwrap();
        let err = ns
            .register("server_b", [tool("read")], ToolConflict::Reject)
            .unwrap_err();
        assert!(err.to_string().contains("conflicts"), "{}", err);
        assert_eq!(
            ns.len(),
            1,
            "conflicting tool should not be counted in the registry"
        );
    }

    /// Reject policy: distinct-named tools coexist normally.
    #[test]
    fn test_reject_policy_allows_distinct_tools() {
        let mut ns = ToolNamespace::new();
        ns.register(
            "server_a",
            [tool("read"), tool("write")],
            ToolConflict::Reject,
        )
        .unwrap();
        ns.register("server_b", [tool("search")], ToolConflict::Reject)
            .unwrap();
        assert_eq!(ns.len(), 3);
        assert_eq!(ns.resolve("server_b:search"), Some(("server_b", "search")));
    }

    /// Re-registering the same `full_name` errors under any policy.
    #[test]
    fn test_duplicate_registration_rejected() {
        let mut ns = ToolNamespace::new();
        ns.register("s", [tool("a")], ToolConflict::Prefix).unwrap();
        let err = ns
            .register("s", [tool("a")], ToolConflict::Prefix)
            .unwrap_err();
        assert!(err.to_string().contains("already registered"), "{}", err);
        assert_eq!(ns.len(), 1);
    }

    /// The conflict policy is decided explicitly per register call: whether a same-named tool conflicts
    /// depends on its source.
    #[test]
    fn test_conflict_policy_per_register_call() {
        let mut ns = ToolNamespace::new();
        ns.register("server_a", [tool("read")], ToolConflict::Prefix)
            .unwrap();
        // server_b's "read" shares the name with server_a: under the Prefix policy both are exposed.
        let b = ns
            .register("server_b", [tool("read")], ToolConflict::Prefix)
            .unwrap();
        assert_eq!(b[0].full_name, "server_b:read");
        assert_eq!(ns.len(), 2);
    }

    /// Namespaced adapter: the LLM sees `server:tool`, the actual call strips the prefix and uses the raw name.
    ///
    /// The fake SSE server's `tools/call` does not validate the tool name — if `run` called with the prefix
    /// and still returned successfully (rather than method-not-found / invalid-params), it proves the routing
    /// went through the original-name path.
    #[tokio::test]
    async fn test_namespaced_adapter_routes_to_raw_name() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let adapter = MCPToolAdapter::namespaced(client, "server_a", tool("echo"));
        // BaseTool::name() → the external name carries the namespace prefix.
        assert_eq!(adapter.name(), "server_a:echo");
        assert_eq!(adapter.display_name(), "server_a:echo");
        assert_eq!(adapter.description(), "echo desc");
        assert!(adapter.args_schema().is_some());
        // run strips the prefix and calls the fake server by the original name "echo"; the fake server echoes
        // the received tool name, so it equals the original name proves the namespace prefix did not leak to
        // the Server side.
        let out = adapter.run("{}".into()).await;
        assert!(
            matches!(out.as_deref(), Ok("echo")),
            "should call with the raw tool name, not the prefixed full name, actual: {:?}",
            out.as_deref()
        );
    }

    /// A non-namespaced adapter keeps the original name, matching the legacy behavior.
    #[tokio::test]
    async fn test_plain_adapter_keeps_raw_name() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("connecting to fake SSE server should succeed");

        let adapter = MCPToolAdapter::new(client, tool("echo"));
        assert_eq!(adapter.name(), "echo");
        assert_eq!(adapter.display_name(), "echo");
    }
}
