//! Static + dynamic layered tool discovery (P2-3).
//!
//! Declaring all tools from 100+ Servers would cost hundreds of thousands of tokens, exceeding the model's
//! context window. So instead of injecting every tool into the Agent, we split them into two layers:
//!
//! - **Static layer**: 20-50 high-frequency resident tools, injected fixedly on every call (`pin` mark);
//! - **Dynamic layer**: top-k tools injected temporarily by query relevance (tool discovery, analogous to RAG
//!   retrieval — treating "looking up tools" as "looking up documents").
//!
//! Relevance scoring goes through the [`ToolScorer`] trait's default [`KeywordScorer`] implementation
//! (token overlap, zero extra dependencies); for vector retrieval, implement `ToolScorer` yourself
//! (e.g. computing query / tool similarity with lc-embeddings) and inject it via `with_scorer`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::MCPToolDefinition;

/// Tool relevance scorer (P2-3).
///
/// The dynamic layer scores the relevance of `query` against a tool's `name + description`. Defaults to
/// [`KeywordScorer`]; for vector-retrieval scenarios implement `ToolScorer` yourself and inject it via
/// `with_scorer`.
pub trait ToolScorer: Send + Sync {
    /// Computes the relevance of `query` to a single tool; higher means more relevant.
    fn score(&self, query: &str, tool: &MCPToolDefinition) -> f64;
}

/// Default keyword scorer: hit ratio of query tokens in the tool name + description.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeywordScorer;

impl KeywordScorer {
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect()
    }
}

impl ToolScorer for KeywordScorer {
    fn score(&self, query: &str, tool: &MCPToolDefinition) -> f64 {
        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() {
            return 0.0;
        }
        let doc_tokens: std::collections::HashSet<String> =
            Self::tokenize(&format!("{} {}", tool.name, tool.description))
                .into_iter()
                .collect();
        let hits = query_tokens
            .iter()
            .filter(|t| doc_tokens.contains(*t))
            .count();
        hits as f64 / query_tokens.len() as f64
    }
}

/// Static + dynamic layered tool selector (P2-3).
///
/// `register` collects the full tool set; `pin` puts high-frequency tools into the static resident layer;
/// `select` returns "static-layer fixed tools + dynamic-layer tools retrieved by query" per
/// `static_limit` + `top_k`, and the dynamic layer automatically excludes tools already in the static layer
/// (dedup).
pub struct ToolDiscovery {
    /// The full tool set: `name` → definition.
    tools: HashMap<String, MCPToolDefinition>,
    /// Resident static-layer tool names (ordered, stable injection order).
    pinned: Vec<String>,
    /// Dynamic-layer relevance scorer.
    scorer: Arc<dyn ToolScorer>,
}

impl Default for ToolDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolDiscovery {
    /// An empty discovery, using the default keyword scorer.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            pinned: Vec::new(),
            scorer: Arc::new(KeywordScorer),
        }
    }

    /// A custom scorer (vector retrieval, etc.).
    pub fn with_scorer(scorer: Arc<dyn ToolScorer>) -> Self {
        Self {
            tools: HashMap::new(),
            pinned: Vec::new(),
            scorer,
        }
    }

    /// Registers a tool; same-name registration overwrites.
    pub fn register(&mut self, def: MCPToolDefinition) {
        self.tools.insert(def.name.clone(), def);
    }

    /// Puts a tool into the static layer (high-frequency resident injection). Returns `false` if the tool is
    /// not registered.
    pub fn pin(&mut self, name: &str) -> bool {
        if !self.tools.contains_key(name) {
            return false;
        }
        if !self.pinned.iter().any(|n| n == name) {
            self.pinned.push(name.to_string());
        }
        true
    }

    /// Removes a tool from the static layer (still kept in the dynamic layer).
    pub fn unpin(&mut self, name: &str) {
        self.pinned.retain(|n| n != name);
    }

    /// Static layer: resident tools injected fixedly (at most `limit` of them, keeping the pin order).
    pub fn static_layer(&self, limit: usize) -> Vec<MCPToolDefinition> {
        self.pinned
            .iter()
            .filter_map(|n| self.tools.get(n).cloned())
            .take(limit)
            .collect()
    }

    /// Dynamic layer: takes the top-k tools by query relevance descending for temporary injection.
    ///
    /// Tools already in `exclude` are skipped (usually the static-layer tool names already injected, for dedup).
    /// An empty query or `top_k == 0` returns nothing.
    pub fn dynamic_layer(
        &self,
        query: &str,
        top_k: usize,
        exclude: &[&str],
    ) -> Vec<MCPToolDefinition> {
        if top_k == 0 || query.trim().is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f64, &MCPToolDefinition)> = self
            .tools
            .values()
            .filter(|t| !exclude.contains(&t.name.as_str()))
            .map(|t| (self.scorer.score(query, t), t))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, t)| t.clone())
            .collect()
    }

    /// Full injection: static + dynamic layers merged, the dynamic layer excluding tools already injected by
    /// the static layer.
    pub fn select(&self, query: &str, top_k: usize, static_limit: usize) -> Vec<MCPToolDefinition> {
        let static_tools = self.static_layer(static_limit);
        // Build an owned name set to avoid borrowing static_tools (subsequently moved).
        let exclude: Vec<String> = static_tools.iter().map(|t| t.name.clone()).collect();
        let exclude_refs: Vec<&str> = exclude.iter().map(String::as_str).collect();
        let mut out = static_tools;
        out.extend(self.dynamic_layer(query, top_k, &exclude_refs));
        out
    }

    /// Total number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, description: &str) -> MCPToolDefinition {
        MCPToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn test_pin_unknown_tool_returns_false() {
        let mut d = ToolDiscovery::new();
        d.register(tool("known", "a known tool"));
        assert!(d.pin("known"));
        assert!(!d.pin("ghost"), "未注册工具不能 pin");
        assert!(!d.pin(""), "空名不能 pin");
    }
    #[test]
    fn test_static_layer_returns_pinned_in_order() {
        let mut d = ToolDiscovery::new();
        d.register(tool("a", "tool a"));
        d.register(tool("b", "tool b"));
        d.register(tool("c", "tool c"));
        d.pin("b");
        d.pin("a");
        let static_tools = d.static_layer(usize::MAX);
        let names: Vec<&str> = static_tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["b", "a"], "静态层按 pin 顺序注入");
    }

    #[test]
    fn test_static_layer_honors_limit() {
        let mut d = ToolDiscovery::new();
        for i in 0..5 {
            d.register(tool(&format!("t{i}"), &format!("tool {i}")));
            d.pin(&format!("t{i}"));
        }
        assert_eq!(d.static_layer(3).len(), 3, "静态层受 limit 约束");
    }

    #[test]
    fn test_dynamic_layer_ranks_by_query_relevance() {
        let mut d = ToolDiscovery::new();
        d.register(tool("search_db", "query the sql database"));
        d.register(tool("render_image", "draw an image from text"));
        d.register(tool("summarize", "summarize a long document"));

        let picks = d.dynamic_layer("search the database", 2, &[]);
        let names: Vec<&str> = picks.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names[0], "search_db", "最相关的工具应排第一");
        assert_eq!(picks.len(), 2);
    }

    #[test]
    fn test_dynamic_layer_excludes_given_names() {
        let mut d = ToolDiscovery::new();
        d.register(tool("search_db", "query the sql database"));
        d.register(tool("search_web", "query the web"));
        let picks = d.dynamic_layer("query database", 10, &["search_db"]);
        let names: Vec<&str> = picks.iter().map(|t| t.name.as_str()).collect();
        assert!(
            !names.contains(&"search_db"),
            "被排除的工具不应返回,实际 {:?}",
            names
        );
        assert!(
            names.contains(&"search_web"),
            "未排除的相关工具仍按相关性返回,实际 {:?}",
            names
        );
    }

    #[test]
    fn test_dynamic_layer_empty_query_returns_nothing() {
        let mut d = ToolDiscovery::new();
        d.register(tool("any", "anything"));
        assert!(d.dynamic_layer("", 10, &[]).is_empty());
        assert!(d.dynamic_layer("  ", 10, &[]).is_empty());
        assert!(d.dynamic_layer("q", 0, &[]).is_empty());
    }

    #[test]
    fn test_select_combines_static_and_dynamic_without_dup() {
        let mut d = ToolDiscovery::new();
        d.register(tool("search_db", "query the sql database"));
        d.register(tool("get_time", "get current time"));
        d.pin("get_time"); // static-layer resident

        let picked = d.select("query the database", 5, usize::MAX);
        let names: Vec<&str> = picked.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), 2, "静态层 + 动态层去重合并");
        assert!(names.contains(&"get_time"));
        assert!(names.contains(&"search_db"));
        // the dynamic layer does not re-inject static-layer tools
        assert_eq!(
            names.len(),
            names.iter().collect::<std::collections::HashSet<_>>().len()
        );
    }

    #[test]
    fn test_unpin_removes_from_static_layer() {
        let mut d = ToolDiscovery::new();
        d.register(tool("x", "tool x"));
        d.pin("x");
        assert_eq!(d.static_layer(usize::MAX).len(), 1);
        d.unpin("x");
        assert!(d.static_layer(usize::MAX).is_empty());
        assert_eq!(d.len(), 1, "unpin 只移出静态层,工具仍在全量集中");
    }

    #[test]
    fn test_custom_scorer_seam() {
        struct AlwaysZero;
        impl ToolScorer for AlwaysZero {
            fn score(&self, _query: &str, _tool: &MCPToolDefinition) -> f64 {
                0.0
            }
        }
        let mut d = ToolDiscovery::with_scorer(Arc::new(AlwaysZero));
        d.register(tool("search_db", "query the sql database"));
        // The custom scorer returns all 0s → the dynamic layer still returns (stable ordering), not relying on
        // the built-in keyword implementation.
        let picks = d.dynamic_layer("query the database", 1, &[]);
        assert_eq!(picks.len(), 1);
    }
}
