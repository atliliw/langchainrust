//! 静态层 + 动态层工具发现(P2-3)。
//!
//! 100+ Server 全量声明几十万 token,超模型上下文窗口。因此不把所有工具一股脑
//! 注入 Agent,而是分两层:
//!
//! - **静态层**:20-50 个高频常驻工具,每次调用固定注入(`pin` 标记);
//! - **动态层**:按 query 相关性取 top-k 个工具临时注入(工具发现,类比 RAG 的
//!   检索,把"查工具"当"查文档"处理)。
//!
//! 相关性评分走 [`ToolScorer`] trait 的 [`KeywordScorer`](KeywordScorer) 默认实现
//! (词元重叠,零额外依赖);需要向量检索时自行实现 `ToolScorer`(如用 lc-embeddings
//! 算 query / tool 相似度)替换,`with_scorer` 注入。

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::MCPToolDefinition;

/// 工具相关性评分器(P2-3)。
///
/// 动态层按 `query` 与工具的 `name + description` 计算相关分。默认用
/// [`KeywordScorer`];向量检索场景自行实现并 `with_scorer` 注入。
pub trait ToolScorer: Send + Sync {
    /// 计算 query 与单个工具的相关性,越高越相关。
    fn score(&self, query: &str, tool: &MCPToolDefinition) -> f64;
}

/// 默认关键词评分器:query 词元在工具名 + 描述中的命中比例。
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

/// 静态层 + 动态层工具选择器(P2-3)。
///
/// `register` 收录全量工具;`pin` 把高频工具加入静态层常驻注入;`select` 按
/// `static_limit` + `top_k` 返回"静态层固定工具 + 动态层按 query 检索的工具",
/// 动态层自动排除已在静态层的工具(去重)。
pub struct ToolDiscovery {
    /// 全量工具:`name` → 定义。
    tools: HashMap<String, MCPToolDefinition>,
    /// 静态层常驻工具名(有序,注入顺序稳定)。
    pinned: Vec<String>,
    /// 动态层相关性评分器。
    scorer: Arc<dyn ToolScorer>,
}

impl Default for ToolDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolDiscovery {
    /// 空发现器,使用默认关键词评分器。
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            pinned: Vec::new(),
            scorer: Arc::new(KeywordScorer),
        }
    }

    /// 自定义评分器(向量检索等)。
    pub fn with_scorer(scorer: Arc<dyn ToolScorer>) -> Self {
        Self {
            tools: HashMap::new(),
            pinned: Vec::new(),
            scorer,
        }
    }

    /// 收录一个工具;同名覆盖。
    pub fn register(&mut self, def: MCPToolDefinition) {
        self.tools.insert(def.name.clone(), def);
    }

    /// 把工具加入静态层(高频常驻注入)。工具未注册时返回 `false`。
    pub fn pin(&mut self, name: &str) -> bool {
        if !self.tools.contains_key(name) {
            return false;
        }
        if !self.pinned.iter().any(|n| n == name) {
            self.pinned.push(name.to_string());
        }
        true
    }

    /// 从静态层移除工具(仍保留在动态层)。
    pub fn unpin(&mut self, name: &str) {
        self.pinned.retain(|n| n != name);
    }

    /// 静态层:固定注入的常驻工具定义(最多 `limit` 个,保持 pin 顺序)。
    pub fn static_layer(&self, limit: usize) -> Vec<MCPToolDefinition> {
        self.pinned
            .iter()
            .filter_map(|n| self.tools.get(n).cloned())
            .take(limit)
            .collect()
    }

    /// 动态层:按 query 相关性降序取 top-k 个工具临时注入。
    ///
    /// `exclude` 中已有的工具跳过(通常传静态层已注入的工具名去重)。
    /// query 为空或 top_k 为 0 时返回空。
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

    /// 全量注入:静态层 + 动态层合并,动态层排除静态层已注入的工具。
    pub fn select(&self, query: &str, top_k: usize, static_limit: usize) -> Vec<MCPToolDefinition> {
        let static_tools = self.static_layer(static_limit);
        // 转成持有所有权的名字集,避免借用 static_tools(随后被 move)。
        let exclude: Vec<String> = static_tools.iter().map(|t| t.name.clone()).collect();
        let exclude_refs: Vec<&str> = exclude.iter().map(String::as_str).collect();
        let mut out = static_tools;
        out.extend(self.dynamic_layer(query, top_k, &exclude_refs));
        out
    }

    /// 已收录的工具总数。
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// 是否为空。
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
        d.pin("get_time"); // 静态层常驻

        let picked = d.select("query the database", 5, usize::MAX);
        let names: Vec<&str> = picked.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), 2, "静态层 + 动态层去重合并");
        assert!(names.contains(&"get_time"));
        assert!(names.contains(&"search_db"));
        // 动态层不重复注入静态层工具
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
        // 自定义评分器返回全 0 → 动态层仍返回(排序稳定),不依赖内置关键词实现。
        let picks = d.dynamic_layer("query the database", 1, &[]);
        assert_eq!(picks.len(), 1);
    }
}
