//! MCP 工具命名空间隔离(P2-2):`server_name:tool_name` 唯一化 + 冲突策略。
//!
//! 100+ Server 各自声明的工具可能同名(如多个 Server 都有 `read_file`),直接
//! 合并注册到 Agent 会撞名。本模块把每个工具唯一化为 `server_name:tool_name`,
//! 并按显式 [`ToolConflict`] 策略决定同名工具的取舍:
//!
//! - [`Prefix`](ToolConflict::Prefix):同名工具都暴露,统一加 `server_name:` 前缀;
//! - [`Reject`](ToolConflict::Reject):同名工具第二次注册时拒绝,返回错误。
//!
//! 命名空间化后的工具交给
//! [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced) 挂到 Agent,
//! LLM 看到带前缀的对外名,实际调用自动剥掉前缀走 Server 侧原始工具名。

use std::collections::HashMap;

use crate::protocol::MCPError;
use crate::types::MCPToolDefinition;

/// 工具命名冲突处理策略(P2-2)。
///
/// `server_name:tool_name` 唯一化时,不同 Server 的同名工具如何处理。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConflict {
    /// 都暴露:冲突工具加 `server_name:` 前缀,同名工具各自唯一。
    Prefix,
    /// 报错拒绝注册:第二次出现同名工具(属不同 Server)时返回错误。
    Reject,
}

/// 命名空间化后的工具条目。
#[derive(Debug, Clone)]
pub struct NamespacedTool {
    /// 对外暴露的唯一工具名(格式 `server_name:tool_name`)。
    pub full_name: String,
    /// 来源 Server 名(连接管理器注册表 key)。
    pub server: String,
    /// 原始工具定义。实际调用仍用 `definition.name`——Server 侧不认识前缀。
    pub definition: MCPToolDefinition,
}

/// 多 Server 工具命名空间注册表(P2-2)。
///
/// 把每个工具唯一化为 `server_name:tool_name`,并按 [`ToolConflict`] 策略
/// 决定同名工具的取舍。对外名用 [`qualify`](Self::qualify) 生成、反查用
/// [`resolve`](Self::resolve);命名空间化后的工具交给
/// [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced) 挂到
/// Agent 时自动剥掉前缀走原始名。
#[derive(Debug, Default)]
pub struct ToolNamespace {
    /// `full_name` → `(server, 原始工具名)`。
    index: HashMap<String, (String, String)>,
}

impl ToolNamespace {
    /// 空命名空间。
    pub fn new() -> Self {
        Self::default()
    }

    /// 生成某个 Server 下工具的唯一对外名。
    pub fn qualify(server: &str, tool: &str) -> String {
        format!("{server}:{tool}")
    }

    /// 从对外全名解析回 `(server, 原始工具名)`;不含 `:` 则返回 `None`。
    pub fn parse(full_name: &str) -> Option<(&str, &str)> {
        full_name.split_once(':')
    }

    /// 注册一批工具,按冲突策略命名空间化。
    ///
    /// - 同一 `full_name`(`server:tool`)重复注册总是报错;
    /// - [`Reject`](ToolConflict::Reject) 下,原始工具名已属于其它 Server(即
    ///   不加前缀就会撞名)时,本次注册报错;
    /// - [`Prefix`](ToolConflict::Prefix) 下,同名工具各自带前缀暴露,互不干扰。
    ///
    /// 返回命名空间化结果,调用方可据此构造
    /// [`MCPToolAdapter::namespaced`](crate::MCPToolAdapter::namespaced)。
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
                return Err(MCPError::new(-1, format!("tool '{full_name}' 重复注册")));
            }
            if conflict == ToolConflict::Reject
                && self.index.values().any(|(_, orig)| orig == &tool.name)
            {
                return Err(MCPError::new(
                    -1,
                    format!("tool '{}' 与其他 Server 冲突,拒绝注册", tool.name),
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

    /// 从对外全名反查 `(server, 原始工具名)`;未注册返回 `None`。
    pub fn resolve(&self, full_name: &str) -> Option<(&str, &str)> {
        self.index
            .get(full_name)
            .map(|(s, t)| (s.as_str(), t.as_str()))
    }

    /// 枚举全部已注册的对外全名(顺序不稳定)。Gateway(P2-8)统一注册表遍历用。
    pub fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    /// 已注册的工具数。
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// 是否为空注册表。
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

    /// Prefix 策略:不同 Server 的同名工具都暴露,各自带前缀唯一。
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
        // 反查路由到正确 server / 原始名。
        assert_eq!(ns.resolve("server_a:read"), Some(("server_a", "read")));
        assert_eq!(ns.resolve("server_b:read"), Some(("server_b", "read")));
    }

    /// Reject 策略:不同 Server 的同名工具第二次注册报错,注册表不增长。
    #[test]
    fn test_reject_policy_rejects_same_name_across_servers() {
        let mut ns = ToolNamespace::new();
        ns.register("server_a", [tool("read")], ToolConflict::Reject)
            .unwrap();
        let err = ns
            .register("server_b", [tool("read")], ToolConflict::Reject)
            .unwrap_err();
        assert!(err.to_string().contains("冲突"), "{}", err);
        assert_eq!(ns.len(), 1, "冲突工具不应计入注册表");
    }

    /// Reject 策略:不同名工具正常共存。
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

    /// 同一 `full_name` 重复注册无论何种策略都报错。
    #[test]
    fn test_duplicate_registration_rejected() {
        let mut ns = ToolNamespace::new();
        ns.register("s", [tool("a")], ToolConflict::Prefix).unwrap();
        let err = ns
            .register("s", [tool("a")], ToolConflict::Prefix)
            .unwrap_err();
        assert!(err.to_string().contains("重复"), "{}", err);
        assert_eq!(ns.len(), 1);
    }

    /// 冲突策略按 register 调用显式决定:同名工具的来源决定是否冲突。
    #[test]
    fn test_conflict_policy_per_register_call() {
        let mut ns = ToolNamespace::new();
        ns.register("server_a", [tool("read")], ToolConflict::Prefix)
            .unwrap();
        // server_b 的 "read" 与 server_a 同名:Prefix 策略下都暴露。
        let b = ns
            .register("server_b", [tool("read")], ToolConflict::Prefix)
            .unwrap();
        assert_eq!(b[0].full_name, "server_b:read");
        assert_eq!(ns.len(), 2);
    }

    /// 命名空间适配器:LLM 看到 `server:tool`,实际调用剥前缀走原始名。
    ///
    /// 假 SSE 服务器的 `tools/call` 不校验工具名——若 `run` 带了前缀调用
    /// 仍能成功返回(而非未实现/参数错误),说明路由走的是原始名路径。
    #[tokio::test]
    async fn test_namespaced_adapter_routes_to_raw_name() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");

        let adapter = MCPToolAdapter::namespaced(client, "server_a", tool("echo"));
        // BaseTool::name() → 对外名带命名空间前缀。
        assert_eq!(adapter.name(), "server_a:echo");
        assert_eq!(adapter.display_name(), "server_a:echo");
        assert_eq!(adapter.description(), "echo desc");
        assert!(adapter.args_schema().is_some());
        // run 剥掉前缀走原始名 "echo" 调假服务器;假服务器回显收到的工具名,
        // 等于原始名即证明命名空间前缀没有泄漏到 Server 侧。
        let out = adapter.run("{}".into()).await;
        assert!(
            matches!(out.as_deref(), Ok("echo")),
            "应携带原始工具名调用,而非带前缀的对外名,实际: {:?}",
            out.as_deref()
        );
    }

    /// 未命名空间适配器保持原始名,行为与旧版一致。
    #[tokio::test]
    async fn test_plain_adapter_keeps_raw_name() {
        let fake = start_fake_sse_server(PostMode::Quiet).await;
        let client = MCPClient::connect(MCPConfig::sse(&fake.sse_url))
            .await
            .expect("连接假 SSE 服务器应成功");

        let adapter = MCPToolAdapter::new(client, tool("echo"));
        assert_eq!(adapter.name(), "echo");
        assert_eq!(adapter.display_name(), "echo");
    }
}
