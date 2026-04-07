// src/core/tools/registry.rs
//! 工具注册表
//!
//! 管理多个工具，提供统一的查找和执行接口。

use super::BaseTool;
use std::collections::HashMap;
use std::sync::Arc;

/// 工具注册表
///
/// 存储和管理多个工具实例，提供按名称查找和执行功能。
/// # 示例
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
    /// 工具存储（按名称索引）
    tools: HashMap<String, Arc<dyn BaseTool>>,
}

impl ToolRegistry {
    /// 创建空的工具注册表
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// 注册工具
    ///
    /// # 参数
    /// * `tool` - 要注册的工具（包装在 Arc 中以便共享）
    ///
    /// # 返回
    /// 如果已有同名工具，返回旧的工具；否则返回 None。
    pub fn register(&mut self, tool: Arc<dyn BaseTool>) -> Option<Arc<dyn BaseTool>> {
        let name = tool.name().to_string();
        self.tools.insert(name, tool)
    }

    /// 获取工具
    ///
    /// # 参数
    /// * `name` - 工具名称
    ///
    /// # 返回
    /// 如果找到工具，返回工具引用；否则返回 None。
    pub fn get(&self, name: &str) -> Option<&Arc<dyn BaseTool>> {
        self.tools.get(name)
    }

    /// 获取所有工具名称
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s: &String| s.as_str()).collect()
    }

    /// 获取所有工具
    pub fn tools(&self) -> Vec<&Arc<dyn BaseTool>> {
        self.tools.values().collect()
    }

    /// 工具数量
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// 移除工具
    ///
    /// # 参数
    /// * `name` - 工具名称
    ///
    /// # 返回
    /// 如果找到并移除工具，返回该工具；否则返回 None。
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn BaseTool>> {
        self.tools.remove(name)
    }

    /// 检查是否包含指定工具
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// 生成工具描述
    ///
    /// 用于向 LLM 展示可用工具列表。
    pub fn describe_tools(&self) -> String {
        if self.tools.is_empty() {
            return "没有可用工具".to_string();
        }

        let mut description = String::from("可用工具:\n");

        for (name, tool) in &self.tools {
            description.push_str(&format!("- {}: {}\n", name, tool.description()));

            // 添加输入格式说明
            if let Some(schema) = tool.args_schema() {
                if let Some(props) = schema.get("properties") {
                    description.push_str("  输入参数:\n");
                    if let Some(obj) = props.as_object() {
                        for (prop_name, prop_value) in obj {
                            let prop_desc = prop_value
                                .get("description")
                                .and_then(|d: &serde_json::Value| d.as_str())
                                .unwrap_or("无描述");
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
