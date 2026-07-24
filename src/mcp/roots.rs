//! MCP Roots - 根目录类型与 `roots/list` 处理
//!
//! MCP Roots 允许 Host 向 Server 声明可访问的根目录(工作区),
//! Server 可据此了解文件系统边界。

use serde::{Deserialize, Serialize};

/// 根目录描述(来自 `roots/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Root {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `roots/list` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRootsResult {
    pub roots: Vec<Root>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_with_name() {
        let root = Root {
            uri: "file:///home/user/project".to_string(),
            name: Some("My Project".to_string()),
        };
        let json = serde_json::to_string(&root).unwrap();
        assert!(json.contains("\"uri\":\"file:///home/user/project\""));
        assert!(json.contains("\"name\":\"My Project\""));
    }

    #[test]
    fn test_root_without_name() {
        let root = Root {
            uri: "file:///tmp".to_string(),
            name: None,
        };
        let json = serde_json::to_string(&root).unwrap();
        assert!(json.contains("\"uri\":\"file:///tmp\""));
        assert!(!json.contains("name"));
    }

    #[test]
    fn test_list_roots_result_deserialization() {
        let json = r#"{"roots":[{"uri":"file:///a","name":"A"},{"uri":"file:///b"}]}"#;
        let result: ListRootsResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.roots.len(), 2);
        assert_eq!(result.roots[0].uri, "file:///a");
        assert_eq!(result.roots[0].name.as_deref(), Some("A"));
        assert!(result.roots[1].name.is_none());
    }

    #[test]
    fn test_list_roots_result_empty() {
        let json = r#"{"roots":[]}"#;
        let result: ListRootsResult = serde_json::from_str(json).unwrap();
        assert!(result.roots.is_empty());
    }
}
