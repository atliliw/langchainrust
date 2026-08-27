//! MCP Roots - root-directory types and `roots/list` handling
//!
//! MCP Roots lets a Host declare accessible root directories (workspaces) to a Server,
//! so the Server understands the filesystem boundaries.

use serde::{Deserialize, Serialize};

/// Root-directory description (from `roots/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Root {
    /// Root-directory URI
    pub uri: String,
    /// Optional name of the root directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `roots/list` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListRootsResult {
    /// List of root directories
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
