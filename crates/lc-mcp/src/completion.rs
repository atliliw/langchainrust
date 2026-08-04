//! MCP Completion - 补全类型与 `completion/complete` 处理
//!
//! MCP Completion 允许 Server 为提示词参数或资源 URI 提供自动补全建议。

use serde::{Deserialize, Serialize};

/// 补全引用类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRef {
    #[serde(rename = "type")]
    pub ref_type: String,
    pub uri: String,
}

/// 补全请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionArgument {
    pub name: String,
    pub value: String,
}

/// `completion/complete` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub reference: CompletionRef,
    pub argument: CompletionArgument,
}

/// 补全值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionValue {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// `completion/complete` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    pub values: Vec<CompletionValue>,
    #[serde(default)]
    pub total: Option<usize>,
    #[serde(rename = "hasMore", default)]
    pub has_more: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_ref_serialization() {
        let ref_val = CompletionRef {
            ref_type: "ref/prompt".to_string(),
            uri: "prompt://code_review".to_string(),
        };
        let json = serde_json::to_string(&ref_val).unwrap();
        assert!(json.contains("\"type\":\"ref/prompt\""));
        assert!(json.contains("\"uri\":\"prompt://code_review\""));
    }

    #[test]
    fn test_completion_argument_serialization() {
        let arg = CompletionArgument {
            name: "language".to_string(),
            value: "ru".to_string(),
        };
        let json = serde_json::to_string(&arg).unwrap();
        assert!(json.contains("\"name\":\"language\""));
        assert!(json.contains("\"value\":\"ru\""));
    }

    #[test]
    fn test_completion_request_serialization() {
        let req = CompletionRequest {
            reference: CompletionRef {
                ref_type: "ref/prompt".to_string(),
                uri: "prompt://code_review".to_string(),
            },
            argument: CompletionArgument {
                name: "language".to_string(),
                value: "ru".to_string(),
            },
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"reference\""));
        assert!(json.contains("\"argument\""));
    }

    #[test]
    fn test_completion_value_with_description() {
        let val = CompletionValue {
            label: "rust".to_string(),
            description: Some("Rust programming language".to_string()),
        };
        let json = serde_json::to_string(&val).unwrap();
        assert!(json.contains("\"label\":\"rust\""));
        assert!(json.contains("\"description\""));
    }

    #[test]
    fn test_completion_value_without_description() {
        let val = CompletionValue {
            label: "python".to_string(),
            description: None,
        };
        let json = serde_json::to_string(&val).unwrap();
        assert!(!json.contains("description"));
    }

    #[test]
    fn test_completion_result_deserialization() {
        let json = r#"{"values":[{"label":"rust"},{"label":"ruby"}],"total":2,"hasMore":false}"#;
        let result: CompletionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.values.len(), 2);
        assert_eq!(result.values[0].label, "rust");
        assert_eq!(result.total, Some(2));
        assert!(!result.has_more);
    }

    #[test]
    fn test_completion_result_defaults() {
        let json = r#"{"values":[{"label":"rust"}]}"#;
        let result: CompletionResult = serde_json::from_str(json).unwrap();
        assert!(result.total.is_none());
        assert!(!result.has_more);
    }
}
