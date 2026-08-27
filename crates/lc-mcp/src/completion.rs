//! MCP Completion - completion types and `completion/complete` handling
//!
//! MCP Completion lets a Server provide autocomplete suggestions for prompt arguments or resource URIs.

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A completion reference type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRef {
    /// Reference type (`ref/prompt` or `ref/resource`)
    #[serde(rename = "type")]
    pub ref_type: String,
    /// URI of the object being completed
    pub uri: String,
}

/// Completion request argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionArgument {
    /// Argument name
    pub name: String,
    /// Current argument value (used to provide completion suggestions)
    pub value: String,
}

/// `completion/complete` request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The reference being completed
    pub reference: CompletionRef,
    /// The completion argument
    pub argument: CompletionArgument,
}

/// A completion value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionValue {
    /// Display label of the completion suggestion
    pub label: String,
    /// Optional description of the completion suggestion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// `completion/complete` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    /// List of completion suggestions
    pub values: Vec<CompletionValue>,
    /// Optional total number of completions
    #[serde(default)]
    pub total: Option<usize>,
    /// Whether more completion suggestions are available
    #[serde(rename = "hasMore", default)]
    pub has_more: bool,
}

/// Completion provider: once a server registers, `completion/complete` has a data source.
///
/// When nothing is registered the method still returns `method_not_found` (an honest boundary, no pretending to support it).
#[async_trait]
pub trait CompletionProvider: Send + Sync {
    /// Returns the suggestion list for a completion request.
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResult, MCPError>;
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
