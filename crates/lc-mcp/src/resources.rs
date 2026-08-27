//! MCP Resources - resource types and `resources/list` / `resources/read` handling
//!
//! MCP Resources lets a Server expose structured data (files, database records, etc.) for a Client to read.

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Resource description (from `resources/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource URI
    pub uri: String,
    /// Resource name
    pub name: String,
    /// Optional description of the resource
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Resource MIME type (optional)
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Resource content (from `resources/read`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource URI
    pub uri: String,
    /// Content MIME type (optional)
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text content (mutually exclusive with `blob`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Binary content (base64-encoded, mutually exclusive with `text`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// `resources/list` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    /// List of resources
    pub resources: Vec<Resource>,
}

/// `resources/read` request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    /// URI of the resource to read
    pub uri: String,
}

/// `resources/read` response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    /// List of resource contents that were read
    pub contents: Vec<ResourceContent>,
}

/// Resource provider: once a server registers, `resources/list` / `resources/read` have a data source.
///
/// When nothing is registered the methods still return `method_not_found` (an honest boundary, no pretending to support it).
#[async_trait]
pub trait ResourceProvider: Send + Sync {
    /// Returns the full list of resources.
    async fn list_resources(&self) -> Result<Vec<Resource>, MCPError>;
    /// Reads the resource content by URI.
    async fn read_resource(&self, uri: &str) -> Result<Vec<ResourceContent>, MCPError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_serialization() {
        let resource = Resource {
            uri: "file:///tmp/test.txt".to_string(),
            name: "test.txt".to_string(),
            description: Some("A test file".to_string()),
            mime_type: Some("text/plain".to_string()),
        };
        let json = serde_json::to_string(&resource).unwrap();
        assert!(json.contains("\"uri\":\"file:///tmp/test.txt\""));
        assert!(json.contains("\"name\":\"test.txt\""));
        assert!(json.contains("\"mimeType\":\"text/plain\""));
    }

    #[test]
    fn test_resource_optional_fields_skipped() {
        let resource = Resource {
            uri: "file:///tmp/test.txt".to_string(),
            name: "test.txt".to_string(),
            description: None,
            mime_type: None,
        };
        let json = serde_json::to_string(&resource).unwrap();
        assert!(!json.contains("description"));
        assert!(!json.contains("mimeType"));
    }

    #[test]
    fn test_resource_content_text() {
        let content = ResourceContent {
            uri: "file:///tmp/test.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            text: Some("hello world".to_string()),
            blob: None,
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"text\":\"hello world\""));
        assert!(!json.contains("blob"));
    }

    #[test]
    fn test_resource_content_blob() {
        let content = ResourceContent {
            uri: "file:///tmp/image.png".to_string(),
            mime_type: Some("image/png".to_string()),
            text: None,
            blob: Some("base64data".to_string()),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"blob\":\"base64data\""));
        assert!(!json.contains("text"));
    }

    #[test]
    fn test_list_resources_result_deserialization() {
        let json = r#"{"resources":[{"uri":"file:///tmp/a.txt","name":"a.txt"}]}"#;
        let result: ListResourcesResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.resources.len(), 1);
        assert_eq!(result.resources[0].uri, "file:///tmp/a.txt");
    }

    #[test]
    fn test_read_resource_params_serialization() {
        let params = ReadResourceParams {
            uri: "file:///tmp/test.txt".to_string(),
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"uri\":\"file:///tmp/test.txt\""));
    }

    #[test]
    fn test_read_resource_result_deserialization() {
        let json = r#"{"contents":[{"uri":"file:///tmp/test.txt","text":"hello"}]}"#;
        let result: ReadResourceResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.contents.len(), 1);
        assert_eq!(result.contents[0].text.as_deref(), Some("hello"));
    }
}
