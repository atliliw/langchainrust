//! MCP Resources - 资源类型与 `resources/list`、`resources/read` 处理
//!
//! MCP Resources 允许 Server 暴露结构化数据(文件、数据库记录等)供 Client 读取。

use serde::{Deserialize, Serialize};

/// 资源描述(来自 `resources/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// 资源内容(来自 `resources/read`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    pub uri: String,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// `resources/list` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    pub resources: Vec<Resource>,
}

/// `resources/read` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    pub uri: String,
}

/// `resources/read` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContent>,
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
