//! MCP Resources - 资源类型与 `resources/list`、`resources/read` 处理
//!
//! MCP Resources 允许 Server 暴露结构化数据(文件、数据库记录等)供 Client 读取。

use crate::protocol::MCPError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// 资源描述(来自 `resources/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// 资源 URI
    pub uri: String,
    /// 资源名称
    pub name: String,
    /// 资源的可选描述
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 资源的 MIME 类型(可选)
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// 资源内容(来自 `resources/read`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// 资源 URI
    pub uri: String,
    /// 内容的 MIME 类型(可选)
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// 文本内容(与 `blob` 二选一)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// 二进制内容(base64 编码,与 `text` 二选一)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// `resources/list` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    /// 资源列表
    pub resources: Vec<Resource>,
}

/// `resources/read` 请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    /// 要读取的资源 URI
    pub uri: String,
}

/// `resources/read` 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    /// 读取到的资源内容列表
    pub contents: Vec<ResourceContent>,
}

/// 资源提供者:server 注册后,`resources/list` / `resources/read` 才有数据源。
///
/// 未注册时对应方法仍返回 `method_not_found`(诚实边界,不假装支持)。
#[async_trait]
pub trait ResourceProvider: Send + Sync {
    /// 返回全部资源列表。
    async fn list_resources(&self) -> Result<Vec<Resource>, MCPError>;
    /// 按 URI 读取资源内容。
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
