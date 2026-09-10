//! MCP type definitions: tools, content, configuration

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP tool definition (from `tools/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    #[serde(default)]
    pub description: String,
    /// JSON Schema of the tool arguments
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP tool-call result (from `tools/call`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolResult {
    /// The content list returned by the tool
    pub content: Vec<MCPContent>,
    /// Whether this is an error result
    #[serde(default)]
    pub is_error: bool,
}

/// MCP content type (a tagged enum, distinguished by the `type` field)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MCPContent {
    /// Text content
    #[serde(rename = "text")]
    Text {
        /// Text data
        text: String,
    },
    /// Image content
    #[serde(rename = "image")]
    Image {
        /// Image data (base64-encoded)
        data: String,
        /// Image MIME type
        mime_type: String,
    },
    /// Resource-reference content
    #[serde(rename = "resource")]
    Resource {
        /// Resource URI
        uri: String,
        /// Resource name
        name: String,
    },
}

impl MCPContent {
    /// Returns a text reference when this is text content
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MCPContent::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Renders as text (P1-7): text stays as-is; images / resources are represented by a placeholder
    /// description, not silently dropped.
    ///
    /// `BaseTool` only accepts String output; multi-type content (image/resource) cannot be expressed
    /// directly, so a placeholder string carrying metadata tells the upper layer "there is a non-text content
    /// here" instead of quietly swallowing it.
    pub fn render_text(&self) -> String {
        match self {
            MCPContent::Text { text } => text.clone(),
            MCPContent::Image { mime_type, .. } => {
                format!("[image: {} (base64 数据已省略)]", mime_type)
            }
            MCPContent::Resource { uri, name } => {
                format!("[resource: {} ({})]", name, uri)
            }
        }
    }
}

impl MCPToolResult {
    /// Renders all content as text, joined by newlines (P1-7).
    ///
    /// Non-text content such as images / resources is represented by a placeholder description, no longer
    /// silently dropped.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .map(|c| c.render_text())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_deserialization() {
        let json = r#"{"name":"read_file","description":"Read a file","inputSchema":{"type":"object","properties":{"path":{"type":"string"}}}}"#;
        let tool: MCPToolDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description, "Read a file");
        assert_eq!(tool.input_schema["type"], "object");
    }

    #[test]
    fn test_content_text_tagged_enum() {
        let json = r#"{"type":"text","text":"hello"}"#;
        let content: MCPContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.as_text(), Some("hello"));
    }

    #[test]
    fn test_content_image_tagged_enum() {
        let json = r#"{"type":"image","data":"base64...","mime_type":"image/png"}"#;
        let content: MCPContent = serde_json::from_str(json).unwrap();
        assert!(content.as_text().is_none());
    }

    #[test]
    fn test_tool_result_text_extraction() {
        let result = MCPToolResult {
            content: vec![
                MCPContent::Text {
                    text: "line1".to_string(),
                },
                MCPContent::Text {
                    text: "line2".to_string(),
                },
            ],
            is_error: false,
        };
        assert_eq!(result.text(), "line1\nline2");
    }

    #[test]
    fn test_tool_result_is_error_default() {
        let json = r#"{"content":[{"type":"text","text":"ok"}]}"#;
        let result: MCPToolResult = serde_json::from_str(json).unwrap();
        assert!(!result.is_error);
    }

    #[test]
    fn test_render_image_placeholder_not_dropped() {
        // P1-7: image content is represented by a placeholder description, no longer silently dropped.
        let content = MCPContent::Image {
            data: "base64...".to_string(),
            mime_type: "image/png".to_string(),
        };
        let text = content.render_text();
        assert!(text.contains("[image: image/png"));
        assert!(text.contains("base64"));
    }

    #[test]
    fn test_render_resource_placeholder_not_dropped() {
        // P1-7: resource content is represented by a placeholder description.
        let content = MCPContent::Resource {
            uri: "file:///tmp/x.json".to_string(),
            name: "x.json".to_string(),
        };
        let text = content.render_text();
        assert!(text.contains("[resource: x.json (file:///tmp/x.json)]"));
    }

    #[test]
    fn test_tool_result_text_mixed_content() {
        // Text + image mixed: text is kept, the image degrades to a placeholder, both go into text().
        let result = MCPToolResult {
            content: vec![
                MCPContent::Text {
                    text: "title".to_string(),
                },
                MCPContent::Image {
                    data: "d".to_string(),
                    mime_type: "image/jpeg".to_string(),
                },
            ],
            is_error: false,
        };
        let text = result.text();
        assert!(text.contains("title"));
        assert!(text.contains("[image: image/jpeg"));
    }
}
