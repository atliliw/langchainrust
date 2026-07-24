//! MCP 类型定义:工具、内容、配置

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// MCP 工具定义(来自 `tools/list`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// 工具参数的 JSON Schema
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP 工具调用结果(来自 `tools/call`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolResult {
    pub content: Vec<MCPContent>,
    #[serde(default)]
    pub is_error: bool,
}

/// MCP 内容类型( tagged enum,`type` 字段区分)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MCPContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { uri: String, name: String },
}

impl MCPContent {
    /// 若为文本内容,返回文本引用
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MCPContent::Text { text } => Some(text),
            _ => None,
        }
    }
}

impl MCPToolResult {
    /// 提取所有文本内容,用换行连接
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text().map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// MCP Client 配置
#[derive(Debug, Clone)]
pub enum MCPConfig {
    /// Stdio 传输:启动子进程,通过 stdin/stdout 通信
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// SSE 传输:HTTP Server-Sent Events
    Sse { url: String },
}

impl MCPConfig {
    /// 创建 Stdio 配置(默认空环境变量)
    pub fn stdio(command: impl Into<String>, args: Vec<String>) -> Self {
        Self::Stdio {
            command: command.into(),
            args,
            env: HashMap::new(),
        }
    }

    /// 创建 SSE 配置
    pub fn sse(url: impl Into<String>) -> Self {
        Self::Sse { url: url.into() }
    }

    /// 追加环境变量(仅 Stdio 生效)
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let MCPConfig::Stdio { env, .. } = &mut self {
            env.insert(key.into(), value.into());
        }
        self
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
    fn test_config_stdio() {
        let config = MCPConfig::stdio(
            "npx",
            vec![
                "@anthropic/mcp-server-filesystem".to_string(),
                "/tmp".to_string(),
            ],
        );
        assert!(matches!(config, MCPConfig::Stdio { .. }));
    }

    #[test]
    fn test_config_sse() {
        let config = MCPConfig::sse("http://localhost:3001/sse");
        assert!(matches!(config, MCPConfig::Sse { .. }));
    }

    #[test]
    fn test_config_with_env() {
        let config = MCPConfig::stdio("npx", vec![]).with_env("API_KEY", "secret");
        if let MCPConfig::Stdio { env, .. } = config {
            assert_eq!(env.get("API_KEY"), Some(&"secret".to_string()));
        } else {
            panic!("应为 Stdio");
        }
    }
}
