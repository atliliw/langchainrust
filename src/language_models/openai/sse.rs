// src/language_models/openai/sse.rs
//! SSE (Server-Sent Events) 解析器
//!
//! 用于解析 OpenAI 流式响应

use serde::Deserialize;

/// SSE 解析器
pub struct SSEParser {
    buffer: String,
}

impl SSEParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// 解析 SSE 数据块
    ///
    /// # 参数
    /// * `chunk` - 接收到的数据块
    ///
    /// # 返回
    /// 完整的事件列表
    pub fn parse(&mut self, chunk: &str) -> Vec<SSEEvent> {
        self.buffer.push_str(chunk);

        let mut events = Vec::new();

        // SSE 事件以双换行分隔
        while let Some(pos) = self.buffer.find("\n\n") {
            let event_text = self.buffer[..pos].to_string();
            self.buffer.drain(..=pos + 1);

            if let Some(event) = self.parse_event(&event_text) {
                events.push(event);
            }
        }

        events
    }

    /// 解析单个 SSE 事件
    fn parse_event(&self, text: &str) -> Option<SSEEvent> {
        let mut event_type = None;
        let mut data = None;

        for line in text.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                event_type = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                data = Some(value.trim().to_string());
            }
        }

        // 如果只有 data 字段，也算有效事件
        if data.is_some() {
            Some(SSEEvent {
                event: event_type,
                data: data?,
            })
        } else {
            None
        }
    }
}

impl Default for SSEParser {
    fn default() -> Self {
        Self::new()
    }
}

/// SSE 事件
#[derive(Debug, Clone)]
pub struct SSEEvent {
    /// 事件类型
    pub event: Option<String>,

    /// 事件数据
    pub data: String,
}

impl SSEEvent {
    /// 检查是否为结束事件
    pub fn is_done(&self) -> bool {
        self.data == "[DONE]"
    }

    /// 解析 OpenAI 流式响应数据
    pub fn parse_openai_chunk(&self) -> Result<Option<OpenAIStreamChunk>, serde_json::Error> {
        if self.is_done() {
            return Ok(None);
        }

        let chunk: OpenAIStreamChunk = serde_json::from_str(&self.data)?;
        Ok(Some(chunk))
    }
}

/// OpenAI 流式响应块
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub index: i32,
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_parser() {
        let mut parser = SSEParser::new();

        let chunk = "data: {\"test\": \"value\"}\n\n";
        let events = parser.parse(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"test\": \"value\"}");
    }

    #[test]
    fn test_sse_done_event() {
        let mut parser = SSEParser::new();

        let chunk = "data: [DONE]\n\n";
        let events = parser.parse(chunk);

        assert_eq!(events.len(), 1);
        assert!(events[0].is_done());
    }

    #[test]
    fn test_openai_chunk_parsing() {
        let event = SSEEvent {
            event: None,
            data: r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#.to_string(),
        };

        let chunk = event.parse_openai_chunk().unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }
}
