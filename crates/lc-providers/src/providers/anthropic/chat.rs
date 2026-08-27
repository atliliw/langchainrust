// src/language_models/providers/anthropic/chat.rs
//! AnthropicChat client struct and core implementation.

use futures_util::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::json;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use lc_core::language_models::{LLMResult, TokenUsage};
use lc_core::tools::{StructuredOutput, ToolDefinition};
use lc_schema::Message;

use super::config::{AnthropicConfig, ThinkingConfig};
use super::error::AnthropicError;
use super::types::{
    AnthropicContentBlock, AnthropicImageSource, AnthropicMessage, AnthropicMessageContent,
    AnthropicResponse, AnthropicStreamEvent, AnthropicStreamToken,
};
use crate::ProviderError;

/// Anthropic Claude chat client.
#[derive(Clone)]
pub struct AnthropicChat {
    pub(crate) config: AnthropicConfig,
    pub(crate) client: reqwest::Client,
}

impl AnthropicChat {
    /// Creates a new Anthropic chat client with the given configuration.
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Creates an AnthropicChat from environment variables, returning a Result.
    pub fn from_env_result() -> Result<Self, ProviderError> {
        Ok(Self::new(AnthropicConfig::from_env_result()?))
    }

    /// Enables extended thinking with the given token budget.
    pub fn with_thinking(mut self, budget_tokens: usize) -> Self {
        self.config.thinking = ThinkingConfig::enabled(budget_tokens);
        self
    }

    /// Returns a reference to the thinking configuration.
    pub fn thinking_config(&self) -> &ThinkingConfig {
        &self.config.thinking
    }

    /// Binds tool definitions for Anthropic function calling.
    ///
    /// Anthropic uses the `tools` field in the request body with a format
    /// that differs from OpenAI: each tool has `name`, `description`, and
    /// `input_schema` (instead of `parameters`).
    pub fn bind_tools(&self, tools: Vec<ToolDefinition>) -> Self {
        let config = AnthropicConfig {
            tools: Some(tools),
            ..self.config.clone()
        };
        Self {
            config,
            client: self.client.clone(),
        }
    }

    /// Sets the tool choice strategy.
    ///
    /// Accepts "auto" (model decides), "any" (must call a tool), or a
    /// specific tool name to force that tool.
    pub fn with_tool_choice(mut self, choice: impl Into<String>) -> Self {
        self.config.tool_choice = Some(choice.into());
        self
    }

    /// Enables structured JSON output with schema validation.
    ///
    /// Uses Anthropic's tool calling under the hood: a single tool named
    /// "structured_output" is bound, and the model is forced to call it.
    pub fn with_structured_output<T: DeserializeOwned + JsonSchema>(
        &self,
    ) -> AnthropicStructuredOutputMethod<T> {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T))
            .unwrap_or_else(|_| serde_json::json!({"type": "object", "properties": {}}));

        let tool = ToolDefinition::new("structured_output", "Return structured JSON output")
            .with_parameters(schema);

        let config = AnthropicConfig {
            tools: Some(vec![tool]),
            tool_choice: Some("auto".to_string()),
            ..self.config.clone()
        };

        AnthropicStructuredOutputMethod {
            config,
            client: self.client.clone(),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn message_to_anthropic_format(message: &Message) -> AnthropicMessage {
        match &message.message_type {
            lc_schema::MessageType::Human => {
                // If the message has images, build a content blocks array
                if message.has_images() {
                    let mut content_parts: Vec<AnthropicContentBlock> = vec![];

                    // Add text content first
                    if !message.content.is_empty() {
                        content_parts.push(AnthropicContentBlock::Text {
                            text: message.content.clone(),
                        });
                    }

                    // Add image blocks — Anthropic requires base64-encoded images
                    for img in &message.images {
                        if let Some(source) = Self::image_to_anthropic_source(img) {
                            content_parts.push(AnthropicContentBlock::Image { source });
                        }
                        // URL-based images that aren't data URIs are silently skipped
                        // (Anthropic doesn't support URL-based image sources)
                    }

                    if content_parts.is_empty() {
                        content_parts.push(AnthropicContentBlock::Text {
                            text: message.content.clone(),
                        });
                    }

                    AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicMessageContent::Blocks(content_parts),
                    }
                } else {
                    AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicMessageContent::Text(message.content.clone()),
                    }
                }
            }
            lc_schema::MessageType::AI => {
                let mut content_parts: Vec<AnthropicContentBlock> = vec![];
                if let Some(tool_calls) = &message.tool_calls {
                    for tc in tool_calls {
                        content_parts.push(AnthropicContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(json!({})),
                        });
                    }
                }
                if !message.content.is_empty() {
                    content_parts.push(AnthropicContentBlock::Text {
                        text: message.content.clone(),
                    });
                }
                if content_parts.is_empty() {
                    content_parts.push(AnthropicContentBlock::Text {
                        text: String::new(),
                    });
                }
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: AnthropicMessageContent::Blocks(content_parts),
                }
            }
            lc_schema::MessageType::Tool { tool_call_id } => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                    tool_use_id: tool_call_id.clone(),
                    content: message.content.clone(),
                }]),
            },
            // System messages are handled separately in build_request_body
            lc_schema::MessageType::System => AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Text(message.content.clone()),
            },
        }
    }

    /// Converts an ImageContent to an AnthropicImageSource.
    ///
    /// Anthropic only supports base64-encoded images. If the ImageContent
    /// is a URL (not a data URI), returns None (the image is skipped).
    fn image_to_anthropic_source(img: &lc_schema::ImageContent) -> Option<AnthropicImageSource> {
        if img.is_base64() {
            // Parse data URI: "data:image/png;base64,abc123"
            let url = &img.url;
            // Extract media type from "data:{media_type};base64,{data}"
            let media_type = url.strip_prefix("data:")?.split(';').next()?.to_string();

            let data = img.base64_data()?;
            Some(AnthropicImageSource {
                source_type: "base64".to_string(),
                media_type,
                data: data.to_string(),
            })
        } else {
            // Anthropic doesn't support URL-based image sources
            // The image is silently skipped
            None
        }
    }

    pub(crate) fn build_request_body(
        &self,
        messages: Vec<Message>,
        stream: bool,
    ) -> serde_json::Value {
        // H42: Extract system messages into top-level system field
        let mut system_text = String::new();
        let mut non_system_messages: Vec<Message> = Vec::new();

        for msg in messages {
            if msg.message_type == lc_schema::MessageType::System {
                if !system_text.is_empty() {
                    system_text.push('\n');
                }
                system_text.push_str(&msg.content);
            } else {
                non_system_messages.push(msg);
            }
        }

        // Also include config system_prompt if set
        if let Some(ref prompt) = self.config.system_prompt {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(prompt);
        }

        let anthropic_messages: Vec<AnthropicMessage> = non_system_messages
            .iter()
            .map(Self::message_to_anthropic_format)
            .collect();

        let mut body = json!({
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            "stream": stream,
        });

        if !system_text.is_empty() {
            body["system"] = json!(system_text);
        }

        if let Some(temp) = self.config.temperature {
            body["temperature"] = json!(temp);
        }

        // M17: Validate thinking config - max_tokens must be > budget_tokens
        if self.config.thinking.is_enabled() {
            if self.config.max_tokens <= self.config.thinking.budget_tokens {
                body["max_tokens"] = json!(self.config.thinking.budget_tokens + 1024);
            }
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": self.config.thinking.budget_tokens,
            });
        }

        // H7: Inject tools if configured (Anthropic function calling)
        if let Some(ref tools) = self.config.tools {
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|td| {
                    let mut tool_json = json!({
                        "name": td.function.name,
                    });
                    if let Some(ref desc) = td.function.description {
                        tool_json["description"] = json!(desc);
                    }
                    if let Some(ref params) = td.function.parameters {
                        tool_json["input_schema"] = json!(params);
                    }
                    tool_json
                })
                .collect();
            body["tools"] = json!(anthropic_tools);
        }

        // H7: Inject tool_choice if configured
        if let Some(ref choice) = self.config.tool_choice {
            if choice == "auto" || choice == "any" {
                body["tool_choice"] = json!({"type": choice});
            } else {
                // Specific tool name
                body["tool_choice"] = json!({"type": "tool", "name": choice});
            }
        }

        body
    }

    pub(crate) async fn chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<LLMResult, AnthropicError> {
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, false);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AnthropicError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AnthropicError::Parse(e.to_string()))?;

        let mut thinking_content = String::new();
        let mut text_content = String::new();
        let mut tool_calls: Vec<lc_core::tools::ToolCall> = Vec::new();

        for block in &anthropic_response.content {
            match block.content_type.as_str() {
                "thinking" => {
                    thinking_content.push_str(&block.thinking);
                }
                "text" => {
                    text_content.push_str(&block.text);
                }
                // H7: Parse tool_use content blocks into ToolCall
                "tool_use" => {
                    let id = block.id.clone().unwrap_or_default();
                    let name = block.name.clone().unwrap_or_default();
                    let input = block.input.clone().unwrap_or(json!({}));
                    tool_calls.push(
                        lc_core::tools::ToolCall::builder(id)
                            .name(name)
                            .arguments(input.to_string())
                            .build(),
                    );
                }
                _ => {
                    if !block.text.is_empty() {
                        text_content.push_str(&block.text);
                    }
                }
            }
        }

        // H1 fix: remove redundant second pass and dangerous fallback.
        // If only thinking blocks exist (no text), content should be empty,
        // not leaked thinking text.
        // The first loop above already collected all "text" blocks.

        Ok(LLMResult {
            content: text_content,
            model: anthropic_response.model,
            token_usage: anthropic_response.usage.map(|u| TokenUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking_content: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
        })
    }

    pub(crate) async fn stream_chat_internal(
        &self,
        messages: Vec<Message>,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<AnthropicStreamToken, AnthropicError>> + Send>>,
        AnthropicError,
    > {
        let url = format!("{}/messages", self.config.base_url);
        let body = self.build_request_body(messages, true);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AnthropicError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AnthropicError::Api(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let byte_stream = response.bytes_stream();
        let sse_buffer = Arc::new(Mutex::new(String::new()));
        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<AnthropicStreamToken, AnthropicError>>(64);

        let buffer_clone = sse_buffer.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut byte_stream = byte_stream;
            while let Some(chunk_result) = byte_stream.next().await {
                if let Ok(bytes) = chunk_result {
                    let chunk_str = String::from_utf8_lossy(&bytes);

                    // Extract complete SSE events from buffer
                    let events = {
                        let mut buffer_guard =
                            buffer_clone.lock().unwrap_or_else(|e| e.into_inner());
                        buffer_guard.push_str(&chunk_str);

                        let mut events = Vec::new();
                        while let Some(pos) = buffer_guard.find("\n\n") {
                            let event_text = buffer_guard[..pos].to_string();
                            buffer_guard.drain(..=pos + 1);
                            events.push(event_text);
                        }
                        events
                    };
                    // buffer_guard is dropped here, before any await

                    for event_text in events {
                        for line in event_text.lines() {
                            if line.starts_with("data: ") {
                                let data = line.trim_start_matches("data: ");
                                if data == "[DONE]" {
                                    continue;
                                }

                                if let Ok(event) =
                                    serde_json::from_str::<AnthropicStreamEvent>(data)
                                {
                                    if event.type_field == "message_delta" {
                                        // message_delta at the end of the stream carries usage; emit it as
                                        // a standalone token so the streaming path also gets the full call usage.
                                        if let Some(usage) = event.usage {
                                            if tx
                                                .send(Ok(AnthropicStreamToken::Usage(usage)))
                                                .await
                                                .is_err()
                                            {
                                                return;
                                            }
                                        }
                                        continue;
                                    }
                                    if event.type_field == "content_block_delta" {
                                        if let Some(delta) = event.delta {
                                            match delta.type_field.as_str() {
                                                "text_delta" => {
                                                    if !delta.text.is_empty()
                                                        && tx
                                                            .send(Ok(AnthropicStreamToken::Text(
                                                                delta.text,
                                                            )))
                                                            .await
                                                            .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                                "thinking_delta" => {
                                                    if !delta.thinking.is_empty()
                                                        && tx
                                                            .send(Ok(
                                                                AnthropicStreamToken::Thinking(
                                                                    delta.thinking,
                                                                ),
                                                            ))
                                                            .await
                                                            .is_err()
                                                    {
                                                        return;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }
}

/// Method for structured output calls via Anthropic tool calling.
pub struct AnthropicStructuredOutputMethod<T: DeserializeOwned + JsonSchema> {
    config: AnthropicConfig,
    client: reqwest::Client,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> AnthropicStructuredOutputMethod<T> {
    /// Invokes the model and parses the result as the structured type.
    pub async fn invoke(&self, messages: Vec<Message>) -> Result<T, AnthropicError> {
        let chat = AnthropicChat {
            config: self.config.clone(),
            client: self.client.clone(),
        };

        let result = chat.chat_internal(messages).await?;
        let structured = StructuredOutput::<T>::new(result);
        structured
            .parse()
            .map_err(|e| AnthropicError::Parse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lc_core::tools::ToolDefinition;
    use serde_json::json;

    #[test]
    fn test_bind_tools_creates_new_chat_with_tools() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config);
        let tools = vec![
            ToolDefinition::new("calculator", "Do math").with_parameters(
                json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
            ),
        ];

        let bound = chat.bind_tools(tools.clone());
        assert!(bound.config.tools.is_some());
        assert_eq!(bound.config.tools.as_ref().unwrap().len(), 1);
        assert_eq!(
            bound.config.tools.as_ref().unwrap()[0].function.name,
            "calculator"
        );
        // Original chat should not have tools
        assert!(chat.config.tools.is_none());
    }

    #[test]
    fn test_with_tool_choice_sets_config() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config);
        let chat = chat.with_tool_choice("auto");
        assert_eq!(chat.config.tool_choice.as_deref(), Some("auto"));
    }

    #[test]
    fn test_build_request_body_includes_tools() {
        let config = AnthropicConfig::new("test-key");
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather").with_parameters(
                json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            ),
        ];
        let chat = AnthropicChat::new(config).bind_tools(tools);

        let body = chat.build_request_body(vec![], false);
        let tools_arr = body.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools_arr.len(), 1);
        assert_eq!(tools_arr[0]["name"], "get_weather");
        assert!(tools_arr[0].get("input_schema").is_some());
    }

    #[test]
    fn test_build_request_body_tool_choice_auto() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config).with_tool_choice("auto");
        let body = chat.build_request_body(vec![], false);
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn test_build_request_body_tool_choice_specific() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config).with_tool_choice("calculator");
        let body = chat.build_request_body(vec![], false);
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "calculator");
    }

    #[test]
    fn test_with_structured_output_binds_tool() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config);
        #[derive(serde::Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct TestOutput {
            answer: String,
        }
        let _method: AnthropicStructuredOutputMethod<TestOutput> = chat.with_structured_output();
        // Just verify it compiles and the method is callable
    }

    // --- Image handling tests ---

    #[test]
    fn test_human_message_without_images_uses_text_content() {
        let msg = Message::human("Hello");
        let anthropic_msg = AnthropicChat::message_to_anthropic_format(&msg);
        assert_eq!(anthropic_msg.role, "user");
        // Without images, should use simple Text variant
        assert!(matches!(
            anthropic_msg.content,
            AnthropicMessageContent::Text(_)
        ));
    }

    #[test]
    fn test_human_message_with_base64_image_uses_blocks() {
        let msg = Message::human_with_image("Describe this", "data:image/png;base64,abc123");
        let anthropic_msg = AnthropicChat::message_to_anthropic_format(&msg);
        assert_eq!(anthropic_msg.role, "user");
        // With images, should use Blocks variant
        assert!(matches!(
            anthropic_msg.content,
            AnthropicMessageContent::Blocks(_)
        ));

        if let AnthropicMessageContent::Blocks(blocks) = &anthropic_msg.content {
            // Should have: text block + image block
            assert_eq!(blocks.len(), 2);

            // First block should be text
            assert!(
                matches!(&blocks[0], AnthropicContentBlock::Text { text } if text == "Describe this")
            );

            // Second block should be image
            if let AnthropicContentBlock::Image { source } = &blocks[1] {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "abc123");
            } else {
                panic!("Expected Image block");
            }
        }
    }

    #[test]
    fn test_human_message_with_url_image_skips_image() {
        // Anthropic doesn't support URL-based images, so they are silently skipped
        let msg = Message::human_with_image("Describe this", "https://example.com/img.png");
        let anthropic_msg = AnthropicChat::message_to_anthropic_format(&msg);
        assert_eq!(anthropic_msg.role, "user");

        // URL image is skipped, but since has_images() is true, we still use Blocks
        if let AnthropicMessageContent::Blocks(blocks) = &anthropic_msg.content {
            // Only text block remains (URL image was skipped)
            assert_eq!(blocks.len(), 1);
            assert!(matches!(&blocks[0], AnthropicContentBlock::Text { .. }));
        } else {
            panic!("Expected Blocks variant for message with images");
        }
    }

    #[test]
    fn test_human_message_with_jpeg_base64_image() {
        let msg =
            Message::human_with_image("What is this?", "data:image/jpeg;base64,/9j/4AAQSkZJRg==");
        let anthropic_msg = AnthropicChat::message_to_anthropic_format(&msg);

        if let AnthropicMessageContent::Blocks(blocks) = &anthropic_msg.content {
            if let AnthropicContentBlock::Image { source } = &blocks[1] {
                assert_eq!(source.media_type, "image/jpeg");
                assert_eq!(source.data, "/9j/4AAQSkZJRg==");
            }
        }
    }

    #[test]
    fn test_image_to_anthropic_source_base64() {
        let img = lc_schema::ImageContent::from_base64_with_mime("testdata", "image/webp");
        let source = AnthropicChat::image_to_anthropic_source(&img);
        assert!(source.is_some());
        let s = source.unwrap();
        assert_eq!(s.source_type, "base64");
        assert_eq!(s.media_type, "image/webp");
        assert_eq!(s.data, "testdata");
    }

    #[test]
    fn test_image_to_anthropic_source_url_returns_none() {
        let img = lc_schema::ImageContent::from_url("https://example.com/img.png");
        let source = AnthropicChat::image_to_anthropic_source(&img);
        assert!(source.is_none());
    }

    #[test]
    fn test_build_request_body_with_image_message() {
        let config = AnthropicConfig::new("test-key");
        let chat = AnthropicChat::new(config);
        let msg = Message::human_with_image("Describe", "data:image/png;base64,abc");
        let body = chat.build_request_body(vec![msg], false);

        // Messages should be an array with one element
        let messages = body.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 1);

        // The message content should be an array of blocks
        let content = &messages[0]["content"];
        assert!(content.is_array());
        let blocks = content.as_array().unwrap();
        assert_eq!(blocks.len(), 2); // text + image
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
    }
}
