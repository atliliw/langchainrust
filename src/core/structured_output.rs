// src/core/structured_output.rs
//! Structured output utilities for extracting typed data from LLM responses.
//!
//! This module provides a provider-agnostic `with_structured_output` function
//! that works with any `BaseChatModel` implementation. It injects a JSON schema
//! into the prompt, calls the LLM, and parses the JSON response into the
//! target type `T`.
//!
//! # Strategy
//!
//! The default (generic) strategy uses **prompt injection**: the JSON schema
//! and format instructions are embedded in the system prompt, and the
//! `JsonOutputParser` is used to extract JSON from the response.
//!
//! Provider-specific implementations (OpenAI function calling, Ollama JSON mode)
//! are available on the concrete types directly (e.g., `OpenAIChat::with_structured_output`).
//!
//! # Example
//!
//! ```ignore
//! use serde::{Deserialize, Serialize};
//! use langchainrust::core::structured_output::{with_structured_output, StructuredOutputError};
//! use langchainrust::{OpenAIChat, OpenAIConfig, Message};
//!
//! #[derive(Debug, Deserialize, Serialize)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! let llm = OpenAIChat::new(OpenAIConfig::default());
//! let schema = serde_json::json!({
//!     "type": "object",
//!     "properties": {
//!         "name": {"type": "string"},
//!         "age": {"type": "integer"}
//!     },
//!     "required": ["name", "age"]
//! });
//!
//! let person: Person = with_structured_output(&llm, schema, "Tell me about Alice who is 30").await?;
//! ```

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::core::language_models::{BaseChatModel, LLMResult};
use crate::core::output_parsers::BaseOutputParser;
use crate::core::output_parsers::JsonOutputParser;
use crate::schema::Message;

/// Errors that can occur during structured output extraction.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StructuredOutputError {
    /// The provided JSON schema is invalid or malformed.
    #[error("Schema error: {0}")]
    SchemaError(String),

    /// The LLM response could not be parsed as the target type.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// The provider does not support the requested structured output method.
    #[error("Provider unsupported: {0}")]
    ProviderUnsupported(String),

    /// The LLM call itself failed.
    #[error("LLM error: {0}")]
    LLMError(String),
}

/// Trait that extends `BaseChatModel` with structured output capabilities.
///
/// Implementors can override the default prompt-injection strategy with
/// provider-specific mechanisms (e.g., OpenAI function calling, Ollama JSON mode).
#[async_trait]
pub trait StructuredOutputExt: BaseChatModel {
    /// Call the LLM with a JSON schema and prompt, returning a parsed result of type `T`.
    ///
    /// The default implementation uses prompt injection: it embeds the schema
    /// into the system prompt and parses the JSON response with `JsonOutputParser`.
    ///
    /// # Arguments
    ///
    /// * `schema` - A JSON Schema (`serde_json::Value`) describing the expected output shape.
    /// * `prompt` - The user prompt / question to send to the LLM.
    ///
    /// # Returns
    ///
    /// A `Result<T, StructuredOutputError>` where `T` is the deserialized output.
    async fn with_structured_output<T: DeserializeOwned + Serialize + Send + Sync + 'static>(
        &self,
        schema: Value,
        prompt: &str,
    ) -> Result<T, StructuredOutputError> {
        with_structured_output(self, schema, prompt).await
    }
}

/// Blanket implementation: every `BaseChatModel` automatically gets `StructuredOutputExt`.
impl<M: BaseChatModel> StructuredOutputExt for M {}

/// Standalone function to extract structured output from any `BaseChatModel`.
///
/// This is the core implementation that works with any chat model by:
/// 1. Building a system prompt that includes the JSON schema and format instructions
/// 2. Calling `llm.chat()` with the combined messages
/// 3. Parsing the LLM's JSON response into the target type `T`
///
/// # Arguments
///
/// * `llm` - Any type implementing `BaseChatModel`.
/// * `schema` - A JSON Schema describing the expected output.
/// * `prompt` - The user prompt to send to the LLM.
///
/// # Returns
///
/// A `Result<T, StructuredOutputError>` where `T` is the deserialized output.
///
/// # Errors
///
/// - `StructuredOutputError::SchemaError` if the schema is not a valid JSON object.
/// - `StructuredOutputError::LLMError` if the underlying `chat()` call fails.
/// - `StructuredOutputError::ParseError` if the response cannot be parsed as JSON
///   or deserialized into type `T`.
pub async fn with_structured_output<T, M>(
    llm: &M,
    schema: Value,
    prompt: &str,
) -> Result<T, StructuredOutputError>
where
    T: DeserializeOwned + Serialize + Send + Sync + 'static,
    M: BaseChatModel + ?Sized,
{
    // Validate the schema is an object
    if !schema.is_object() {
        return Err(StructuredOutputError::SchemaError(format!(
            "Schema must be a JSON object, got: {}",
            schema
        )));
    }

    // Build the system prompt with schema and format instructions
    let system_prompt = build_structured_system_prompt(&schema);

    let messages = vec![
        Message::system(system_prompt),
        Message::human(prompt),
    ];

    // Call the LLM
    let result: LLMResult = llm
        .chat(messages, None)
        .await
        .map_err(|e| StructuredOutputError::LLMError(e.to_string()))?;

    // Parse the response
    parse_structured_response::<T>(&result.content).await
}

/// Build a system prompt that instructs the LLM to output JSON conforming to the schema.
fn build_structured_system_prompt(schema: &Value) -> String {
    let schema_str = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());

    format!(
        "You are a helpful assistant that responds exclusively in valid JSON format.\n\
         \n\
         You must respond with a JSON object that conforms to the following JSON Schema:\n\
         ```json\n\
         {schema_str}\n\
         ```\n\
         \n\
         Important rules:\n\
         1. Respond ONLY with valid JSON. Do not include any explanatory text before or after the JSON.\n\
         2. The JSON must conform exactly to the schema above.\n\
         3. All required fields must be present.\n\
         4. Do not include fields that are not in the schema.\n\
         5. If you cannot satisfy the schema, respond with the closest valid JSON you can produce."
    )
}

/// Parse the LLM response content into the target type `T`.
///
/// Uses `JsonOutputParser` to handle markdown code blocks and other common
/// LLM output formatting, then deserializes into `T`.
async fn parse_structured_response<T: DeserializeOwned + Serialize + Send + Sync + 'static>(
    content: &str,
) -> Result<T, StructuredOutputError> {
    let parser = JsonOutputParser::new();

    let json_value: Value = parser
        .parse(content)
        .await
        .map_err(|e| StructuredOutputError::ParseError(format!(
            "Failed to parse LLM response as JSON: {}",
            e
        )))?;

    serde_json::from_value::<T>(json_value).map_err(|e| StructuredOutputError::ParseError(format!(
        "Failed to deserialize JSON into target type: {}. Response was: {}",
        e,
        &content[..std::cmp::min(200, content.len())]
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use std::pin::Pin;

    use crate::core::language_models::{BaseLanguageModel, LLMResult};
    use crate::core::runnables::Runnable;
    use crate::RunnableConfig;

    /// A mock chat model that returns a predefined JSON response.
    struct MockChatModel {
        response: String,
    }

    impl MockChatModel {
        fn new(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
            }
        }
    }

    #[async_trait]
    impl Runnable<Vec<Message>, LLMResult> for MockChatModel {
        type Error = MockError;

        async fn invoke(
            &self,
            _input: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.response.clone(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
            })
        }
    }

    #[async_trait]
    impl BaseLanguageModel<Vec<Message>, LLMResult> for MockChatModel {
        fn model_name(&self) -> &str {
            "mock"
        }

        fn get_num_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }

        fn temperature(&self) -> Option<f32> {
            None
        }

        fn max_tokens(&self) -> Option<usize> {
            None
        }

        fn with_temperature(self, _temp: f32) -> Self
        where
            Self: Sized,
        {
            self
        }

        fn with_max_tokens(self, _max: usize) -> Self
        where
            Self: Sized,
        {
            self
        }
    }

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        async fn chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<LLMResult, Self::Error> {
            Ok(LLMResult {
                content: self.response.clone(),
                model: "mock".to_string(),
                token_usage: None,
                tool_calls: None,
            })
        }

        async fn stream_chat(
            &self,
            _messages: Vec<Message>,
            _config: Option<RunnableConfig>,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
            let content = self.response.clone();
            let stream = futures_util::stream::once(async move { Ok(content) });
            Ok(Box::pin(stream))
        }
    }

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockError: {}", self.0)
        }
    }

    impl std::error::Error for MockError {}

    // -- Test types --

    #[derive(Debug, serde::Deserialize, serde::Serialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    #[derive(Debug, serde::Deserialize, serde::Serialize, PartialEq)]
    struct Country {
        name: String,
        capital: String,
        population: u64,
    }

    // -- Tests --

    #[tokio::test]
    async fn test_with_structured_output_parses_valid_json() {
        let llm = MockChatModel::new(r#"{"name": "Alice", "age": 30}"#);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let result: Person = with_structured_output(&llm, schema, "Tell me about Alice").await.unwrap();
        assert_eq!(result, Person { name: "Alice".to_string(), age: 30 });
    }

    #[tokio::test]
    async fn test_with_structured_output_parses_json_in_markdown_block() {
        let llm = MockChatModel::new(
            "Here is the result:\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```\n"
        );
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let result: Person = with_structured_output(&llm, schema, "Tell me about Bob").await.unwrap();
        assert_eq!(result, Person { name: "Bob".to_string(), age: 25 });
    }

    #[tokio::test]
    async fn test_with_structured_output_rejects_invalid_schema() {
        let llm = MockChatModel::new("{}");
        // Schema is an array, not an object
        let schema = serde_json::json!([1, 2, 3]);

        let result = with_structured_output::<Person, _>(&llm, schema, "test").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            StructuredOutputError::SchemaError(msg) => {
                assert!(msg.contains("must be a JSON object"));
            }
            other => panic!("Expected SchemaError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_with_structured_output_parse_error_on_invalid_json() {
        let llm = MockChatModel::new("This is not JSON at all!");
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });

        let result = with_structured_output::<Person, _>(&llm, schema, "test").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            StructuredOutputError::ParseError(msg) => {
                assert!(msg.contains("Failed to parse LLM response as JSON"));
            }
            other => panic!("Expected ParseError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_with_structured_output_type_mismatch() {
        // Returns valid JSON but age is a string, not an integer
        let llm = MockChatModel::new(r#"{"name": "Alice", "age": "thirty"}"#);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let result = with_structured_output::<Person, _>(&llm, schema, "test").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            StructuredOutputError::ParseError(msg) => {
                assert!(msg.contains("Failed to deserialize JSON into target type"));
            }
            other => panic!("Expected ParseError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_with_structured_output_complex_type() {
        let llm = MockChatModel::new(
            r#"{"name": "France", "capital": "Paris", "population": 67000000}"#
        );
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "capital": {"type": "string"},
                "population": {"type": "integer"}
            },
            "required": ["name", "capital", "population"]
        });

        let result: Country = with_structured_output(&llm, schema, "Tell me about France").await.unwrap();
        assert_eq!(result, Country {
            name: "France".to_string(),
            capital: "Paris".to_string(),
            population: 67000000,
        });
    }

    #[tokio::test]
    async fn test_structured_output_ext_trait() {
        let llm = MockChatModel::new(r#"{"name": "Charlie", "age": 40}"#);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        // Test via the trait method
        let result: Person = llm
            .with_structured_output(schema, "Tell me about Charlie")
            .await
            .unwrap();
        assert_eq!(result, Person { name: "Charlie".to_string(), age: 40 });
    }

    #[test]
    fn test_build_structured_system_prompt() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });

        let prompt = build_structured_system_prompt(&schema);
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("Schema"));
        assert!(prompt.contains("required fields"));
        assert!(prompt.contains("\"type\": \"object\""));
    }

    #[tokio::test]
    async fn test_parse_structured_response_valid() {
        let content = r#"{"name": "Test", "age": 99}"#;
        let result: Person = parse_structured_response(content).await.unwrap();
        assert_eq!(result.name, "Test");
        assert_eq!(result.age, 99);
    }

    #[tokio::test]
    async fn test_parse_structured_response_with_markdown() {
        let content = "```json\n{\"name\": \"Test\", \"age\": 1}\n```";
        let result: Person = parse_structured_response(content).await.unwrap();
        assert_eq!(result.name, "Test");
        assert_eq!(result.age, 1);
    }

    #[test]
    fn test_structured_output_error_display() {
        let err = StructuredOutputError::SchemaError("bad schema".to_string());
        assert_eq!(format!("{}", err), "Schema error: bad schema");

        let err = StructuredOutputError::ParseError("bad parse".to_string());
        assert_eq!(format!("{}", err), "Parse error: bad parse");

        let err = StructuredOutputError::ProviderUnsupported("no func call".to_string());
        assert_eq!(format!("{}", err), "Provider unsupported: no func call");

        let err = StructuredOutputError::LLMError("timeout".to_string());
        assert_eq!(format!("{}", err), "LLM error: timeout");
    }
}
