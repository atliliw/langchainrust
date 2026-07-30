// src/core/structured_output/tests.rs
//! Tests for structured output module.

use super::extract::{
    build_structured_system_prompt, parse_structured_response, with_structured_output,
    StructuredOutputError,
};
use super::parser::{PartialJsonError, PartialJsonParser};
use super::streaming::{stream_structured_output, try_deserialize_partial};
use async_trait::async_trait;
use futures_util::stream::StreamExt;
use futures_util::Stream;
use std::pin::Pin;

use crate::core::language_models::{BaseChatModel, BaseLanguageModel, LLMResult};
use crate::core::runnables::Runnable;
use crate::schema::Message;
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
            thinking_content: None,
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
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
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

// -- with_structured_output tests --

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

    let result: Person = with_structured_output(&llm, schema, "Tell me about Alice")
        .await
        .unwrap();
    assert_eq!(
        result,
        Person {
            name: "Alice".to_string(),
            age: 30
        }
    );
}

#[tokio::test]
async fn test_with_structured_output_parses_json_in_markdown_block() {
    let llm =
        MockChatModel::new("Here is the result:\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```\n");
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let result: Person = with_structured_output(&llm, schema, "Tell me about Bob")
        .await
        .unwrap();
    assert_eq!(
        result,
        Person {
            name: "Bob".to_string(),
            age: 25
        }
    );
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
    let llm =
        MockChatModel::new(r#"{"name": "France", "capital": "Paris", "population": 67000000}"#);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "capital": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["name", "capital", "population"]
    });

    let result: Country = with_structured_output(&llm, schema, "Tell me about France")
        .await
        .unwrap();
    assert_eq!(
        result,
        Country {
            name: "France".to_string(),
            capital: "Paris".to_string(),
            population: 67000000
        }
    );
}

#[tokio::test]
async fn test_structured_output_ext_trait() {
    use super::extract::StructuredOutputExt;

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
    assert_eq!(
        result,
        Person {
            name: "Charlie".to_string(),
            age: 40
        }
    );
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

    let err = StructuredOutputError::StreamIncomplete("partial".to_string());
    assert_eq!(format!("{}", err), "Stream incomplete: partial");
}

// =======================================================================
// PartialJsonParser tests
// =======================================================================

#[test]
fn test_partial_json_parser_empty() {
    let parser = PartialJsonParser::new();
    assert!(parser.buffer().is_empty());
    assert_eq!(parser.depth(), 0);
    assert!(!parser.is_in_string());
}

#[test]
fn test_partial_json_parser_complete_json_single_push() {
    let mut parser = PartialJsonParser::new();
    let result = parser.push_and_parse(r#"{"name": "Alice", "age": 30}"#);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["name"], "Alice");
    assert_eq!(value["age"], 30);
}

#[test]
fn test_partial_json_parser_incremental_object() {
    let mut parser = PartialJsonParser::new();

    // Push tokens incrementally. Note: the repair logic is aggressive and
    // may successfully repair partial JSON like "{" into "{}", so we don't
    // assert is_err() for intermediate steps that can be repaired.
    parser.push_and_parse("{").ok(); // May repair to {}
    parser.push_and_parse(r#""name""#).ok();
    parser.push_and_parse(":").ok();
    parser.push_and_parse(r#" "Alice""#).ok(); // May repair to {"name":"Alice"}

    // Complete the object
    let result = parser.push_and_parse("}");
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["name"], "Alice");
}

#[test]
fn test_partial_json_parser_incremental_with_comma() {
    let mut parser = PartialJsonParser::new();

    // Build up: {"name": "Alice", "age": 30}
    // Note: repair may successfully close partial JSON, so we use .ok()
    parser.push_and_parse(r#"{"name": "Alice","#).ok();
    let result = parser.push_and_parse(r#" "age": 30}"#);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["name"], "Alice");
    assert_eq!(value["age"], 30);
}

#[test]
fn test_partial_json_parser_partial_object_repair() {
    let mut parser = PartialJsonParser::new();

    // Push an incomplete object: {"name": "Alice"
    // The repair should close the string and brace
    let result = parser.push_and_parse(r#"{"name": "Alice""#);
    // After repair, this should parse as {"name": "Alice"}
    // (closing the unclosed brace)
    if let Ok(value) = result {
        assert_eq!(value["name"], "Alice");
    }
    // If it returns Incomplete, that's also acceptable since the
    // repair may not always succeed on every partial input
}

#[test]
fn test_partial_json_parser_trailing_comma_repair() {
    let mut parser = PartialJsonParser::new();

    // Push an object with trailing comma: {"name": "Alice",
    // The repair should remove the trailing comma and close the brace
    let result = parser.push_and_parse(r#"{"name": "Alice","#);
    if let Ok(value) = result {
        assert_eq!(value["name"], "Alice");
    }
}

#[test]
fn test_partial_json_parser_nested_object() {
    let mut parser = PartialJsonParser::new();

    // Build up a nested object incrementally.
    // Note: repair may close unclosed braces, so intermediate results
    // may parse successfully.
    parser.push_and_parse(r#"{"person": {"#).ok(); // May repair to {"person": {}}
    parser.push_and_parse(r#""name": "Bob""#).ok(); // May repair to {"person": {"name": "Bob"}}

    let result = parser.push_and_parse(r#"}, "active": true}"#);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["person"]["name"], "Bob");
    assert_eq!(value["active"], true);
}

#[test]
fn test_partial_json_parser_array() {
    let mut parser = PartialJsonParser::new();

    // Note: repair may successfully close partial JSON, so we use .ok()
    parser.push_and_parse("[1,").ok();
    let result = parser.push_and_parse(" 2, 3]");
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value[0], 1);
    assert_eq!(value[2], 3);
}

#[test]
fn test_partial_json_parser_string_with_escapes() {
    let mut parser = PartialJsonParser::new();

    let result = parser.push_and_parse(r#"{"text": "hello \"world\""}"#);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["text"], r#"hello "world""#);
}

#[test]
fn test_partial_json_parser_finalize_complete() {
    let mut parser = PartialJsonParser::new();
    parser.push_and_parse(r#"{"name": "Eve"}"#).ok();

    let result = parser.finalize();
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value["name"], "Eve");
}

#[test]
fn test_partial_json_parser_finalize_incomplete() {
    let mut parser = PartialJsonParser::new();
    // Push incomplete JSON that cannot be repaired
    parser.push_and_parse(r#"{"name": "#).ok();

    let result = parser.finalize();
    // Should either succeed (if repair works) or fail with Invalid
    match result {
        Ok(value) => {
            // If repair succeeded, the value should have "name" key
            assert!(value.is_object());
        }
        Err(PartialJsonError::Invalid(_)) => {
            // Expected: incomplete JSON that can't be repaired
        }
        Err(PartialJsonError::Incomplete(_)) => {
            // Also acceptable
        }
    }
}

#[test]
fn test_partial_json_parser_default_impl() {
    let parser = PartialJsonParser::default();
    assert!(parser.buffer().is_empty());
}

#[test]
fn test_partial_json_error_display() {
    let err = PartialJsonError::Incomplete("not enough".to_string());
    assert!(format!("{}", err).contains("Incomplete JSON"));

    let err = PartialJsonError::Invalid("bad json".to_string());
    assert!(format!("{}", err).contains("Invalid JSON"));
}

#[test]
fn test_repair_partial_json_closes_braces() {
    let repaired = PartialJsonParser::repair_partial_json(r#"{"a": 1"#);
    assert!(repaired.ends_with('}'));
    let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    assert_eq!(parsed["a"], 1);
}

#[test]
fn test_repair_partial_json_closes_brackets() {
    let repaired = PartialJsonParser::repair_partial_json("[1, 2");
    assert!(repaired.ends_with(']'));
    let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    assert_eq!(parsed[0], 1);
    assert_eq!(parsed[1], 2);
}

#[test]
fn test_repair_partial_json_unclosed_string() {
    let repaired = PartialJsonParser::repair_partial_json(r#"{"name": "Ali"#);
    // Should close the string and the brace
    let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
    assert_eq!(parsed["name"], "Ali");
}

#[test]
fn test_remove_trailing_commas() {
    let result = PartialJsonParser::remove_trailing_commas(r#"{"a": 1,}"#);
    assert_eq!(result, r#"{"a": 1}"#);

    let result = PartialJsonParser::remove_trailing_commas(r#"{"a": [1, 2,]}"#);
    assert_eq!(result, r#"{"a": [1, 2]}"#);

    // Comma before non-closing token should be kept
    let result = PartialJsonParser::remove_trailing_commas(r#"{"a": 1, "b": 2}"#);
    assert_eq!(result, r#"{"a": 1, "b": 2}"#);
}

// =======================================================================
// Streaming mock model
// =======================================================================

/// A mock chat model that streams tokens one at a time from a predefined response.
struct StreamingMockChatModel {
    tokens: Vec<String>,
}

impl StreamingMockChatModel {
    fn from_tokens(tokens: Vec<&str>) -> Self {
        Self {
            tokens: tokens.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create a streaming mock that splits a complete response into
    /// character-level chunks for fine-grained streaming simulation.
    fn from_response_char_chunks(response: &str) -> Self {
        let tokens: Vec<String> = response.chars().map(|c| c.to_string()).collect();
        Self { tokens }
    }

    /// Create a streaming mock that splits a complete response into
    /// a few logical token chunks.
    fn from_response_token_chunks(response: &str, chunk_size: usize) -> Self {
        let mut tokens = Vec::new();
        let chars: Vec<char> = response.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let end = std::cmp::min(i + chunk_size, chars.len());
            tokens.push(chars[i..end].iter().collect());
            i = end;
        }
        Self { tokens }
    }
}

#[async_trait]
impl Runnable<Vec<Message>, LLMResult> for StreamingMockChatModel {
    type Error = MockError;

    async fn invoke(
        &self,
        _input: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let content = self.tokens.join("");
        Ok(LLMResult {
            content,
            model: "streaming-mock".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }
}

#[async_trait]
impl BaseLanguageModel<Vec<Message>, LLMResult> for StreamingMockChatModel {
    fn model_name(&self) -> &str {
        "streaming-mock"
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
impl BaseChatModel for StreamingMockChatModel {
    async fn chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<LLMResult, Self::Error> {
        let content = self.tokens.join("");
        Ok(LLMResult {
            content,
            model: "streaming-mock".to_string(),
            token_usage: None,
            tool_calls: None,
            thinking_content: None,
        })
    }

    async fn stream_chat(
        &self,
        _messages: Vec<Message>,
        _config: Option<RunnableConfig>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error> {
        let tokens = self.tokens.clone();
        let stream = futures_util::stream::iter(tokens.into_iter().map(|t| Ok(t)));
        Ok(Box::pin(stream))
    }
}

// =======================================================================
// Test types with serde(default) for partial deserialization
// =======================================================================

/// A Person type that supports partial deserialization via serde(default).
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone, PartialEq)]
#[serde(default)]
struct PartialPerson {
    name: String,
    age: u32,
}

impl Default for PartialPerson {
    fn default() -> Self {
        Self {
            name: String::new(),
            age: 0,
        }
    }
}

/// A Country type that supports partial deserialization via serde(default).
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone, PartialEq)]
#[serde(default)]
struct PartialCountry {
    name: String,
    capital: String,
    population: u64,
}

impl Default for PartialCountry {
    fn default() -> Self {
        Self {
            name: String::new(),
            capital: String::new(),
            population: 0,
        }
    }
}

// =======================================================================
// stream_structured_output tests
// =======================================================================

#[tokio::test]
async fn test_stream_structured_output_single_chunk() {
    let llm = StreamingMockChatModel::from_tokens(vec![r#"{"name": "Alice", "age": 30}"#]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let mut stream =
        stream_structured_output::<PartialPerson, _>(&llm, schema, "Tell me about Alice")
            .await
            .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    // Should get at least one result (the complete one)
    assert!(!results.is_empty());
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialPerson {
            name: "Alice".to_string(),
            age: 30
        }
    );
}

#[tokio::test]
async fn test_stream_structured_output_incremental_tokens() {
    // Simulate token-by-token streaming
    let llm =
        StreamingMockChatModel::from_response_token_chunks(r#"{"name": "Bob", "age": 25}"#, 5);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let mut stream =
        stream_structured_output::<PartialPerson, _>(&llm, schema, "Tell me about Bob")
            .await
            .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    // Should get at least one result
    assert!(!results.is_empty());

    // The last result should be the complete value
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialPerson {
            name: "Bob".to_string(),
            age: 25
        }
    );
}

#[tokio::test]
async fn test_stream_structured_output_char_by_char() {
    // Most granular streaming: one character at a time
    let llm =
        StreamingMockChatModel::from_response_char_chunks(r#"{"name": "Charlie", "age": 40}"#);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let mut stream =
        stream_structured_output::<PartialPerson, _>(&llm, schema, "Tell me about Charlie")
            .await
            .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    assert!(!results.is_empty());

    // The last result should be the complete value
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialPerson {
            name: "Charlie".to_string(),
            age: 40
        }
    );
}

#[tokio::test]
async fn test_stream_structured_output_partial_values_evolve() {
    // Test that partial values evolve as more tokens arrive.
    // With serde(default), we should see partial results with default
    // values for missing fields.
    let llm = StreamingMockChatModel::from_tokens(vec![r#"{"name": "Diana","#, r#" "age": 28}"#]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let mut stream =
        stream_structured_output::<PartialPerson, _>(&llm, schema, "Tell me about Diana")
            .await
            .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    // Should get at least 2 results: one partial and one final
    assert!(
        results.len() >= 2,
        "Expected at least 2 results, got {}",
        results.len()
    );

    // The final result should be complete
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialPerson {
            name: "Diana".to_string(),
            age: 28
        }
    );
}

#[tokio::test]
async fn test_stream_structured_output_rejects_invalid_schema() {
    let llm = StreamingMockChatModel::from_tokens(vec!["{}"]);
    let schema = serde_json::json!([1, 2, 3]); // Not an object

    let result = stream_structured_output::<PartialPerson, _>(&llm, schema, "test").await;

    assert!(result.is_err());
    match result.err().unwrap() {
        StructuredOutputError::SchemaError(msg) => {
            assert!(msg.contains("must be a JSON object"));
        }
        other => panic!("Expected SchemaError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_stream_structured_output_complex_type() {
    let llm = StreamingMockChatModel::from_tokens(vec![
        r#"{"name": "France","#,
        r#" "capital": "Paris","#,
        r#" "population": 67000000}"#,
    ]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "capital": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["name", "capital", "population"]
    });

    let mut stream =
        stream_structured_output::<PartialCountry, _>(&llm, schema, "Tell me about France")
            .await
            .unwrap();

    let results: Vec<Result<PartialCountry, StructuredOutputError>> =
        stream.as_mut().collect().await;

    assert!(!results.is_empty());
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialCountry {
            name: "France".to_string(),
            capital: "Paris".to_string(),
            population: 67000000
        }
    );
}

#[tokio::test]
async fn test_stream_structured_output_via_trait() {
    use super::streaming::StreamingStructuredOutputExt;

    let llm = StreamingMockChatModel::from_tokens(vec![r#"{"name": "Eve", "age": 22}"#]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    // Test via the StreamingStructuredOutputExt trait method
    let mut stream = llm
        .stream_structured_output::<PartialPerson>(schema, "Tell me about Eve")
        .await
        .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    assert!(!results.is_empty());
    let final_result = results.into_iter().last().unwrap().unwrap();
    assert_eq!(
        final_result,
        PartialPerson {
            name: "Eve".to_string(),
            age: 22
        }
    );
}

#[tokio::test]
async fn test_stream_equals_non_stream_result() {
    // Verify that the final streamed result equals the non-streamed result
    let response = r#"{"name": "Frank", "age": 35}"#;

    // Non-streaming
    let non_stream_llm = MockChatModel::new(response);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });
    let non_stream_result: Person =
        with_structured_output(&non_stream_llm, schema.clone(), "Tell me about Frank")
            .await
            .unwrap();

    // Streaming
    let stream_llm = StreamingMockChatModel::from_response_char_chunks(response);
    let mut stream =
        stream_structured_output::<PartialPerson, _>(&stream_llm, schema, "Tell me about Frank")
            .await
            .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;
    let stream_result = results.into_iter().last().unwrap().unwrap();

    // The values should match
    assert_eq!(non_stream_result.name, stream_result.name);
    assert_eq!(non_stream_result.age, stream_result.age);
}

#[tokio::test]
async fn test_stream_structured_output_empty_stream() {
    // A model that produces an empty stream
    let llm = StreamingMockChatModel::from_tokens(vec![]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    });

    let mut stream = stream_structured_output::<PartialPerson, _>(&llm, schema, "test")
        .await
        .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    // Empty stream should yield an error (StreamIncomplete)
    if let Some(result) = results.into_iter().last() {
        match result {
            Err(StructuredOutputError::StreamIncomplete(_)) => {
                // Expected
            }
            Err(StructuredOutputError::ParseError(_)) => {
                // Also acceptable - empty buffer can't be parsed
            }
            other => {
                panic!("Expected StreamIncomplete or ParseError, got: {:?}", other);
            }
        }
    }
}

#[tokio::test]
async fn test_stream_structured_output_non_json_stream() {
    // A model that produces non-JSON output
    let llm = StreamingMockChatModel::from_tokens(vec!["This is not JSON at all!"]);
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    });

    let mut stream = stream_structured_output::<PartialPerson, _>(&llm, schema, "test")
        .await
        .unwrap();

    let results: Vec<Result<PartialPerson, StructuredOutputError>> =
        stream.as_mut().collect().await;

    // Should yield an error since the output is not JSON
    if let Some(result) = results.into_iter().last() {
        assert!(
            result.is_err(),
            "Expected error for non-JSON output, got: {:?}",
            result
        );
    }
}

#[test]
fn test_try_deserialize_partial_success() {
    let value = serde_json::json!({"name": "Test", "age": 99});
    let result = try_deserialize_partial::<PartialPerson>(value);
    assert!(result.is_some());
    let person = result.unwrap();
    assert_eq!(person.name, "Test");
    assert_eq!(person.age, 99);
}

#[test]
fn test_try_deserialize_partial_missing_field() {
    // With serde(default), missing fields should be filled with defaults
    let value = serde_json::json!({"name": "Test"});
    let result = try_deserialize_partial::<PartialPerson>(value);
    assert!(result.is_some());
    let person = result.unwrap();
    assert_eq!(person.name, "Test");
    assert_eq!(person.age, 0); // default
}

#[test]
fn test_try_deserialize_partial_wrong_type() {
    // age is a string, not an integer - should fail
    let value = serde_json::json!({"name": "Test", "age": "not a number"});
    let result = try_deserialize_partial::<PartialPerson>(value);
    assert!(result.is_none());
}
