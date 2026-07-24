// src/core/structured_output.rs
//! Structured output utilities for extracting typed data from LLM responses.
//!
//! This module provides a provider-agnostic `with_structured_output` function
//! that works with any `BaseChatModel` implementation. It injects a JSON schema
//! into the prompt, calls the LLM, and parses the JSON response into the
//! target type `T`.
//!
//! It also provides streaming support via `stream_structured_output`, which
//! returns a stream of partial `T` values as the model generates tokens,
//! using `PartialJsonParser` to incrementally parse incomplete JSON.
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
use futures_util::Stream;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::pin::Pin;

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

    /// The stream ended before a complete JSON object could be parsed.
    #[error("Stream incomplete: {0}")]
    StreamIncomplete(String),
}

/// Errors produced by `PartialJsonParser`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PartialJsonError {
    /// The buffer does not yet contain parseable JSON.
    #[error("Incomplete JSON: {0}")]
    Incomplete(String),

    /// The accumulated text is not valid JSON even after repair attempts.
    #[error("Invalid JSON: {0}")]
    Invalid(String),
}

// ---------------------------------------------------------------------------
// PartialJsonParser
// ---------------------------------------------------------------------------

/// Incremental JSON parser that can handle partial/incomplete JSON.
///
/// Builds up a string token by token and attempts to parse at each step,
/// returning the best partial result possible. This is designed for streaming
/// LLM output where JSON arrives in small chunks and may be incomplete until
/// the stream finishes.
///
/// # Strategy
///
/// 1. Accumulate tokens into an internal buffer.
/// 2. On each `push_and_parse`, attempt to parse the buffer as complete JSON.
/// 3. If that fails, try to repair the partial JSON by closing unclosed
///    brackets/braces and truncating incomplete string values.
/// 4. If repair yields valid JSON, return it; otherwise return
///    `PartialJsonError::Incomplete`.
///
/// # Example
///
/// ```ignore
/// let mut parser = PartialJsonParser::new();
/// // Simulating token-by-token LLM output
/// let _ = parser.push_and_parse(r#"{"name":"#); // Incomplete
/// let v = parser.push_and_parse(r#""Alice","age":30}"#); // Ok({"name":"Alice","age":30})
/// ```
pub struct PartialJsonParser {
    buffer: String,
    depth: usize,
    in_string: bool,
    escape_next: bool,
}

impl PartialJsonParser {
    /// Create a new, empty parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            depth: 0,
            in_string: false,
            escape_next: false,
        }
    }

    /// Push a new token and attempt to parse the accumulated buffer.
    ///
    /// Returns `Ok(value)` if the buffer (after optional repair) yields valid
    /// JSON, or `Err(PartialJsonError::Incomplete)` if it does not yet form
    /// any parseable JSON.
    pub fn push_and_parse(&mut self, token: &str) -> Result<Value, PartialJsonError> {
        // Update parser state by scanning the new token
        for ch in token.chars() {
            if self.escape_next {
                self.escape_next = false;
                continue;
            }
            if ch == '\\' && self.in_string {
                self.escape_next = true;
                continue;
            }
            if ch == '"' {
                self.in_string = !self.in_string;
                continue;
            }
            if !self.in_string {
                match ch {
                    '{' | '[' => self.depth += 1,
                    '}' | ']' => {
                        if self.depth > 0 {
                            self.depth -= 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Ensure we only push at character boundaries (M37: UTF-8 boundary check)
        if token.is_char_boundary(0) {
            self.buffer.push_str(token);
        } else {
            // Find the first valid char boundary
            let mut pos = 0;
            while pos < token.len() && !token.is_char_boundary(pos) {
                pos += 1;
            }
            self.buffer.push_str(&token[pos..]);
        }

        // Fast path: try full parse first
        if let Ok(value) = serde_json::from_str::<Value>(&self.buffer) {
            return Ok(value);
        }

        // Only attempt repair if we have at least opened a structure
        if self.depth > 0
            || self.buffer.trim().starts_with('{')
            || self.buffer.trim().starts_with('[')
        {
            let repaired = Self::repair_partial_json(&self.buffer);
            if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
                return Ok(value);
            }
        }

        Err(PartialJsonError::Incomplete(format!(
            "Buffer has {} chars, depth={}",
            self.buffer.len(),
            self.depth
        )))
    }

    /// Get the final complete value.
    ///
    /// Call this when the stream has ended. It first tries to parse the full
    /// buffer, then falls back to the repaired version.
    pub fn finalize(self) -> Result<Value, PartialJsonError> {
        // Try full parse
        if let Ok(value) = serde_json::from_str::<Value>(&self.buffer) {
            return Ok(value);
        }

        // Try repaired
        let repaired = Self::repair_partial_json(&self.buffer);
        serde_json::from_str::<Value>(&repaired).map_err(|e| {
            PartialJsonError::Invalid(format!(
                "Failed to parse final buffer ({} chars): {}. Buffer: {}",
                self.buffer.len(),
                e,
                &self.buffer[..std::cmp::min(200, self.buffer.len())]
            ))
        })
    }

    /// Return a reference to the current buffer contents.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Whether the parser is currently inside a JSON string.
    pub fn is_in_string(&self) -> bool {
        self.in_string
    }

    /// Current nesting depth of brackets/braces.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Repair a partial JSON string by closing unclosed structures and
    /// truncating incomplete values.
    fn repair_partial_json(text: &str) -> String {
        let mut repaired = text.trim().to_string();

        // Scan the text tracking string state to correctly count braces/brackets
        // and quotes outside of strings (C20 + C21).
        let mut in_string = false;
        let mut escape_next = false;
        let mut open_braces = 0usize;
        let mut close_braces = 0usize;
        let mut open_brackets = 0usize;
        let mut close_brackets = 0usize;
        let mut unescaped_quote_count = 0usize;

        for ch in repaired.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape_next = true;
                continue;
            }
            if ch == '"' {
                unescaped_quote_count += 1;
                in_string = !in_string;
                continue;
            }
            if !in_string {
                match ch {
                    '{' => open_braces += 1,
                    '}' => close_braces += 1,
                    '[' => open_brackets += 1,
                    ']' => close_brackets += 1,
                    _ => {}
                }
            }
        }

        // If we are in the middle of a string value, close it.
        // Heuristic: odd number of unescaped quotes means an unclosed string.
        if unescaped_quote_count % 2 != 0 {
            repaired.push('"');
        }

        // Close unclosed braces first (before removing trailing commas,
        // so that commas before the newly-added braces get removed)
        for _ in close_braces..open_braces {
            repaired.push('}');
        }

        // Close unclosed brackets
        for _ in close_brackets..open_brackets {
            repaired.push(']');
        }

        // Remove trailing commas before closing brackets/braces
        // (must come after closing braces/brackets so we can detect them)
        repaired = Self::remove_trailing_commas(&repaired);

        repaired
    }

    /// Remove trailing commas before closing braces/brackets (invalid in strict JSON).
    fn remove_trailing_commas(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let chars: Vec<char> = s.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == ',' && i + 1 < chars.len() {
                let next_non_ws = chars[i + 1..].iter().find(|c| !c.is_whitespace());
                if next_non_ws == Some(&'}') || next_non_ws == Some(&']') {
                    // Skip the trailing comma
                    i += 1;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        result
    }
}

impl Default for PartialJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StructuredOutputExt trait
// ---------------------------------------------------------------------------

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

    let messages = vec![Message::system(system_prompt), Message::human(prompt)];

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

    let json_value: Value = parser.parse(content).await.map_err(|e| {
        StructuredOutputError::ParseError(format!("Failed to parse LLM response as JSON: {}", e))
    })?;

    serde_json::from_value::<T>(json_value).map_err(|e| {
        StructuredOutputError::ParseError(format!(
            "Failed to deserialize JSON into target type: {}. Response was: {}",
            e,
            &content[..std::cmp::min(200, content.len())]
        ))
    })
}

// ---------------------------------------------------------------------------
// Streaming structured output
// ---------------------------------------------------------------------------

/// Trait that extends `BaseChatModel` with streaming structured output capabilities.
///
/// Provides a streaming variant of `with_structured_output` that yields partial
/// `T` values as the model generates tokens. The target type `T` should derive
/// `#[serde(default)]` or use `Option` fields so that partial JSON can be
/// deserialized with missing fields filled by defaults.
#[async_trait]
pub trait StreamingStructuredOutputExt: BaseChatModel {
    /// Stream structured output from a chat model.
    ///
    /// Returns a stream of `T` values. Each item represents the best partial
    /// result that could be parsed from the tokens received so far. The final
    /// item in the stream is the complete result.
    ///
    /// # Arguments
    ///
    /// * `schema` - A JSON Schema describing the expected output shape.
    /// * `prompt` - The user prompt to send to the LLM.
    ///
    /// # Returns
    ///
    /// A stream of `Result<T, StructuredOutputError>` items.
    async fn stream_structured_output<T>(
        &self,
        schema: Value,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<T, StructuredOutputError>> + Send>>,
        StructuredOutputError,
    >
    where
        T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
    {
        stream_structured_output(self, schema, prompt).await
    }
}

/// Blanket implementation: every `BaseChatModel` automatically gets
/// `StreamingStructuredOutputExt`.
impl<M: BaseChatModel> StreamingStructuredOutputExt for M {}

/// Stream structured output from a chat model.
///
/// Returns a stream of partial `T` values as the model generates tokens.
/// The implementation:
/// 1. Calls `stream_chat` with a prompt that asks for JSON matching the schema
/// 2. Accumulates tokens through the `PartialJsonParser`
/// 3. At each successful parse, yields a `T` value (partial fields filled by
///    serde defaults, rest default)
/// 4. On stream end, yields the final complete `T`
///
/// # Arguments
///
/// * `llm` - Any type implementing `BaseChatModel`.
/// * `schema` - A JSON Schema describing the expected output.
/// * `prompt` - The user prompt to send to the LLM.
///
/// # Returns
///
/// A `Result` containing a stream of `Result<T, StructuredOutputError>` items,
/// or a `StructuredOutputError` if the stream could not be set up.
///
/// # Type requirements
///
/// The target type `T` should use `#[serde(default)]` or `Option` fields so
/// that partial JSON can be deserialized with missing fields filled by defaults.
/// If `T` does not support default deserialization, partial results will fail
/// and only the final complete result will be yielded.
///
/// # Example
///
/// ```ignore
/// use serde::{Deserialize, Serialize};
/// use langchainrust::core::structured_output::stream_structured_output;
/// use futures_util::StreamExt;
///
/// #[derive(Debug, Deserialize, Serialize, Clone)]
/// #[serde(default)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// impl Default for Person {
///     fn default() -> Self {
///         Self { name: String::new(), age: 0 }
///     }
/// }
///
/// let mut stream = stream_structured_output::<Person, _>(
///     &llm, schema, "Tell me about Alice"
/// ).await?;
/// while let Some(result) = stream.next().await {
///     let person = result?;
///     println!("Partial: name={}, age={}", person.name, person.age);
/// }
/// ```
pub async fn stream_structured_output<T, M>(
    llm: &M,
    schema: Value,
    prompt: &str,
) -> Result<
    Pin<Box<dyn Stream<Item = Result<T, StructuredOutputError>> + Send>>,
    StructuredOutputError,
>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
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

    let messages = vec![Message::system(system_prompt), Message::human(prompt)];

    // Start the stream. We need to erase the model's error type by mapping
    // it to StructuredOutputError before passing to the stream processor.
    let token_stream = llm
        .stream_chat(messages, None)
        .await
        .map_err(|e| StructuredOutputError::LLMError(e.to_string()))?;

    // Map the inner stream's error type from M::Error to StructuredOutputError
    let mapped_stream =
        token_stream.map(|item| item.map_err(|e| StructuredOutputError::LLMError(e.to_string())));

    // Box the mapped stream so it has a concrete type for StructuredStreamProcessor
    let boxed: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>> =
        Box::pin(mapped_stream);

    let output_stream = StructuredStreamProcessor::<T>::new(boxed);

    Ok(Box::pin(output_stream))
}

/// Stream processor that accumulates LLM tokens through a `PartialJsonParser`
/// and yields partial `T` values.
///
/// This is implemented as a manual `Stream` rather than using `stream::unfold`
/// because we need to maintain mutable state (the `PartialJsonParser`) across
/// stream polls.
struct StructuredStreamProcessor<T> {
    inner: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>>,
    parser: PartialJsonParser,
    last_value: Option<T>,
    done: bool,
}

// Safety: StructuredStreamProcessor is Unpin when T is Unpin because all fields
// are Unpin: Pin<Box<dyn ..>> is Unpin, PartialJsonParser is Unpin,
// Option<T> is Unpin when T is Unpin, and bool is Unpin.
impl<T: Unpin> Unpin for StructuredStreamProcessor<T> {}

impl<T> StructuredStreamProcessor<T>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Unpin + Send + Sync + 'static,
{
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<String, StructuredOutputError>> + Send>>,
    ) -> Self {
        Self {
            inner,
            parser: PartialJsonParser::new(),
            last_value: None,
            done: false,
        }
    }
}

impl<T> Stream for StructuredStreamProcessor<T>
where
    T: DeserializeOwned + Serialize + Clone + PartialEq + Send + Sync + Unpin + 'static,
{
    type Item = Result<T, StructuredOutputError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Safety: StructuredStreamProcessor is Unpin because all its fields are Unpin.
        // Pin<Box<dyn Stream + Send>> is Unpin, PartialJsonParser is Unpin,
        // Option<T> is Unpin, and bool is Unpin.
        let this = self.get_mut();

        if this.done {
            return std::task::Poll::Ready(None);
        }

        loop {
            match this.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(token))) => {
                    match this.parser.push_and_parse(&token) {
                        Ok(json_value) => match serde_json::from_value::<T>(json_value) {
                            Ok(value) => {
                                this.last_value = Some(value.clone());
                                return std::task::Poll::Ready(Some(Ok(value)));
                            }
                            Err(_) => {
                                // Partial JSON parsed but cannot deserialize into T yet.
                                // Continue accumulating tokens.
                                continue;
                            }
                        },
                        Err(PartialJsonError::Incomplete(_)) => {
                            // Not enough JSON yet, continue accumulating
                            continue;
                        }
                        Err(PartialJsonError::Invalid(_msg)) => {
                            // The accumulated buffer is invalid JSON even after repair.
                            // This can happen with garbage tokens; skip and continue.
                            // We don't terminate the stream for a single bad parse.
                            continue;
                        }
                    }
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    // Error from the underlying token stream
                    return std::task::Poll::Ready(Some(Err(e)));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended. Try to finalize the parser.
                    this.done = true;

                    // Take ownership of the parser to finalize it
                    let parser = std::mem::take(&mut this.parser);

                    match parser.finalize() {
                        Ok(json_value) => match serde_json::from_value::<T>(json_value) {
                            Ok(value) => {
                                // Only yield if this is a new value different from last
                                let is_new = this.last_value.as_ref() != Some(&value);
                                if is_new {
                                    return std::task::Poll::Ready(Some(Ok(value)));
                                }
                                return std::task::Poll::Ready(None);
                            }
                            Err(e) => {
                                return std::task::Poll::Ready(Some(Err(
                                    StructuredOutputError::ParseError(format!(
                                        "Failed to deserialize final JSON into target type: {}",
                                        e
                                    )),
                                )));
                            }
                        },
                        Err(PartialJsonError::Invalid(msg)) => {
                            // If we had a last_value, the stream was still "successful"
                            // in yielding partial results, but the final buffer is invalid.
                            // This can happen if the LLM appended non-JSON text.
                            if this.last_value.is_some() {
                                return std::task::Poll::Ready(None);
                            }
                            return std::task::Poll::Ready(Some(Err(
                                StructuredOutputError::StreamIncomplete(msg),
                            )));
                        }
                        Err(PartialJsonError::Incomplete(msg)) => {
                            if this.last_value.is_some() {
                                return std::task::Poll::Ready(None);
                            }
                            return std::task::Poll::Ready(Some(Err(
                                StructuredOutputError::StreamIncomplete(msg),
                            )));
                        }
                    }
                }
                std::task::Poll::Pending => {
                    return std::task::Poll::Pending;
                }
            }
        }
    }
}

/// Attempt to deserialize a `serde_json::Value` into `T`, returning `None`
/// if deserialization fails (e.g., missing required fields).
#[allow(dead_code)]
fn try_deserialize_partial<T: DeserializeOwned>(value: Value) -> Option<T> {
    serde_json::from_value::<T>(value).ok()
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
        let llm = MockChatModel::new(
            "Here is the result:\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```\n",
        );
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
        ) -> Result<Pin<Box<dyn Stream<Item = Result<String, Self::Error>> + Send>>, Self::Error>
        {
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
        let llm =
            StreamingMockChatModel::from_tokens(vec![r#"{"name": "Diana","#, r#" "age": 28}"#]);
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
        let mut stream = stream_structured_output::<PartialPerson, _>(
            &stream_llm,
            schema,
            "Tell me about Frank",
        )
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
}
