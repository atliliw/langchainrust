// src/core/structured_output/extract.rs
//! Structured output extraction from LLM responses.

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

    /// The stream ended before a complete JSON object could be parsed.
    #[error("Stream incomplete: {0}")]
    StreamIncomplete(String),
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
pub(crate) fn build_structured_system_prompt(schema: &Value) -> String {
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
pub(crate) async fn parse_structured_response<
    T: DeserializeOwned + Serialize + Send + Sync + 'static,
>(
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
