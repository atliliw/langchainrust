// src/core/tools/structured_output.rs
//! Structured output utilities for type-safe LLM responses.

use crate::core::language_models::LLMResult;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

/// Wrapper for parsing structured JSON output from LLM responses.
pub struct StructuredOutput<T> {
    result: LLMResult,
    schema: serde_json::Value,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> StructuredOutput<T> {
    /// Creates a new StructuredOutput from an LLM result.
    pub fn new(result: LLMResult) -> Self {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T)).unwrap_or(serde_json::Value::Null);
        Self {
            result,
            schema,
            _phantom: PhantomData,
        }
    }

    /// Parses the LLM response into the target type.
    ///
    /// Handles markdown code blocks (```json ... ```) wrapping the JSON,
    /// which LLMs commonly produce.
    pub fn parse(&self) -> Result<T, serde_json::Error> {
        let content = self.result.content.trim();

        // Strip markdown code block wrapping (L5)
        let json_str = if content.starts_with("```") {
            // Find the end of the opening fence (may include language tag like ```json)
            let after_fence = if let Some(newline_pos) = content.find('\n') {
                &content[newline_pos + 1..]
            } else {
                content
            };
            // Find the closing fence
            if let Some(end_pos) = after_fence.find("```") {
                after_fence[..end_pos].trim()
            } else {
                after_fence.trim()
            }
        } else {
            content
        };

        serde_json::from_str(json_str)
    }

    /// Returns the raw response content.
    pub fn raw_content(&self) -> &str {
        &self.result.content
    }

    /// Returns the JSON schema for the output type.
    pub fn schema(&self) -> &serde_json::Value {
        &self.schema
    }
}
