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
    pub fn parse(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.result.content)
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
