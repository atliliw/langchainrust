// src/core/tools/structured_output.rs
//! Structured output utilities

use crate::core::language_models::LLMResult;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::marker::PhantomData;

pub struct StructuredOutput<T> {
    result: LLMResult,
    schema: serde_json::Value,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> StructuredOutput<T> {
    pub fn new(result: LLMResult) -> Self {
        use schemars::schema_for;
        let schema = serde_json::to_value(schema_for!(T)).unwrap_or(serde_json::Value::Null);
        Self {
            result,
            schema,
            _phantom: PhantomData,
        }
    }

    pub fn parse(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.result.content)
    }

    pub fn raw_content(&self) -> &str {
        &self.result.content
    }

    pub fn schema(&self) -> &serde_json::Value {
        &self.schema
    }
}
