//! Provider-agnostic raw JSON Schema constraint for structured output.
//!
//! [`JsonSchemaOutput`] carries a schema the caller already holds as a
//! `serde_json::Value` (hand-written, loaded from a file, fetched from another
//! service) so it can be handed straight to a provider's native
//! schema-constrained decoding without first defining a Rust type that
//! implements `schemars::JsonSchema`.
//!
//! Provider mapping:
//!
//! - **OpenAI** — `response_format: {"type":"json_schema","json_schema":
//!   {"name","schema","strict"}}`. When `strict` is on, the schema is
//!   normalized first (`additionalProperties: false`, every property listed in
//!   `required`) because OpenAI rejects a strict schema that omits them.
//! - **Anthropic** — `output_config.format: {"type":"json_schema","schema":…}`.
//!   Anthropic has no `name`/`strict` fields, so those are dropped by the
//!   mapping.
//!
//! Types that do implement `JsonSchema` should keep using
//! `with_json_schema_output::<T>()`; this type exists for the dynamic case.

/// A raw JSON Schema used to constrain a model's structured output.
///
/// ```
/// use lc_core::structured_output::JsonSchemaOutput;
/// use serde_json::json;
///
/// let output = JsonSchemaOutput::new(
///     "person",
///     json!({"type": "object", "properties": {"name": {"type": "string"}}}),
/// );
/// assert_eq!(output.name(), "person");
/// assert!(output.is_strict());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchemaOutput {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

impl JsonSchemaOutput {
    /// Creates a schema constraint from an already-parsed JSON Schema.
    ///
    /// `strict` defaults to `true`: engine-side schema enforcement is the
    /// point of passing a schema, and both OpenAI and Anthropic treat
    /// schema-constrained decoding as best-effort unless asked for the
    /// strict form. Use [`JsonSchemaOutput::with_strict`] to opt out.
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            schema,
            strict: true,
        }
    }

    /// Creates a schema constraint from raw JSON text (a schema kept as a
    /// string, e.g. read from a file or a config blob).
    ///
    /// # Errors
    /// Returns the `serde_json` error when `json` is not valid JSON.
    pub fn from_json_str(name: impl Into<String>, json: &str) -> Result<Self, serde_json::Error> {
        Ok(Self::new(name, serde_json::from_str(json)?))
    }

    /// Sets whether the provider should enforce the schema strictly.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Schema name (OpenAI requires `^[a-zA-Z0-9_-]+$`, 1-64 chars).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The JSON Schema itself.
    pub fn schema(&self) -> &serde_json::Value {
        &self.schema
    }

    /// Whether strict enforcement was requested.
    pub fn is_strict(&self) -> bool {
        self.strict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_defaults_to_strict() {
        let output = JsonSchemaOutput::new("out", json!({"type": "object"}));
        assert_eq!(output.name(), "out");
        assert_eq!(output.schema(), &json!({"type": "object"}));
        assert!(output.is_strict());
    }

    #[test]
    fn with_strict_can_be_disabled() {
        let output = JsonSchemaOutput::new("out", json!({"type": "object"})).with_strict(false);
        assert!(!output.is_strict());
    }

    #[test]
    fn from_json_str_parses_a_raw_schema() {
        let output =
            JsonSchemaOutput::from_json_str("out", r#"{"type":"object","properties":{}}"#).unwrap();
        assert_eq!(
            output.schema(),
            &json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn from_json_str_rejects_invalid_json() {
        assert!(JsonSchemaOutput::from_json_str("out", "{not json").is_err());
    }
}
